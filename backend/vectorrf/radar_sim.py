"""Waveform-level point-target radar with physical array channels and arbitrary TX."""
from typing import Literal
from functools import lru_cache
import numpy as np
from scipy import signal,fft
from pydantic import BaseModel,Field,ConfigDict,model_validator
from .programs import WaveProgram,program_samples
from .pipeline import Pipeline,process_chain,preview
from .radar_studio import RadarStudioConfig,generate_radar
from .models import TxConfig,SpectrumConfig
from .signal import generate
from .spectrum import analyze
from .engine import reduce_spectrum
from .radar_processing import RadarProcessing,range_doppler
C=299792458.

class ArrayModel(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    geometry:Literal['linear','planar','custom']='planar'
    columns:int=Field(default=4,ge=1,le=16)
    rows:int=Field(default=2,ge=1,le=8)
    spacing_lambda:float=Field(default=.5,gt=0,le=4)
    positions:list[list[float]]=Field(default_factory=lambda:[[0,-.5,0],[0,0,0],[0,.5,0]],min_length=1,max_length=32)
    weights_i:list[float]=Field(default_factory=list,max_length=32)
    weights_q:list[float]=Field(default_factory=list,max_length=32)
    element:Literal['isotropic','cosine','gaussian','table']='cosine'
    beamwidth_deg:float=Field(default=100,gt=0,le=360)
    cosine_power:float=Field(default=1,ge=0,le=20)
    pattern_angles:list[float]=Field(default_factory=lambda:[-90,-45,0,45,90],min_length=2,max_length=361)
    pattern_db:list[float]=Field(default_factory=lambda:[-40,-6,0,-6,-40],min_length=2,max_length=361)
    steer_az_deg:float=Field(default=0,ge=-90,le=90)
    steer_el_deg:float=Field(default=0,ge=-60,le=60)
    scan_deg_s:float=Field(default=0,ge=-180,le=180)
    beamformer:Literal['conventional','mvdr']='conventional'
    @model_validator(mode='after')
    def validate_array(self):
        if self.geometry!='custom' and self.columns*(self.rows if self.geometry=='planar' else 1)>32:raise ValueError('Array supports up to 32 channels')
        if any(len(p)!=3 or not np.isfinite(p).all() for p in self.positions):raise ValueError('Each position needs x,y,z in wavelengths')
        if len(self.pattern_angles)!=len(self.pattern_db) or any(b<=a for a,b in zip(self.pattern_angles,self.pattern_angles[1:])):raise ValueError('Pattern angles must increase and match dB values')
        n=len(self.positions) if self.geometry=='custom' else self.columns*(self.rows if self.geometry=='planar' else 1)
        if self.weights_i and len(self.weights_i)!=n:raise ValueError('Custom weights require one I value per element')
        if self.weights_q and len(self.weights_q)!=n:raise ValueError('Custom weights require one Q value per element')
        return self

class SceneTarget(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    name:str=Field(default='Target',max_length=48)
    position:list[float]=Field(default_factory=lambda:[3000,600,200],min_length=3,max_length=3)
    velocity:list[float]=Field(default_factory=lambda:[-30,15,0],min_length=3,max_length=3)
    acceleration:list[float]=Field(default_factory=lambda:[0,0,0],min_length=3,max_length=3)
    rcs_m2:float=Field(default=1,gt=0,le=1e6)
    gain_db:float=Field(default=-6,ge=-100,le=30)
    fluctuation:Literal['none','swerling1']='none'

class RadarExperiment(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    source:Literal['legacy','program','ofdm']='program'
    legacy:RadarStudioConfig=Field(default_factory=RadarStudioConfig)
    program:WaveProgram=Field(default_factory=WaveProgram)
    ofdm:TxConfig=Field(default_factory=lambda:TxConfig(waveform='ofdm',snr_db=None))
    sample_rate:float=Field(default=1_000_000,gt=1000,le=10_000_000)
    samples:int=Field(default=4096,ge=512,le=32768)
    capture_id:int=Field(default=0,ge=0,le=2147483647)
    time_s:float=Field(default=0,ge=0,le=86400)
    carrier_hz:float=Field(default=3e9,ge=1e6,le=100e9)
    targets:list[SceneTarget]=Field(default_factory=lambda:[SceneTarget()],max_length=12)
    antenna:ArrayModel=Field(default_factory=ArrayModel)
    processing:RadarProcessing=Field(default_factory=RadarProcessing)
    pipeline:Pipeline=Field(default_factory=Pipeline)
    power_model:Literal['relative','radar-equation']='relative'
    power_w:float=Field(default=1000,gt=0,le=1e9)
    tx_gain_db:float=Field(default=20,ge=-30,le=80)
    rx_gain_db:float=Field(default=20,ge=-30,le=80)
    noise_figure_db:float=Field(default=6,ge=0,le=40)
    noise_db:float=Field(default=-45,ge=-200,le=0)
    loss_db:float=Field(default=0,ge=0,le=100)
    clutter_db:float=Field(default=-100,ge=-150,le=0)
    leakage_db:float=Field(default=-100,ge=-150,le=0)
    max_range_m:float=Field(default=15000,gt=0,le=1e7)
    max_velocity_mps:float=Field(default=100,gt=0,le=3000)
    doppler_bins:int=Field(default=41,ge=9,le=81)
    cfar_guard:int=Field(default=3,ge=1,le=20)
    cfar_train:int=Field(default=12,ge=2,le=64)
    cfar_pfa:float=Field(default=.001,gt=0,lt=.5)
    seed:int=7

def directions(az,el):
    a=np.deg2rad(az);e=np.deg2rad(el)
    return np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)]).T

def positions(a):
    if a.geometry=='custom':return np.asarray(a.positions)
    yy,zz=np.meshgrid(np.arange(a.columns)-(a.columns-1)/2,np.arange(a.rows if a.geometry=='planar' else 1)-((a.rows if a.geometry=='planar' else 1)-1)/2)
    return np.c_[np.zeros(yy.size),yy.ravel()*a.spacing_lambda,zz.ravel()*a.spacing_lambda]

def element_voltage(a,az,el):
    if a.element=='isotropic':return np.ones_like(np.asarray(az),float)
    if a.element=='cosine':return np.maximum(0,np.cos(np.deg2rad(az))*np.cos(np.deg2rad(el)))**a.cosine_power
    if a.element=='gaussian':return np.exp(-2*np.log(2)*(np.asarray(az)**2+np.asarray(el)**2)/a.beamwidth_deg**2)
    return 10**(np.interp(az,a.pattern_angles,a.pattern_db)/20)

def array_weights(a,time_s):
    pos=positions(a);az=60*np.sin(np.deg2rad(a.scan_deg_s)*time_s) if a.scan_deg_s else a.steer_az_deg
    steer=np.exp(2j*np.pi*(pos@directions(az,a.steer_el_deg)))
    taper=np.asarray(a.weights_i or [1]*len(pos))+1j*np.asarray(a.weights_q or [0]*len(pos))
    w=steer*taper
    if not np.any(abs(w)>1e-12):raise ValueError('At least one antenna weight must be nonzero')
    return pos,w/max(np.sum(abs(w)),1e-12),float(az)

@lru_cache(maxsize=8)
def legacy_record(config_json):
    cfg=RadarStudioConfig.model_validate_json(config_json)
    first=generate_radar(cfg)
    duration=first.metadata.get('logical_sequence_s',cfg.duration_s)
    if duration>cfg.duration_s:first=generate_radar(cfg.model_copy(update={'duration_s':duration}))
    length=max(1,int(round(duration*cfg.sample_rate)))
    return first.samples[:length],cfg.sample_rate

@lru_cache(maxsize=4)
def ofdm_record(config_json):
    return generate(TxConfig.model_validate_json(config_json)).samples

def transmitter(req,start,n):
    if req.source=='program':return program_samples(req.program,req.sample_rate,n,start,req.capture_id,req.seed+req.capture_id)
    if req.source=='legacy':base,rate=legacy_record(req.legacy.model_dump_json())
    else:
        cpv=req.ofdm.ofdm_design.cp_lengths or [req.ofdm.ofdm_cp]
        frame_samples=sum(req.ofdm.ofdm_fft+cpv[i%len(cpv)] for i in range(req.ofdm.ofdm_design.symbols))
        cfg=req.ofdm.model_copy(update={'sample_rate':req.sample_rate,'duration':frame_samples/req.sample_rate,'snr_db':None})
        base=ofdm_record(cfg.model_dump_json());rate=req.sample_rate
    p=((start+np.arange(n)/req.sample_rate)*rate)%len(base)
    return np.interp(p,np.arange(len(base)+1),np.r_[base.real,base.real[0]])+1j*np.interp(p,np.arange(len(base)+1),np.r_[base.imag,base.imag[0]])

def ca_cfar(power,guard,train,pfa):
    n=2*train;kernel=np.ones(2*(train+guard)+1);kernel[train:train+2*guard+1]=0
    threshold=np.convolve(power,kernel/kernel.sum(),mode='same')*n*(pfa**(-1/n)-1)
    valid=np.zeros(len(power),bool);valid[train+guard:len(power)-train-guard]=True
    peaks,_=signal.find_peaks(power)
    detected=[int(i) for i in peaks if valid[i] and power[i]>threshold[i]]
    return threshold,sorted(detected,key=lambda i:power[i],reverse=True)[:24]

@lru_cache(maxsize=2)
def doppler_mixers(n,fs,doppler):
    return np.exp(-2j*np.pi*np.asarray(doppler)[:,None]*np.arange(n)[None,:]/fs)

@lru_cache(maxsize=8)
def angle_steering(pos,el_count):
    az,el=np.meshgrid(np.linspace(-85,85,171),np.linspace(-45,45,31) if el_count>1 else [0.])
    return np.exp(2j*np.pi*(directions(az.ravel(),el.ravel())@np.asarray(pos).T))

def simulate(req:RadarExperiment):
    fs=req.sample_rate;n=req.samples;lam=C/req.carrier_hz;t=np.arange(n)/fs
    tx=transmitter(req,req.time_s,n);pos,w,steer_az=array_weights(req.antenna,req.time_s)
    channels=np.zeros((len(pos),n),complex);truth=[];warnings=[]
    rng=np.random.default_rng((req.seed+req.capture_id*104729)%(2**32))
    for target in req.targets:
        p=np.asarray(target.position)+np.asarray(target.velocity)*req.time_s+.5*np.asarray(target.acceleration)*req.time_s**2
        vel=np.asarray(target.velocity)+np.asarray(target.acceleration)*req.time_s
        r=max(float(np.linalg.norm(p)),1);u=p/r;vr=float(vel@u);az=float(np.rad2deg(np.arctan2(p[1],p[0])));el=float(np.rad2deg(np.arcsin(np.clip(u[2],-1,1))))
        steering=np.exp(2j*np.pi*(pos@u));element=float(element_voltage(req.antenna,az,el));tx_pattern=abs(np.vdot(w,steering))*element
        if req.power_model=='radar-equation':
            power=req.power_w*10**((req.tx_gain_db+req.rx_gain_db-req.loss_db)/10)*lam**2*target.rcs_m2/((4*np.pi)**3*r**4)
            gain=np.sqrt(power)*10**(target.gain_db/20)
        else:gain=10**(target.gain_db/20)*np.sqrt(target.rcs_m2)*(3000/r)**2*10**(-req.loss_db/20)
        if target.fluctuation=='swerling1':gain*=np.sqrt(rng.exponential())
        echo=transmitter(req,req.time_s-2*r/C,n)*np.exp(-4j*np.pi*(r+vr*t)/lam)
        channels+=steering[:,None]*echo[None,:]*gain*tx_pattern*element
        truth.append({'name':target.name,'position':p.tolist(),'range_m':r,'velocity_mps':vr,'az_deg':az,'el_deg':el,'echo_power_db':float(20*np.log10(max(abs(gain*tx_pattern*element),1e-30)))})
        if r>req.max_range_m:warnings.append(target.name+': outside displayed range')
        if abs(vr)>req.max_velocity_mps:warnings.append(target.name+': outside Doppler search interval')
    if req.power_model=='radar-equation':noise_power=1.380649e-23*290*fs*10**(req.noise_figure_db/10)
    else:noise_power=10**(req.noise_db/10)
    channels+=np.sqrt(noise_power/2)*(rng.normal(size=channels.shape)+1j*rng.normal(size=channels.shape))
    channels+=10**(req.leakage_db/20)*tx[None,:]
    clutter=(rng.normal(size=n)+1j*rng.normal(size=n))*10**(req.clutter_db/20)/np.sqrt(2)
    channels+=clutter[None,:]
    if req.antenna.beamformer=='mvdr' and len(pos)>1:
        cov=channels@channels.conj().T/n;load=max(float(np.trace(cov).real/len(pos)),1e-20)*.01
        a=np.exp(2j*np.pi*(pos@directions(steer_az,req.antenna.steer_el_deg)))
        iw=np.linalg.solve(cov+np.eye(len(pos))*load,a);w=iw/max(float(np.vdot(a,iw).real),1e-20)
    beam=np.einsum("i,ij->j",np.conj(w),channels)
    processed,pfs,stages,probe=process_chain(beam,fs,req.pipeline,req.time_s,req.capture_id)
    if pfs!=fs or len(processed)!=n:
        tt=np.arange(len(processed))/pfs
        ref=np.interp(tt,t,tx.real)+1j*np.interp(tt,t,tx.imag)
    else:ref=tx
    if len(processed)>65536:raise ValueError('Radar processing output is limited to 65536 samples per capture')
    processor_info={}
    if req.processing.method!='correlation':
        rd,range_axis,velocity,processor_info=range_doppler(processed,ref,pfs,req.time_s,req.carrier_hz,req.processing,req.max_range_m,req.max_velocity_mps,req.doppler_bins)
        maxlag=len(range_axis);doppler=-2*velocity/lam
    else:
        nout=len(processed);maxlag=min(nout//2,max(2,int(2*req.max_range_m/C*pfs)))
        velocity=np.linspace(-req.max_velocity_mps,req.max_velocity_mps,req.doppler_bins)
        doppler=-2*velocity/lam;ref_energy=max(float(np.vdot(ref,ref).real),1e-20)
        nout=len(processed);tt=np.arange(nout)/pfs
        size=fft.next_fast_len(nout+len(ref)-1)
        reference_fft=fft.fft(np.conj(ref[::-1]),size)
        mixers=doppler_mixers(nout,pfs,tuple(doppler))
        rd=np.empty((len(velocity),maxlag),complex)
        for begin in range(0,len(velocity),8):
            batch=processed[None,:]*mixers[begin:begin+8]
            values=fft.ifft(fft.fft(batch,size,axis=1)*reference_fft,axis=1)
            rd[begin:begin+8]=values[:,len(ref)-1:len(ref)-1+maxlag]/ref_energy
        range_axis=np.arange(maxlag)*C/(2*pfs)
        keep=range_axis>=req.processing.min_range_m
        if not np.any(keep):raise ValueError('Range gate contains no correlation bins')
        rd=rd[:,keep];range_axis=range_axis[keep];maxlag=len(range_axis)
    power=abs(rd)**2;peak=np.unravel_index(np.argmax(power),power.shape)
    profile=np.max(power,axis=0);threshold,detections=ca_cfar(profile,req.cfar_guard,req.cfar_train,req.cfar_pfa) if maxlag>2*(req.cfar_train+req.cfar_guard)+1 else (np.zeros(maxlag),[])
    if req.processing.detector=='manual':
        threshold=np.full(maxlag,10**(req.processing.threshold_db/10))
        found,_=signal.find_peaks(profile);detections=sorted([int(i) for i in found if profile[i]>threshold[i]],key=lambda i:profile[i],reverse=True)[:24]
    # Direction finding is derived from array-channel samples at the measured range/Doppler peak.
    delay=2*range_axis[peak[1]]/C;lag=int(round(delay*fs));fd=doppler[peak[0]]
    snapshot=(channels[:,lag:]*np.exp(-2j*np.pi*fd*t[lag:]))@np.conj(tx[:n-lag]) if lag<n else np.zeros(len(pos))
    azimuth=np.linspace(-85,85,171);elevation=np.linspace(-45,45,31) if np.ptp(pos[:,2])>0 else np.array([0.])
    azg,elg=np.meshgrid(azimuth,elevation);steerings=angle_steering(tuple(map(tuple,pos)),len(elevation))
    angular=abs(np.einsum("ij,j->i",np.conj(steerings),snapshot))**2;best=int(np.argmax(angular))
    est_az=float(azg.ravel()[best]) if len(pos)>1 else None;est_el=float(elg.ravel()[best]) if len(elevation)>1 else None
    pattern=abs(np.exp(-2j*np.pi*(directions(azimuth,np.full_like(azimuth,req.antenna.steer_el_deg))@pos.T))@w)*element_voltage(req.antenna,azimuth,req.antenna.steer_el_deg)
    sp=analyze(processed,pfs,SpectrumConfig(rbw_hz=pfs/1024,averaging=2,max_fft=2048));sf,sd=reduce_spectrum(sp.freq_hz,sp.psd_db,1000)
    edges=np.linspace(0,maxlag,min(480,maxlag)+1,dtype=int);reduced=np.maximum.reduceat(power,edges[:-1],axis=1);ranges=(range_axis[edges[:-1]]+range_axis[edges[1:]-1])/2
    ix=np.unique(np.linspace(0,n-1,min(1200,n),dtype=int));inst=np.r_[0,np.angle(tx[1:]*tx[:-1].conj())*fs/(2*np.pi)]
    measured_range=float(range_axis[peak[1]])
    ac=signal.fftconvolve(tx,tx[::-1].conj(),'full')[n-1:n-1+min(n,1600)];ac=20*np.log10(abs(ac)/max(float(abs(ac[0])),1e-20)+1e-12)
    if req.antenna.spacing_lambda>.5:warnings.append('Element spacing above half a wavelength can produce grating lobes.')
    if req.source=='program' and req.max_velocity_mps>lam/(4*req.program.period_s):warnings.append('Velocity search exceeds the nominal periodic-waveform unambiguous interval; repeated designs can have ambiguous peaks.')
    if req.processing.method=='fmcw':warnings.append('FMCW range includes Doppler/range coupling; configured chirp slope and period must match TX. Reset/down ramps need separate selection.')
    if req.processing.method!='correlation':warnings.append('Coherent processing assumes repeated pulses or linear chirps at the configured PRI. For arbitrary timing use full-waveform correlation.')
    if req.source=='legacy':warnings.append('Complete stored sequence repeats; PRI agility is retained within that sequence.')
    return {'capture_id':req.capture_id,'time_s':req.time_s,'sample_rate':fs,'processed_rate':pfs,'capture_duration_s':n/fs,
      'tx':preview(tx,fs),'rx':preview(beam,fs),'processed':probe,'stages':stages,
      'tx_frequency_hz':inst[ix].tolist(),'tx_time_s':t[ix].tolist(),'freq_hz':sf.tolist(),'psd_db':sd.tolist(),
      'range_m':ranges.tolist(),'range_db':(10*np.log10(np.maximum.reduceat(profile,edges[:-1])+1e-30)).tolist(),
      'threshold_db':(10*np.log10(np.maximum.reduceat(threshold,edges[:-1])+1e-30)).tolist(),
      'rd_db':(10*np.log10(reduced+1e-30)).ravel().tolist(),'rd_shape':list(reduced.shape),'velocity_mps':velocity.tolist(),
      'peak_range_m':measured_range,'peak_velocity_mps':float(velocity[peak[0]]),'azimuth_deg':est_az,'elevation_deg':est_el,
      'estimated_position':(directions(est_az,est_el)*measured_range).tolist() if est_az is not None and est_el is not None else None,
      'angle_az':azimuth.tolist(),'angle_el':elevation.tolist(),'angle_db':(10*np.log10(angular/max(float(angular.max()),1e-30)+1e-12)).tolist(),'angle_shape':[len(elevation),len(azimuth)],
      'pattern_db':(20*np.log10(pattern+1e-12)).tolist(),'steer_az_deg':steer_az,'array_positions':pos.tolist(),
      'autocorr_db':ac.tolist(),'autocorr_delay_s':(np.arange(len(ac))/fs).tolist(),
      'detections':[{'range_m':float(range_axis[i]),'power_db':float(10*np.log10(profile[i]+1e-30))} for i in detections],
      'truth':truth,'warnings':warnings,'rx_rms':float(np.sqrt(np.mean(abs(beam)**2))),
      'processing_method':req.processing.method,'processor_info':processor_info,
      'waveform_info':{'rms':float(np.sqrt(np.mean(abs(tx)**2))),'peak':float(np.max(abs(tx))),'duty_fraction':float(np.mean(abs(tx)>np.max(abs(tx))*.01)),'samples':n,'nyquist_hz':fs/2},
      'method':req.processing.method+'; full-waveform delay/Doppler correlation; narrowband far-field point targets. CA-CFAR over Doppler-maximized range profile has uncalibrated overall Pfa.'}
