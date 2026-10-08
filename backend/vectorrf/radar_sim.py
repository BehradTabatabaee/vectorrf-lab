"""Waveform-level point-target radar with physical array channels and arbitrary TX."""
import math
from typing import Literal
from functools import lru_cache
import numpy as np
from scipy import signal,fft
from pydantic import BaseModel,Field,ConfigDict,model_validator
from .programs import WaveProgram,program_samples
from .pipeline import Pipeline,process_chain,preview
from .radar_studio import RadarStudioConfig,design_summary,generate_radar
from .models import TxConfig,SpectrumConfig,MAX_SAMPLE_RATE
from .recordings import FileSource
from .signal import generate
from .dsp import fractional_delay
from .spectrum import analyze
from .engine import reduce_spectrum
from .radar_processing import RadarProcessing,cfar_threshold,detect,range_doppler
C=299792458.
# Sample-count bounds (rate-independent).  Channels × samples bounds the
# complex128 array cube; range–Doppler cells bound the correlator surface.
RADAR_MAX_SAMPLES=262_144
RADAR_MAX_CHANNEL_SAMPLES=1<<22
RADAR_MAX_RD_CELLS=1<<23
RD_DISPLAY_ROWS=161

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
    source:Literal['legacy','program','ofdm','file']='program'
    legacy:RadarStudioConfig=Field(default_factory=RadarStudioConfig)
    program:WaveProgram=Field(default_factory=WaveProgram)
    ofdm:TxConfig=Field(default_factory=lambda:TxConfig(waveform='ofdm',snr_db=None))
    file:FileSource|None=None
    sample_rate:float=Field(default=1_000_000,gt=1000,le=MAX_SAMPLE_RATE)
    samples:int=Field(default=4096,ge=512,le=RADAR_MAX_SAMPLES)
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
    # Impairments relative to the transmitted signal (dBc: unit amplitude in the
    # relative model, the transmit power in the radar equation); None = off.
    clutter_db:float|None=Field(default=None,ge=-300,le=0)
    leakage_db:float|None=Field(default=None,ge=-300,le=0)
    max_range_m:float=Field(default=15000,gt=0,le=1e7)
    max_velocity_mps:float=Field(default=100,gt=0,le=3000)
    doppler_bins:int=Field(default=41,ge=9,le=81)
    cfar_guard:int=Field(default=3,ge=1,le=20)
    cfar_train:int=Field(default=12,ge=2,le=64)
    cfar_pfa:float=Field(default=.001,gt=0,lt=.5)
    seed:int=7
    @model_validator(mode='after')
    def validate_experiment(self):
        if self.source=='file' and self.file is None:raise ValueError('Choose a recording for the file-based radar waveform')
        if self.source=='legacy' and self.legacy.sample_rate!=self.sample_rate:
            # One sample clock: the designed pulse/CW record is generated at the
            # simulation rate (and revalidated there), never interpolated from
            # a second, silently different rate.
            self.legacy=RadarStudioConfig.model_validate({**self.legacy.model_dump(),'sample_rate':self.sample_rate})
        channels=len(positions(self.antenna))
        if channels*self.samples>RADAR_MAX_CHANNEL_SAMPLES:
            raise ValueError(f'{channels} channels × {self.samples:,} samples exceeds {RADAR_MAX_CHANNEL_SAMPLES:,} channel-samples; reduce samples or array size')
        return self

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

@lru_cache(maxsize=4)
def file_record(spec_json):
    """One window of a recording, repeated as the radar's periodic transmit waveform."""
    from .recordings import file_capture
    spec=FileSource.model_validate_json(spec_json)
    g=file_capture(spec.model_copy(update={'loop':True}),0)
    return np.asarray(g.samples),g.sample_rate

def transmitter(req,start,n):
    if req.source=='program':return program_samples(req.program,req.sample_rate,n,start,req.capture_id,req.seed+req.capture_id)
    if req.source=='legacy':base,rate=legacy_record(req.legacy.model_dump_json())
    elif req.source=='file':base,rate=file_record(req.file.model_dump_json())
    else:
        cpv=req.ofdm.ofdm_design.cp_lengths or [req.ofdm.ofdm_cp]
        frame_samples=sum(req.ofdm.ofdm_fft+cpv[i%len(cpv)] for i in range(req.ofdm.ofdm_design.symbols))
        cfg=req.ofdm.model_copy(update={'sample_rate':req.sample_rate,'duration':frame_samples/req.sample_rate,'snr_db':None})
        base=ofdm_record(cfg.model_dump_json());rate=req.sample_rate
    return periodic_slice(base,rate,req.sample_rate,start,n)


def periodic_slice(base,rate,fs,start,n,margin=40):
    """``n`` samples at ``fs`` of the periodic record ``base`` (rate ``rate``) from time ``start``.

    At the record's own rate a delayed echo starts between samples: the slice
    is read on the record's grid and moved by the remaining fraction of a
    sample with a windowed sinc (band-limited, about -65 dB error up to 0.8
    Nyquist), as an ideal delay of the sampled waveform would; linear
    interpolation would attenuate and distort content near Nyquist.  A
    recording at another rate is resampled by linear interpolation.
    """
    base=np.asarray(base);m=len(base)
    if abs(rate-fs)<=1e-9*fs:
        position=start*rate;first=math.floor(position);fraction=position-first
        segment=base[(first-margin+np.arange(n+2*margin))%m]
        if fraction>1e-12:segment=fractional_delay(segment,-fraction,ntaps=2*(margin//2)+1)
        return np.asarray(segment[margin:margin+n],complex)
    p=((start+np.arange(n)/fs)*rate)%m
    return np.interp(p,np.arange(m+1),np.r_[base.real,base.real[0]])+1j*np.interp(p,np.arange(m+1),np.r_[base.imag,base.imag[0]])

def mainlobe_bins(ref,level=.1):
    """Half-width in samples of the reference's autocorrelation mainlobe, down to ``level`` of its peak power (−10 dB)."""
    n=len(ref)
    ac=abs(signal.fftconvolve(ref,np.conj(ref[::-1]),'full')[n-1:n-1+min(n,4096)])**2
    if ac[0]<=0:return 1
    below=np.flatnonzero(ac<level*ac[0])
    return int(below[0]) if len(below) else min(n,4096)

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
    # Leakage and clutter are set relative to the transmitted signal: unit
    # amplitude in the relative model, sqrt(transmit power) in the radar equation.
    tx_scale=np.sqrt(req.power_w) if req.power_model=='radar-equation' else 1.
    if req.leakage_db is not None:
        channels+=tx_scale*10**(req.leakage_db/20)*tx[None,:]
        strongest=max((t['echo_power_db'] for t in truth),default=None)
        if strongest is not None and 20*np.log10(tx_scale)+req.leakage_db>strongest:
            warnings.append(f'TX leakage is {20*np.log10(tx_scale)+req.leakage_db-strongest:.0f} dB above the strongest echo and dominates the zero-range cells; reduce it or set a minimum range.')
    if req.clutter_db is not None:
        clutter=(rng.normal(size=n)+1j*rng.normal(size=n))*tx_scale*10**(req.clutter_db/20)/np.sqrt(2)
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
    if len(processed)>2*RADAR_MAX_SAMPLES:raise ValueError(f'Radar processing output is limited to {2*RADAR_MAX_SAMPLES:,} samples per capture')
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
        if len(velocity)*maxlag>RADAR_MAX_RD_CELLS:raise ValueError(f'Delay/Doppler search of {len(velocity)}×{maxlag:,} cells exceeds {RADAR_MAX_RD_CELLS:,}; reduce maximum range or Doppler bins')
        rd=np.empty((len(velocity),maxlag),complex)
        # The Doppler window tapers the capture (normalized to unit coherent
        # gain, so a target's peak keeps its level): Hann sidelobes are -31 dB
        # instead of a rectangular window's -13 dB.
        taper=signal.get_window(req.processing.doppler_window,nout)
        tapered=processed*(taper/max(float(np.mean(taper)),1e-12))
        for begin in range(0,len(velocity),8):
            batch=tapered[None,:]*mixers[begin:begin+8]
            values=fft.ifft(fft.fft(batch,size,axis=1)*reference_fft,axis=1)
            rd[begin:begin+8]=values[:,len(ref)-1:len(ref)-1+maxlag]/ref_energy
        range_axis=np.arange(maxlag)*C/(2*pfs)
        keep=range_axis>=req.processing.min_range_m
        if not np.any(keep):raise ValueError('Range gate contains no correlation bins')
        rd=rd[:,keep];range_axis=range_axis[keep];maxlag=len(range_axis)
    power=abs(rd)**2;peak=np.unravel_index(np.argmax(power),power.shape)
    profile=np.max(power,axis=0);profile_rows=np.argmax(power,axis=0)
    # CFAR guard cells must cover the compressed mainlobe, or a target raises its
    # own threshold: the mainlobe spans more range bins at higher sample rates.
    guard,train=req.cfar_guard,req.cfar_train
    if req.processing.method!='fmcw':
        lobe=mainlobe_bins(ref)
        guard=max(guard,lobe);train=max(train,lobe)
    detector=req.processing.detector
    if detector=='manual':
        threshold=np.full(power.shape,10**(req.processing.threshold_db/10))
    elif maxlag>train+guard:
        # Along range in every Doppler row: each cell is tested against its own row.
        threshold=cfar_threshold(power,detector,guard,train,req.cfar_pfa,req.processing.os_rank)
    else:
        threshold=np.full(power.shape,np.inf)
        warnings.append(f'The CFAR window (guard {guard} + {train} training bins on a side) does not fit the {maxlag}-bin range profile: no CFAR detections. Increase the maximum range or use the manual detector.')
    cells=detect(power,threshold)
    # The range profile is the Doppler maximum; its threshold is the one of the
    # row holding that maximum, so a profile crossing is a detected cell.
    threshold_profile=threshold[profile_rows,np.arange(maxlag)]
    # Direction finding is derived from array-channel samples at the measured range/Doppler peak.
    delay=2*range_axis[peak[1]]/C;lag=int(round(delay*fs));fd=doppler[peak[0]]
    snapshot=(channels[:,lag:]*np.exp(-2j*np.pi*fd*t[lag:]))@np.conj(tx[:n-lag]) if lag<n else np.zeros(len(pos))
    azimuth=np.linspace(-85,85,171);elevation=np.linspace(-45,45,31) if np.ptp(pos[:,2])>0 else np.array([0.])
    azg,elg=np.meshgrid(azimuth,elevation);steerings=angle_steering(tuple(map(tuple,pos)),len(elevation))
    angular=abs(np.einsum("ij,j->i",np.conj(steerings),snapshot))**2;best=int(np.argmax(angular))
    est_az=float(azg.ravel()[best]) if len(pos)>1 else None;est_el=float(elg.ravel()[best]) if len(elevation)>1 else None
    pattern=abs(np.exp(-2j*np.pi*(directions(azimuth,np.full_like(azimuth,req.antenna.steer_el_deg))@pos.T))@w)*element_voltage(req.antenna,azimuth,req.antenna.steer_el_deg)
    sp=analyze(processed,pfs,SpectrumConfig(rbw_hz=pfs/1024,averaging=2,max_fft=2048));sf,sd=reduce_spectrum(sp.freq_hz,sp.psd_db,1000)
    # Display surface: at most 480 range columns and RD_DISPLAY_ROWS Doppler rows, max-pooled
    # so no peak disappears (measurements above use the full surface).
    edges=np.linspace(0,maxlag,min(480,maxlag)+1,dtype=int);reduced=np.maximum.reduceat(power,edges[:-1],axis=1);ranges=(range_axis[edges[:-1]]+range_axis[edges[1:]-1])/2
    rows=len(velocity);row_edges=np.linspace(0,rows,min(RD_DISPLAY_ROWS,rows)+1,dtype=int)
    reduced=np.maximum.reduceat(reduced,row_edges[:-1],axis=0);shown_velocity=(velocity[row_edges[:-1]]+velocity[row_edges[1:]-1])/2
    ix=np.unique(np.linspace(0,n-1,min(1200,n),dtype=int));inst=np.r_[0,np.angle(tx[1:]*tx[:-1].conj())*fs/(2*np.pi)]
    measured_range=float(range_axis[peak[1]])
    ac=signal.fftconvolve(tx,tx[::-1].conj(),'full')[n-1:n-1+min(n,1600)];ac=20*np.log10(abs(ac)/max(float(abs(ac[0])),1e-20)+1e-12)
    if req.antenna.spacing_lambda>.5:warnings.append('Element spacing above half a wavelength can produce grating lobes.')
    if req.source=='program' and req.max_velocity_mps>lam/(4*req.program.period_s):warnings.append('Velocity search exceeds the nominal periodic-waveform unambiguous interval; repeated designs can have ambiguous peaks.')
    if req.processing.method=='fmcw':warnings.append('FMCW range includes Doppler/range coupling; configured chirp slope and period must match TX. Reset/down ramps need separate selection.')
    if req.processing.method!='correlation':warnings.append('Coherent processing assumes repeated pulses or linear chirps at the configured PRI. For arbitrary timing use full-waveform correlation.')
    if req.source=='legacy':warnings.append('Complete stored sequence repeats; PRI agility is retained within that sequence.')
    if req.source=='file':
        _,file_rate=file_record(req.file.model_dump_json())
        warnings.append('Recorded window repeats as the transmit waveform; the radar applies delay and Doppler to it.')
        if abs(file_rate-fs)>1e-6*fs:warnings.append(f'Recording rate {file_rate:,.0f} S/s differs from the simulation rate; samples are linearly interpolated, which attenuates content near Nyquist.')
    return {'capture_id':req.capture_id,'time_s':req.time_s,'sample_rate':fs,'processed_rate':pfs,'capture_duration_s':n/fs,
      'tx':preview(tx,fs),'rx':preview(beam,fs),'processed':probe,'stages':stages,
      'tx_frequency_hz':inst[ix].tolist(),'tx_time_s':t[ix].tolist(),'freq_hz':sf.tolist(),'psd_db':sd.tolist(),
      'range_m':ranges.tolist(),'range_db':(10*np.log10(np.maximum.reduceat(profile,edges[:-1])+1e-30)).tolist(),
      'threshold_db':[float(v) if np.isfinite(v) else None for v in 10*np.log10(np.minimum.reduceat(np.where(np.isfinite(threshold_profile),threshold_profile,np.inf),edges[:-1])+1e-30)],
      'rd_db':(10*np.log10(reduced+1e-30)).ravel().tolist(),'rd_shape':list(reduced.shape),'velocity_mps':shown_velocity.tolist(),
      'peak_range_m':measured_range,'peak_velocity_mps':float(velocity[peak[0]]),'azimuth_deg':est_az,'elevation_deg':est_el,
      'estimated_position':(directions(est_az,est_el)*measured_range).tolist() if est_az is not None and est_el is not None else None,
      'angle_az':azimuth.tolist(),'angle_el':elevation.tolist(),'angle_db':(10*np.log10(angular/max(float(angular.max()),1e-30)+1e-12)).tolist(),'angle_shape':[len(elevation),len(azimuth)],
      'pattern_db':(20*np.log10(pattern+1e-12)).tolist(),'steer_az_deg':steer_az,'array_positions':pos.tolist(),
      'autocorr_db':ac.tolist(),'autocorr_delay_s':(np.arange(len(ac))/fs).tolist(),
      'detections':[{'range_m':float(range_axis[c]),'velocity_mps':float(velocity[r]),'power_db':float(10*np.log10(power[r,c]+1e-30)),
                     'margin_db':float(10*np.log10(power[r,c]/threshold[r,c])) if np.isfinite(threshold[r,c]) and threshold[r,c]>0 else None} for r,c in cells],
      'detector':detector,
      'truth':truth,'warnings':warnings,'rx_rms':float(np.sqrt(np.mean(abs(beam)**2))),
      'processing_method':req.processing.method,'processor_info':processor_info,
      'design':design_summary(req.legacy,req.carrier_hz) if req.source=='legacy' else None,
      'waveform_info':{'rms':float(np.sqrt(np.mean(abs(tx)**2))),'peak':float(np.max(abs(tx))),'duty_fraction':float(np.mean(abs(tx)>np.max(abs(tx))*.01)),'samples':n,'nyquist_hz':fs/2},
      'method':req.processing.method+'; narrowband far-field point targets. '+('Fixed threshold.' if detector=='manual' else
               f'{detector.upper()} along range in every Doppler row, Pfa {req.cfar_pfa:g} per cell for noise-like cells; detections are local maxima of the range–Doppler map.')}
