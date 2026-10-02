"""Single-channel monostatic coherent-burst experiment using a designed pulse."""
from typing import Literal
from functools import lru_cache
import numpy as np
from scipy.signal import fftconvolve
from pydantic import BaseModel, Field
from .radar_studio import RadarStudioConfig, _pulse_samples

C=299792458.

class Target(BaseModel):
    name: str = Field(default='Target',max_length=48)
    range_m: float = Field(default=3500,ge=0,le=1e7)
    velocity_mps: float = Field(default=35,ge=-3000,le=3000)
    gain_db: float = Field(default=-6,ge=-80,le=0)

class SceneRequest(BaseModel):
    radar: RadarStudioConfig = Field(default_factory=RadarStudioConfig)
    pulse_index: int = Field(default=0,ge=0)
    carrier_hz: float = Field(default=3e9,ge=1e6,le=100e9)
    pulses: Literal[16,32,64] = 32
    noise_db: float = Field(default=-35,ge=-100,le=0)
    elapsed_s: float = Field(default=0,ge=0,le=86400)
    targets: list[Target] = Field(default_factory=lambda:[Target()],max_length=8)

@lru_cache(maxsize=8)
def _template(pulse_json,fs,seed):
    from .radar_studio import RadarPulseConfig
    return _pulse_samples(RadarPulseConfig.model_validate_json(pulse_json),fs,seed)


def scene_products(req: SceneRequest):
    cfg=req.radar
    if cfg.mode != 'sequence' or req.pulse_index >= len(cfg.sequence):
        return {'available':False,'reason':'Select a pulse in Pulse sequence to run the coherent return experiment.'}
    item=cfg.sequence[req.pulse_index]
    if item.kind != 'pulse' or item.pulse is None:
        return {'available':False,'reason':'Select a pulse, rather than off time, to inspect its returns.'}
    fs=cfg.sample_rate; pri=item.pri_s
    n=int(round(pri*fs))
    if n<16 or n>16384 or req.pulses*n>524288:
        return {'available':False,'reason':'For this bounded return experiment, choose 16–16384 samples per PRI and at most 524288 samples per burst.'}
    ref=_template(item.pulse.model_dump_json(),fs,cfg.seed).copy()*item.scale
    fast=np.arange(n)/fs
    ref=ref[:n]*np.exp(1j*(2*np.pi*item.frequency_offset_hz*fast[:len(ref[:n])]+np.deg2rad(item.phase_offset_deg)))
    # Complex baseband time and phase remain coherent across all pulses.
    time=fast[None,:]+np.arange(req.pulses)[:,None]*pri
    rx=np.zeros((req.pulses,n),complex)
    truth=[];warnings=[]
    wavelength=C/req.carrier_hz
    for target in req.targets:
        distance=target.range_m+target.velocity_mps*req.elapsed_s
        if distance<0:
            warnings.append(f'{target.name}: target has passed zero range.');continue
        delay=2*distance/C
        fd=-2*target.velocity_mps/wavelength
        if delay+len(ref)/fs>=pri:
            warnings.append(f'{target.name}: return extends beyond the PRI; it is clipped in this experiment.')
        if abs(fd)>1/(2*pri):warnings.append(f'{target.name}: velocity exceeds the unambiguous interval.')
        positions=(fast-delay)*fs
        echo=np.interp(positions,np.arange(len(ref)),ref.real,left=0,right=0)+1j*np.interp(positions,np.arange(len(ref)),ref.imag,left=0,right=0)
        rx+=10**(target.gain_db/20)*echo[None,:]*np.exp(2j*np.pi*fd*time)
        truth.append({'name':target.name,'range_m':distance,'velocity_mps':target.velocity_mps,'doppler_hz':fd})
    rng=np.random.default_rng(cfg.seed+int(req.elapsed_s*4))
    rx+=10**(req.noise_db/20)*(rng.normal(size=rx.shape)+1j*rng.normal(size=rx.shape))/np.sqrt(2)
    matched=fftconvolve(rx,np.conj(ref[::-1])[None,:],mode='full',axes=1)[:,len(ref)-1:len(ref)-1+n]
    matched/=max(float(np.vdot(ref,ref).real),1e-15)
    rd=np.fft.fftshift(np.fft.fft(matched*np.hanning(req.pulses)[:,None],axis=0),axes=0)/max(np.sum(np.hanning(req.pulses)),1)
    power=np.abs(rd)**2
    edges=np.linspace(0,n,min(n,640)+1,dtype=int)
    reduced=np.maximum.reduceat(power,edges[:-1],axis=1)
    db=10*np.log10(reduced+1e-12)
    ranges=(edges[:-1]+np.diff(edges)/2)*C/(2*fs)
    velocity=-np.fft.fftshift(np.fft.fftfreq(req.pulses,pri))*wavelength/2
    peak=np.unravel_index(np.argmax(power),power.shape)
    stride=max(1,n//1200)
    return {'available':True,'time_s':req.elapsed_s,'truth':truth,'warnings':warnings,
        'range_m':ranges.tolist(),'velocity_mps':velocity.tolist(),
        'range_db':(10*np.log10(np.maximum.reduceat(np.mean(np.abs(matched)**2,axis=0),edges[:-1])+1e-12)).tolist(),
        'rd_db':db.ravel().tolist(),'rd_shape':list(db.shape),
        'receive_envelope':np.abs(rx[0,::stride]).tolist(),'receive_time_s':fast[::stride].tolist(),
        'peak_range_m':float(peak[1]*C/(2*fs)),'peak_velocity_mps':float(velocity[peak[0]]),
        'unambiguous_range_m':C*pri/2,'unambiguous_velocity_mps':wavelength/(4*pri),
        'range_sample_m':C/(2*fs),'velocity_bin_mps':wavelength/(2*req.pulses*pri),
        'model':'Selected pulse repeated at fixed PRI; complex echo gain, AWGN, constant radial velocity. Single channel; angles are not observable.'}
