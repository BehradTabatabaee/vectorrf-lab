"""User-designed complex waveform functions and local Python extension points."""
from typing import Literal
from pathlib import Path
import importlib.util
import numpy as np
from pydantic import BaseModel,Field,ConfigDict,model_validator
from .expressions import evaluate

class Knot(BaseModel):
    time_us:float=Field(default=0,ge=0,le=1e6)
    frequency_hz:float=0
    amplitude:float=Field(default=1,ge=0,le=10)
    phase_deg:float=0

class WaveProgram(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    mode:Literal['expression','points','iq','extension']='expression'
    period_s:float=Field(default=.0002,gt=0,le=1)
    expression:str=Field(default='where(t < 0.00008, exp(j*2*pi*(-100000*t+1250000000*t*t)), 0)',max_length=8000)
    points:list[Knot]=Field(default_factory=lambda:[Knot(),Knot(time_us=400,frequency_hz=100000),Knot(time_us=401,amplitude=0),Knot(time_us=1000,amplitude=0)],min_length=2,max_length=512)
    samples_i:list[float]=Field(default_factory=lambda:[1,0,-1,0],min_length=1,max_length=16384)
    samples_q:list[float]=Field(default_factory=lambda:[0,1,0,-1],min_length=1,max_length=16384)
    table_rate:float=Field(default=100000,gt=0,le=1e8)
    table_interp:Literal['linear','hold']='linear'
    extension:str=Field(default='',max_length=80)
    parameters:dict[str,float]=Field(default_factory=dict,max_length=32)
    @model_validator(mode='after')
    def check(self):
        if self.mode=='points' and any(b.time_us<=a.time_us for a,b in zip(self.points,self.points[1:])):raise ValueError('Control point times must increase')
        if len(self.samples_i)!=len(self.samples_q):raise ValueError('I and Q tables must have the same length')
        return self

EXTENSIONS=Path(__file__).resolve().parents[2]/'extensions'
_registry=None

def extensions():
    global _registry
    if _registry is None:
        _registry={}
        for p in sorted(EXTENSIONS.glob('*.py')):
            if p.name.startswith('_'):continue
            spec=importlib.util.spec_from_file_location('vectorrf_user_'+p.stem,p)
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            _registry[p.stem]=m
    return _registry

def extension_catalog():
    return [{'name':k,'waveform':callable(getattr(v,'waveform',None)),'process':callable(getattr(v,'process',None))} for k,v in extensions().items()]

def program_samples(cfg:WaveProgram,fs,n,start_s=0,frame=0,seed=7,start_sample=None):
    # From an absolute sample index, time is exact for every sample of the
    # stream (start_s + k/fs rounds differently in different captures).
    time=(start_sample+np.arange(n))/fs if start_sample is not None else start_s+np.arange(n)/fs;t=time%cfg.period_s
    if cfg.mode=='expression':
        y=evaluate(cfg.expression,{'t':t,'time':time,'n':np.arange(n),'fs':fs,'T':cfg.period_s,'frame':frame,**cfg.parameters},n)
    elif cfg.mode=='iq':
        a=np.asarray(cfg.samples_i)+1j*np.asarray(cfg.samples_q);pos=(time*cfg.table_rate)%len(a)
        if cfg.table_interp=='hold':y=a[np.floor(pos+1e-9).astype(int)%len(a)]
        else:y=np.interp(pos,np.arange(len(a)+1),np.r_[a.real,a.real[0]])+1j*np.interp(pos,np.arange(len(a)+1),np.r_[a.imag,a.imag[0]])
    elif cfg.mode=='extension':
        m=extensions().get(cfg.extension)
        if m is None or not callable(getattr(m,'waveform',None)):raise ValueError('Choose an installed waveform extension')
        y=np.asarray(m.waveform({'t':t,'time':time,'fs':fs,'frame':frame,'parameters':cfg.parameters,'rng':np.random.default_rng(seed)}),complex)
    else:
        times=np.array([k.time_us*1e-6 for k in cfg.points]);f=np.array([k.frequency_hz for k in cfg.points]);a=np.array([k.amplitude for k in cfg.points]);ph=np.array([k.phase_deg for k in cfg.points])
        # Exact integral of a piecewise-linear frequency law; independent of capture start.
        ix=np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2)
        slope=np.diff(f)/np.diff(times);integral=np.r_[0,np.cumsum((f[1:]+f[:-1])*.5*np.diff(times))]
        u=np.clip(t-times[ix],0,np.diff(times)[ix]);phase=integral[ix]+f[ix]*u+.5*slope[ix]*u*u
        y=np.interp(t,times,a,left=0,right=0)*np.exp(1j*(2*np.pi*phase+np.deg2rad(np.interp(t,times,ph))))
    if y.shape!=(n,) or not np.isfinite(y).all():raise ValueError('Waveform must return one finite complex sample per requested time')
    return y.astype(np.complex64)
