"""Ordered user processing blocks with inspectable intermediate products."""
from fractions import Fraction
from typing import Literal
import numpy as np
from scipy import signal
from pydantic import BaseModel,Field,ConfigDict
from .expressions import evaluate
from .programs import extensions
from .models import MAX_CAPTURE_SAMPLES
from .dsp import display_indices

class DSPBlock(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    name:str=Field(default='Block',max_length=64)
    kind:Literal['gain','mixer','fir','lowpass','resample','dc','agc','expression','costas','gardner','mueller-muller','cma','extension']='gain'
    enabled:bool=True
    gain_db:float=Field(default=0,ge=-120,le=80)
    frequency_hz:float=0
    taps_i:list[float]=Field(default_factory=lambda:[1],min_length=1,max_length=2048)
    taps_q:list[float]=Field(default_factory=list,max_length=2048)
    cutoff_hz:float=Field(default=100000,gt=0)
    taps:int=Field(default=63,ge=3,le=2047)
    up:int=Field(default=1,ge=1,le=32)
    down:int=Field(default=2,ge=1,le=128)
    expression:str=Field(default='x',max_length=8000)
    symbol_rate:float=Field(default=50000,gt=0)
    loop_gain:float=Field(default=.05,gt=0,le=1)
    beta:float=Field(default=.0005,gt=0,le=.2)
    order:Literal[2,4]=4
    extension:str=Field(default='',max_length=80)
    parameters:dict[str,float]=Field(default_factory=dict,max_length=32)

class Pipeline(BaseModel):
    blocks:list[DSPBlock]=Field(default_factory=list,max_length=24)
    reset_epoch:int=Field(default=0,ge=0)
    probe:int=Field(default=-1,ge=-1,le=23)

def preview(x,fs,maximum=640):
    ix=display_indices(len(x),maximum)
    return {'i':x.real[ix].tolist(),'q':x.imag[ix].tolist(),'time_s':np.round(ix/fs,9).tolist()}

def process_chain(samples,fs,cfg:Pipeline,start_s=0,frame=0,memory=None,start_sample=None):
    """Run the enabled blocks in order.

    ``start_sample`` (absolute index of the first input sample) gives the mixer
    an exact capture-start phase; otherwise ``start_s`` is used.
    """
    from .sync import gardner,mueller_muller,cma_equalizer
    from .dsp import phasor
    memory={} if memory is None else memory
    t0=Fraction(int(start_sample))/Fraction(float(fs)) if start_sample is not None else Fraction(float(start_s))
    start_s=float(t0)
    x=np.asarray(samples,complex);stages=[];probe=preview(x,fs)
    for index,b in enumerate(cfg.blocks):
        if not b.enabled:continue
        saved=memory.setdefault(str(index),{})
        t=start_s+np.arange(len(x))/fs
        if b.kind=='gain':x=x*10**(b.gain_db/20)
        elif b.kind=='mixer':x=x*phasor(-b.frequency_hz,fs,len(x),start_time=t0)
        elif b.kind=='dc':x=x-np.mean(x)
        elif b.kind=='agc':x=x/max(float(np.sqrt(np.mean(abs(x)**2))),1e-12)
        elif b.kind=='fir':
            if b.taps_q and len(b.taps_q)!=len(b.taps_i):raise ValueError('FIR I/Q tap arrays must have equal length')
            h=np.asarray(b.taps_i)+1j*(np.asarray(b.taps_q) if b.taps_q else 0)
            x,saved['zi']=signal.lfilter(h,[1],x,zi=saved.get('zi',np.zeros(len(h)-1,complex)))
        elif b.kind=='lowpass':
            if b.cutoff_hz>=fs/2:raise ValueError('Filter cutoff must be below Nyquist at this stage')
            h=signal.firwin(b.taps,b.cutoff_hz,fs=fs)
            x,saved['zi']=signal.lfilter(h,[1],x,zi=saved.get('zi',np.zeros(len(h)-1,complex)))
        elif b.kind=='resample':
            if len(x)*b.up/b.down>MAX_CAPTURE_SAMPLES:raise ValueError(f'Resampler output exceeds {MAX_CAPTURE_SAMPLES:,} samples')
            x=signal.resample_poly(x,b.up,b.down);fs=fs*b.up/b.down
        elif b.kind=='expression':x=evaluate(b.expression,{'x':x,'t':t,'time':t,'n':np.arange(len(x)),'fs':fs,'frame':frame},len(x))
        elif b.kind=='costas':
            phase=saved.get('phase',0.);freq=saved.get('frequency',0.);out=np.empty_like(x)
            for k,value in enumerate(x):
                z=value*np.exp(-1j*phase);out[k]=z
                error=z.real*z.imag if b.order==2 else np.sign(z.real)*z.imag-np.sign(z.imag)*z.real
                freq+=b.beta*error;phase=(phase+freq+b.loop_gain*error+np.pi)%(2*np.pi)-np.pi
            x=out;saved.update(phase=phase,frequency=freq)
        elif b.kind in ('gardner','mueller-muller'):
            sps=fs/b.symbol_rate
            if not 2<=sps<=2048:raise ValueError('Timing stage requires 2–2048 samples per symbol')
            x=(gardner(x,sps,b.loop_gain) if b.kind=='gardner' else mueller_muller(x,round(sps),b.loop_gain)).output;fs=b.symbol_rate
        elif b.kind=='cma':x,_,_=cma_equalizer(x,min(21,b.taps),min(.03,b.loop_gain))
        elif b.kind=='extension':
            module=extensions().get(b.extension)
            if module is None or not callable(getattr(module,'process',None)):raise ValueError('Choose an installed processing extension')
            result=module.process(x.copy(),fs,{'time_s':start_s,'frame':frame,'parameters':b.parameters,'state':saved})
            x,fs=result if isinstance(result,tuple) else (result,fs);x=np.asarray(x,complex)
        if x.ndim!=1 or not 2<=len(x)<=MAX_CAPTURE_SAMPLES or not np.isfinite(x).all() or not np.isfinite(fs) or fs<=0:raise ValueError(f'{b.name}: block must produce a finite 1-D array and positive sample rate')
        stages.append({'index':index,'name':b.name,'kind':b.kind,'state':('carried' if b.kind in {'fir','lowpass','costas','extension'} else 'capture-local'),'samples':len(x),'sample_rate':float(fs),'rms':float(np.sqrt(np.mean(abs(x)**2))),'peak':float(np.max(abs(x)))})
        if cfg.probe in (-1,index):probe=preview(x,fs)
    return x,fs,stages,probe


def channel_samples(g,channel,frame):
    """Apply the receiver channel experiment to one capture (``None`` bypasses it)."""
    from .sync import apply_channel
    if channel is None or not getattr(channel,'enabled',True):
        return np.asarray(g.samples,dtype=np.complex64)
    source=g.samples
    if channel.echo_gain:
        delay=max(1,int(g.sample_rate/(g.metadata.get('tx_symbol_rate') or 50000)))*channel.echo_delay_symbols
        source=source+channel.echo_gain*np.pad(source,(delay,0))[:len(source)]
    start=g.metadata.get('start_sample')
    return apply_channel(source,g.sample_rate,channel,seed=(g.metadata.get('source_seed',7)+frame*104729+1)%(2**32),
                         start_s=float(g.metadata.get('simulation_time_s',0)),start_sample=None if start is None else int(start))

def dehop(x,schedules,fs,start_sample=0):
    """Remove known hop/sweep schedules (synchronized frequency-hopping receiver)."""
    from .agility import oscillator
    for agility in schedules or ():
        if agility.enabled:
            lo,_=oscillator(agility,fs,len(x),int(start_sample or 0))
            x=np.asarray(x)*np.conj(lo)
    return x

def pipeline_product(g,channel,cfg,frame,memory=None):
    from .models import SpectrumConfig
    from .spectrum import analyze
    start=float(g.metadata.get('simulation_time_s',0))
    y=channel_samples(g,channel,frame)
    z,fs,stages,probe=process_chain(y,g.sample_rate,cfg,start,frame,memory,start_sample=g.metadata.get('start_sample'))
    spec=analyze(z,fs,SpectrumConfig(rbw_hz=fs/1024,averaging=2,max_fft=2048))
    from .engine import reduce_spectrum
    freq,db=reduce_spectrum(spec.freq_hz,spec.psd_db,640)
    return {'capture_id':frame,'input':preview(y,g.sample_rate),'output':probe,'stages':stages,
            'output_rate':fs,'freq_hz':np.round(freq,3).tolist(),'psd_db':np.round(db,4).tolist(),
            'rms':float(np.sqrt(np.mean(abs(z)**2))),'papr_db':float(10*np.log10((np.max(abs(z)**2)+1e-20)/(np.mean(abs(z)**2)+1e-20)))}
