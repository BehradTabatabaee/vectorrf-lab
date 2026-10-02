"""Explicit pulse-Doppler and FMCW receive processing at configured timing."""
from typing import Literal
import numpy as np
from scipy import signal,fft
from pydantic import BaseModel,Field,ConfigDict
C=299792458.
class RadarProcessing(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    method:Literal['correlation','pulse-doppler','fmcw']='correlation'
    pri_s:float=Field(default=.0002,gt=1e-7,le=1)
    pulse_width_s:float=Field(default=.00008,gt=1e-8,le=1)
    chirp_slope_hz_s:float=Field(default=2.5e9)
    clutter:Literal['off','mean','two-pulse','three-pulse']='off'
    range_window:Literal['boxcar','hann','hamming','blackman']='boxcar'
    doppler_window:Literal['boxcar','hann','hamming','blackman']='hann'
    min_range_m:float=Field(default=0,ge=0)
    detector:Literal['ca-cfar','manual']='ca-cfar'
    threshold_db:float=Field(default=-40,ge=-300,le=100)

def cancel_clutter(cube,mode):
    if mode=='mean':return cube-cube.mean(axis=0,keepdims=True)
    if mode in ('two-pulse','three-pulse'):return np.diff(cube,n=1 if mode=='two-pulse' else 2,axis=0)
    return cube

def range_doppler(processed,reference,fs,time_s,carrier_hz,settings,max_range,max_velocity,bins):
    p=settings;length=int(round(p.pri_s*fs));offset=(-int(round(time_s*fs)))%max(1,length)
    count=(len(processed)-offset)//max(1,length)
    if length<4 or count<4:raise ValueError('This radar processor needs at least four complete PRI/chirp intervals per capture; increase samples or shorten PRI')
    cube=processed[offset:offset+count*length].reshape(count,length)
    refs=reference[offset:offset+count*length].reshape(count,length)
    if p.method=='fmcw':
        if abs(p.chirp_slope_hz_s)<1:raise ValueError('FMCW needs the signed chirp slope in Hz/s')
        cube=refs*np.conj(cube)
        cube=cancel_clutter(cube,p.clutter)
        spectrum=fft.fft(cube*signal.get_window(p.range_window,length),axis=1)
        freqs=fft.fftfreq(length,1/fs);ranges=C*freqs/(2*p.chirp_slope_hz_s)
        keep=np.flatnonzero((ranges>=p.min_range_m)&(ranges<=max_range));keep=keep[np.argsort(ranges[keep])]
        cube=spectrum[:,keep]/length;ranges=ranges[keep];sign=1
    else:
        width=int(round(p.pulse_width_s*fs))
        if not 1<=width<length:raise ValueError('Pulse width must be shorter than PRI and at least one sample')
        # Same transmitted waveform supplies a distinct matched reference per pulse.
        taps=refs[:,:width]*signal.get_window(p.range_window,width)
        size=fft.next_fast_len(length+width-1)
        compressed=fft.ifft(fft.fft(cube,size,axis=1)*fft.fft(taps[:,::-1].conj(),size,axis=1),axis=1)
        cube=compressed[:,width-1:width-1+length]/np.maximum(np.sum(abs(taps)**2,axis=1,keepdims=True),1e-20)
        cube=cancel_clutter(cube,p.clutter)
        ranges=np.arange(length)*C/(2*fs);keep=(ranges>=p.min_range_m)&(ranges<=max_range);ranges=ranges[keep];cube=cube[:,keep];sign=-1
    if not len(ranges):raise ValueError('Range gate contains no processor bins')
    slow=len(cube);window=signal.get_window(p.doppler_window,slow)
    count_fft=max(slow,bins);rd=fft.fftshift(fft.fft(cube*window[:,None],count_fft,axis=0),axes=0)/max(float(window.sum()),1e-20)
    velocity=sign*fft.fftshift(fft.fftfreq(count_fft,length/fs))*C/carrier_hz/2
    order=np.argsort(velocity);rd=rd[order];velocity=velocity[order]
    valid=np.flatnonzero(abs(velocity)<=max_velocity)
    if not len(valid):raise ValueError('Velocity gate contains no processor bins')
    valid=valid[np.unique(np.linspace(0,len(valid)-1,min(bins,len(valid)),dtype=int))]
    return rd[valid],ranges,velocity[valid],{'complete_intervals':count,'used_intervals':slow,'actual_pri_s':length/fs,'coherent_time_s':slow*length/fs,'velocity_resolution_mps':C/carrier_hz/(2*slow*length/fs),'range_bin_m':float(abs(ranges[1]-ranges[0])) if len(ranges)>1 else None}
