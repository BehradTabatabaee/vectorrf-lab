"""Explicit pulse-Doppler and FMCW receive processing at configured timing."""
from functools import lru_cache
import math
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
    # Constant-false-alarm-rate detectors (see ``cfar_threshold``) or a fixed threshold.
    detector:Literal['ca-cfar','go-cfar','so-cfar','os-cfar','manual']='ca-cfar'
    # Ordered-statistic CFAR: the training cell of this rank (fraction of the cells) sets the noise level.
    os_rank:float=Field(default=.75,gt=0,le=1)
    threshold_db:float=Field(default=-40,ge=-300,le=100)

# ----------------------------------------------------------------- CFAR --
#
# Every detector compares a cell's power with a multiple of a noise estimate
# taken from ``train`` cells on each side of it along range, outside ``guard``
# cells.  For square-law detected complex Gaussian noise the cells are
# exponential, and each multiplier below is solved exactly for the requested
# per-cell false-alarm probability (Gandhi & Kassam 1988; Rohling 1983):
#
# * CA (cell averaging), mean of the N cells:   Pfa = (1 + α/N)^-N
# * SO (smallest of the two sides' means, n cells each):
#       Pfa_SO(α) = 2 Σ_{k=0}^{n-1} C(n-1+k, k) (2 + α/n)^-(n+k)
# * GO (greatest of the two sides' means):  Pfa_GO(α) = 2 (1 + α/n)^-n − Pfa_SO(α)
# * OS (the k-th smallest of the N cells):  Pfa = Π_{i=0}^{k-1} (N − i) / (N − i + α)
#
# GO holds the false-alarm rate at clutter edges, SO resolves closely spaced
# targets, OS tolerates interfering targets in the training cells.  Near the
# ends of the range axis a side may be short; the cells that exist are used
# (cell averaging when a side is missing).


def _solve(pfa_of, pfa, hi=1.0):
    """α with ``pfa_of(α) = pfa`` for a decreasing ``pfa_of`` (bisection in log α)."""
    while pfa_of(hi) > pfa and hi < 1e12:
        hi *= 4.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pfa_of(mid) > pfa:
            lo = mid
        else:
            hi = mid
    return hi


@lru_cache(maxsize=4096)
def cfar_alpha(mode, left, right, pfa, rank=0.75):
    """Threshold multiplier of a CFAR detector with ``left``/``right`` training
    cells: it multiplies the cells' mean (CA, GO, SO) or the chosen order
    statistic (OS)."""
    n = left + right
    if n <= 0:
        return math.inf
    if mode == 'os-cfar':
        k = min(n, max(1, int(round(rank * n))))
        return _solve(lambda a: math.prod((n - i) / (n - i + a) for i in range(k)), pfa)
    if mode in ('go-cfar', 'so-cfar') and left == right:
        m = left

        def smallest(a):
            return 2.0 * sum(math.comb(m - 1 + k, k) * (2.0 + a / m) ** -(m + k) for k in range(m))
        if mode == 'so-cfar':
            return _solve(smallest, pfa)
        return _solve(lambda a: 2.0 * (1.0 + a / m) ** -m - smallest(a), pfa)
    return n * (pfa ** (-1.0 / n) - 1.0)


def cfar_threshold(power, mode, guard, train, pfa, rank=0.75):
    """CFAR threshold of every cell of ``power`` (rows × range cells), along range."""
    power = np.atleast_2d(np.asarray(power, float))
    rows, n = power.shape
    idx = np.arange(n)
    a_lo, a_hi = np.clip(idx - guard - train, 0, n), np.clip(idx - guard, 0, n)
    b_lo, b_hi = np.clip(idx + guard + 1, 0, n), np.clip(idx + guard + train + 1, 0, n)
    left, right = a_hi - a_lo, b_hi - b_lo
    pairs = {(int(l), int(r)) for l, r in zip(left, right)}
    lookup = {pair: cfar_alpha(mode, pair[0], pair[1], float(pfa), float(rank)) for pair in pairs}
    alpha = np.array([lookup[(int(l), int(r))] for l, r in zip(left, right)])
    if mode == 'os-cfar':
        threshold = np.empty_like(power)
        padded = np.pad(power, ((0, 0), (guard + train, guard + train)), constant_values=np.nan)
        for row in range(rows):
            windows = np.lib.stride_tricks.sliding_window_view(padded[row], 2 * (guard + train) + 1)
            cells = np.concatenate([windows[:, :train], windows[:, train + 2 * guard + 1:]], axis=1)
            count = np.sum(np.isfinite(cells), axis=1)
            k = np.clip(np.rint(rank * count).astype(int), 1, np.maximum(count, 1)) - 1
            ordered = np.sort(np.where(np.isfinite(cells), cells, np.inf), axis=1)
            threshold[row] = alpha * ordered[np.arange(n), k]
        return np.where(left + right > 0, threshold, np.inf)
    c = np.concatenate([np.zeros((rows, 1)), np.cumsum(power, axis=1)], axis=1)
    left_sum, right_sum = c[:, a_hi] - c[:, a_lo], c[:, b_hi] - c[:, b_lo]
    with np.errstate(divide='ignore', invalid='ignore'):
        both = (left > 0) & (right > 0) & (left == right)
        mean = (left_sum + right_sum) / np.maximum(left + right, 1)
        if mode in ('go-cfar', 'so-cfar'):
            pick = np.maximum if mode == 'go-cfar' else np.minimum
            sided = pick(left_sum / np.maximum(left, 1), right_sum / np.maximum(right, 1))
            level = np.where(both, sided, mean)
        else:
            level = mean
        return np.where(left + right > 0, alpha * level, np.inf)


def detect(power, threshold, limit=24):
    """Detections of a range-Doppler map, strongest first: ``[(row, column), …]``.

    Cells above their threshold are grouped into 8-connected clusters and
    each cluster reports its strongest cell, so one target is one detection
    however many range and Doppler cells (mainlobe and sidelobes) its echo
    lifts above the threshold.
    """
    from scipy import ndimage
    power = np.atleast_2d(power)
    hits = power > threshold
    if not hits.any():
        return []
    labels, count = ndimage.label(hits, structure=np.ones((3, 3), bool))
    peaks = ndimage.maximum_position(power, labels, index=np.arange(1, count + 1))
    peaks = sorted(peaks, key=lambda rc: power[rc], reverse=True)[:limit]
    return [(int(r), int(c)) for r, c in peaks]


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
    # Every Doppler bin inside the gate is kept: measurements use the full
    # resolution (thinning the axis here used to discard the peak bin); the
    # display reduces it by max-pooling.
    return rd[valid],ranges,velocity[valid],{'complete_intervals':count,'used_intervals':slow,'actual_pri_s':length/fs,'coherent_time_s':slow*length/fs,'velocity_resolution_mps':C/carrier_hz/(2*slow*length/fs),'range_bin_m':float(abs(ranges[1]-ranges[0])) if len(ranges)>1 else None}
