"""Editable notch filter with filter memory carried across Receiver captures."""
import numpy as np
from scipy.signal import iirnotch,lfilter

def process(samples,sample_rate,context):
    p=context['parameters'];memory=context['state']
    frequency=p.get('frequency_hz',10000);quality=p.get('Q',30)
    if not 0<frequency<sample_rate/2 or quality<=0:
        raise ValueError('Notch frequency must lie between DC and Nyquist; Q must be positive')
    b,a=iirnotch(frequency,quality,fs=sample_rate)
    zi=memory.get('zi',np.zeros(max(len(a),len(b))-1,dtype=complex))
    out,memory['zi']=lfilter(b,a,samples,zi=zi)
    return out,sample_rate
