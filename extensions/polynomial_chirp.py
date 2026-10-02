"""Editable polynomial frequency law; selected in the custom waveform editor."""
import numpy as np

def waveform(context):
    t=context['t'];p=context['parameters']
    # Instantaneous frequency is f0 + f1*t + f2*t**2 + f3*t**3.
    phase=(p.get('f0',-100000)*t+p.get('f1',2.5e9)*t**2/2
           +p.get('f2',0)*t**3/3+p.get('f3',0)*t**4/4)
    envelope=np.where(t<p.get('width_s',80e-6),p.get('amplitude',1),0)
    return envelope*np.exp(2j*np.pi*phase)
