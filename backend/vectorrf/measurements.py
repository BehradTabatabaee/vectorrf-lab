"""Time-domain measurements at acquisition rate, reduced only after DSP."""
import numpy as np

def zero_span(samples, fs, offset=0., duration=.002, max_points=900):
    start = min(len(samples)-2,max(0,int(offset*fs)))
    end = min(len(samples),max(start+2,start+int(duration*fs)))
    x=np.asarray(samples[start:end])
    phase=np.unwrap(np.angle(x))
    fm=np.r_[0.,np.angle(x[1:]*np.conj(x[:-1]))*fs/(2*np.pi)]
    amplitude=np.abs(x)
    # Phase/frequency of near-zero samples is undefined; mask rather than invent frequency.
    valid=amplitude>max(float(amplitude.max())*.01,1e-12)
    fm[~valid]=0
    ix=np.unique(np.linspace(0,len(x)-1,min(max_points,len(x)),dtype=int))
    return {'time_s':((ix+start)/fs).tolist(),'i':x.real[ix].tolist(),'q':x.imag[ix].tolist(),
        'amplitude':amplitude[ix].tolist(),'phase_deg':np.rad2deg(phase[ix]).tolist(),
        'frequency_hz':fm[ix].tolist(),'valid':valid[ix].tolist(),
        'sample_rate':fs,'capture_start_s':start/fs,'capture_duration_s':len(x)/fs,
        'rms':float(np.sqrt(np.mean(amplitude**2))), 'peak':float(amplitude.max())}
