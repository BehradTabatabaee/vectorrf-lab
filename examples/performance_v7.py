import time,json,statistics
from vectorrf.radar_sim import simulate,RadarExperiment
from vectorrf.models import TxConfig,RxConfig
from vectorrf.ofdm import OFDMDesign
from vectorrf.signal import generate
from vectorrf.rx import demodulate
from vectorrf.engine import reduce_spectrum
import numpy as np

def bench(fn):
 fn();a=[]
 for _ in range(9):
  t=time.perf_counter();fn();a.append((time.perf_counter()-t)*1000)
 return round(statistics.median(a),3)
d=OFDMDesign(enabled=True,modulation='16qam');tx=TxConfig(waveform='ofdm',duration=.06,ofdm_design=d);x=generate(tx).samples
out={'radar_8192_ms':bench(lambda:simulate(RadarExperiment())), 'ofdm_tx_60000_ms':bench(lambda:generate(tx)), 'ofdm_rx_60000_ms':bench(lambda:demodulate(x,RxConfig(modulation='ofdm',ofdm_design=d))),'reduce_32768_ms':bench(lambda:reduce_spectrum(np.arange(32768),np.random.default_rng(1).normal(size=32768)))}
print(json.dumps(out))
