"""Small reproducible workload benchmark; no hardware real-time guarantee."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from statistics import median
from time import perf_counter
import json
import platform

from vectorrf.models import TxConfig, RxConfig, SpectrumConfig
from vectorrf.signal import generate
from vectorrf.engine import frame_product
from vectorrf.rx import demodulate
from vectorrf.blind import analyze as classify
from vectorrf.radar_sim import RadarExperiment, simulate


def elapsed_ms(fn, repeat=3):
    fn()
    measurements = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        measurements.append(1000 * (perf_counter() - start))
    return round(median(measurements), 3)


def bench():
    rows = []
    for fs in (1_000_000, 5_000_000):
        cfg = TxConfig(waveform="qpsk", sample_rate=fs, duration=.06,
                       symbol_rate=50_000, data_source="pn", snr_db=25)
        g = generate(cfg)
        spec = SpectrumConfig(rbw_hz=2000, averaging=2, max_fft=32768)
        rx = RxConfig(sample_rate=fs, symbol_rate=cfg.symbol_rate)
        rows.append({"sample_rate": fs, "samples": len(g.samples),
                     "generate_ms": elapsed_ms(lambda: generate(cfg)),
                     "spectrum_frame_ms": elapsed_ms(lambda: frame_product(g, spec, 0)),
                     "receiver_ms": elapsed_ms(lambda: demodulate(g.samples, rx)),
                     "classifier_ms": elapsed_ms(lambda: classify(g.samples, fs))})
    scene = RadarExperiment()
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "method": "median of 3 after one warm-up; milliseconds; excludes HTTP/JSON/browser",
              "communications": rows,
              "waveform_array_radar_ms": elapsed_ms(lambda: simulate(scene))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    bench()
