"""Small reproducible workload benchmark; no hardware real-time guarantee."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from statistics import median
from time import perf_counter
import json
import platform
import tempfile
from pathlib import Path

import numpy as np

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


def communications(fs, samples, symbol_rate):
    cfg = TxConfig(waveform="qpsk", sample_rate=fs, duration=samples / fs,
                   symbol_rate=symbol_rate, data_source="pn", snr_db=25)
    g = generate(cfg)
    spec = SpectrumConfig(rbw_hz=fs / 20_000, averaging=2, max_fft=1 << 18)
    rx = RxConfig(sample_rate=fs, symbol_rate=cfg.symbol_rate, timing_recovery="mueller-muller", coarse_cfo=True)
    return {"sample_rate": fs, "samples": len(g.samples),
            "generate_ms": elapsed_ms(lambda: generate(cfg)),
            "spectrum_frame_ms": elapsed_ms(lambda: frame_product(g, spec, 0)),
            "receiver_ms": elapsed_ms(lambda: demodulate(g.samples, rx)),
            "classifier_ms": elapsed_ms(lambda: classify(g.samples, fs))}


def spectrum_fft_sizes(fs=2e9):
    g = generate(TxConfig(waveform="16qam", sample_rate=fs, symbol_rate=fs / 10,
                          duration=(1 << 21) / fs, data_source="pn", snr_db=30))
    rows = []
    for nfft in (1 << 15, 1 << 17, 1 << 19, 1 << 20):
        spec = SpectrumConfig(rbw_hz=1.5 * fs / nfft, averaging=2, max_fft=nfft)
        rows.append({"nfft": nfft, "rbw_hz": round(1.5 * fs / nfft),
                     "frame_ms": elapsed_ms(lambda: frame_product(g, spec, 0))})
    return rows


def recording_read(samples=1 << 23):
    from vectorrf.iqfile import RecordingReader, SampleFormat, inspect
    x = (np.random.default_rng(1).normal(size=samples) * 0.2 + 1j * np.random.default_rng(2).normal(size=samples) * 0.2).astype(np.complex64)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("cf32_le", "ci16_le", "cu8"):
            fmt = SampleFormat.parse(name)
            path = Path(tmp) / f"bench.{name}"
            path.write_bytes(fmt.encode(x)[0].tobytes())
            reader = RecordingReader(inspect(path, name, 2e9))
            window = 1 << 21
            ms = elapsed_ms(lambda: reader.read(samples // 3, window))
            rows.append({"format": name, "window_samples": window, "read_ms": ms,
                         "msps_decoded": round(window / ms / 1e3, 1)})
            reader.close()
    return rows


def agility_and_composite(samples=1 << 23, fs=125e6):
    """Hop/sweep synthesis, a composite, and a zoom-FFT frame."""
    from vectorrf.agility import FrequencyAgility, oscillator
    hop = FrequencyAgility(enabled=True, start_hz=-50e6, stop_hz=50e6, step_hz=100e3, dwell_s=100e-6, off_s=20e-6, ramp_s=2e-6)
    sweep = FrequencyAgility(enabled=True, mode="sweep", start_hz=-50e6, stop_hz=50e6, dwell_s=1e-3)
    hop_ms = elapsed_ms(lambda: oscillator(hop, fs, samples, 10 * samples))
    sweep_ms = elapsed_ms(lambda: oscillator(sweep, fs, samples, 10 * samples))
    design = Path(__file__).resolve().parent / "designs" / "tone-plus-qpsk.json"
    composite = TxConfig.model_validate(json.loads(design.read_text(encoding="utf-8"))["tx"])
    composite = composite.model_copy(update={"duration": 0.02})
    g = generate(composite)
    zoom = SpectrumConfig(center_hz=20e6, span_hz=1e6, rbw_hz=200, averaging=1, max_fft=1 << 17)
    return {"oscillator_samples": samples,
            "hop_oscillator_ms": hop_ms, "hop_msps": round(samples / hop_ms / 1e3, 1),
            "sweep_oscillator_ms": sweep_ms, "sweep_msps": round(samples / sweep_ms / 1e3, 1),
            "composite_samples": len(g.samples), "composite_generate_ms": elapsed_ms(lambda: generate(composite)),
            "zoom_1mhz_span_frame_ms": elapsed_ms(lambda: frame_product(g, zoom, 0))}


def bench():
    scene = RadarExperiment()
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "method": "median of 3 after one warm-up; milliseconds; excludes HTTP/JSON/browser",
              "communications": [communications(1e6, 60_000, 50_000), communications(5e6, 300_000, 50_000),
                                 communications(2e9, 2_000_000, 200e6)],
              "spectrum_frame_at_2gsps": spectrum_fft_sizes(),
              "recording_window_read": recording_read(),
              "agility_composite_zoom": agility_and_composite(),
              "waveform_array_radar_ms": elapsed_ms(lambda: simulate(scene))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    bench()
