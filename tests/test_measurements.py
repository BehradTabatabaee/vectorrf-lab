"""Zero span and modulation quality behave like an analyzer's: tuned, RBW
filtered, steady from sweep to sweep, and cheap enough to run live."""
import json

import numpy as np
import pytest

from vectorrf.engine import MeasurementSpec, measurement_product
from vectorrf.measurements import auto_bandwidth, buckets, zero_span
from vectorrf.models import TxConfig
from vectorrf.source import source_record

TWO_TONES = TxConfig(waveform="composite", sample_rate=200e6, duration=0.02, snr_db=40, components=[
    {"waveform": "tone", "label": "Tone", "center_hz": 0, "power_dbm": -10},
    {"waveform": "tone", "label": "Tone copy", "center_hz": 5e6, "power_dbm": -10}])


def _band(out, key):
    low = np.asarray(out[key], dtype=float)
    return low, np.asarray(out.get(f"{key}_max", out[key]), dtype=float)


def test_zero_span_of_a_tone_is_steady_and_calibrated():
    """The reported case: a tone and its copy 5 MHz away at 200 MS/s.  At the
    tone, the default RBW passes it alone: a flat envelope at -10 dBm, the
    same in every sweep, settled from the first sample to the last (the
    filter reads the stream beyond the capture; only the stream's very first
    sample has nothing before it)."""
    sweeps = [measurement_product(source_record(TWO_TONES, k), TWO_TONES, False, k,
                                  MeasurementSpec(duration_s=0.002, offset_s=0.003 * (k - 1))) for k in range(1, 4)]
    for z in sweeps:
        low, high = _band(z, "amplitude")
        assert z["bandwidth_auto"] and z["bandwidth_actual_hz"] == pytest.approx(auto_bandwidth(0.002))
        assert high.max() / low.min() < 1.01                    # flat at both edges too
        assert z["level_db"] == pytest.approx(-10, abs=0.02) and z["level_unit"] == "dBm"
        assert abs(z["mean_frequency_hz"]) < 5
        f_low, f_high = _band(z, "frequency_hz")
        assert np.nanmax(np.abs(np.r_[f_low, f_high])) < 2e3     # was ±0.5 MHz of full-band noise
    medians = [np.median(_band(z, "amplitude")[0]) for z in sweeps]
    assert max(medians) / min(medians) < 1.001


def test_zero_span_tunes_to_the_copy_and_full_band_shows_the_beat():
    record = source_record(TWO_TONES, 1)
    copy = measurement_product(record, TWO_TONES, False, 1, MeasurementSpec(center_hz=5e6))
    assert copy["level_db"] == pytest.approx(-10, abs=0.02) and abs(copy["mean_frequency_hz"]) < 5
    beat = measurement_product(record, TWO_TONES, False, 1, MeasurementSpec(full_band=True))
    low, high = _band(beat, "amplitude")
    # The 5 MHz beat is far faster than the display: every bucket spans it, null to peak.
    assert beat["bucketed"] and np.all(low < 0.1 * high) and np.ptp(high) < 0.05 * high.max()


def test_display_buckets_show_extremes_instead_of_aliasing():
    t = np.arange(400_000) / 200e6
    x = np.cos(2 * np.pi * 5e6 * t)
    low, high = buckets(x, 600)
    assert np.allclose(low, -1, atol=1e-3) and np.allclose(high, 1, atol=1e-3)


def test_zero_span_rbw_sets_noise_and_rejects_bands_beyond_nyquist():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)).astype(np.complex64) / np.sqrt(2)
    for rbw in (10e3, 100e3):
        z = zero_span(x, 10e6, duration=0.02, bandwidth_hz=rbw)
        # Unit-density noise through the Gaussian RBW: power = noise bandwidth / fs.
        assert z["rms"] ** 2 == pytest.approx(z["noise_bandwidth_hz"] / 10e6, rel=0.1)
        assert z["noise_bandwidth_hz"] == pytest.approx(1.0645 * rbw, rel=1e-3)
    with pytest.raises(ValueError, match="Nyquist"):
        zero_span(x, 10e6, center_hz=4.9e6, bandwidth_hz=1e6)


def test_quality_of_a_composite_measures_one_component():
    tx = TxConfig(waveform="composite", sample_rate=10e6, duration=0.02, snr_db=40, components=[
        {"waveform": "qpsk", "label": "Data", "center_hz": -2e6, "symbol_rate": 500e3, "data_source": "pn", "power_dbm": -10},
        {"waveform": "tone", "label": "Pilot", "center_hz": 1.5e6, "power_dbm": -20}])
    record = source_record(tx, 3)
    data = measurement_product(record, tx, False, 3, MeasurementSpec(quality=True, component=0))["quality"]
    assert data["kind"] == "evm" and data["component"] == 0 and data["metrics"]["evm_rms_percent"] < 3
    assert "component 1 (Data)" in data["method"]
    pilot = measurement_product(record, tx, False, 3, MeasurementSpec(quality=True, component=1))["quality"]
    m = pilot["metrics"]
    assert pilot["kind"] == "basic" and pilot["component"] == 1 and pilot["method"].startswith("Component 2 (Pilot)")
    assert m["center_hz"] == 1.5e6 and abs(m["mean_frequency_hz"]) < 1 and m["am_percent"] < 0.5
    json.dumps(pilot, allow_nan=False)


def test_every_sample_quality_sweeps_each_record_in_segments():
    """A segment of a record measures that segment, contiguous with the next."""
    tx = TxConfig(waveform="tone", sample_rate=1e6, duration=0.02, snr_db=40, center_hz=100e3)
    record = source_record(tx, 2)
    first = measurement_product(record, tx, False, 2, MeasurementSpec(quality=True), segment=(0, 8000))
    second = measurement_product(record, tx, False, 2, MeasurementSpec(quality=True), segment=(8000, 16000))
    assert first["capture_start_s"] == 0 and second["capture_start_s"] == pytest.approx(0.008)
    assert first["capture_duration_s"] == second["capture_duration_s"] == pytest.approx(0.008)
    for q in (first["quality"], second["quality"]):
        assert q["metrics"]["mean_frequency_hz"] == pytest.approx(0, abs=1) and q["metrics"]["am_percent"] < 0.5


# --------------------------------------------- modulation quality per update

def _gapped_updates(tx, updates, seed=0):
    """Quality updates as real time makes them: prefixes of captures a few apart."""
    from vectorrf.engine import _quality_samples
    m = MeasurementSpec(quality=True)
    size, memory = _quality_samples(tx, None, m), {}
    rng = np.random.default_rng(seed)
    capture, out = 3, []
    for _ in range(updates):
        capture += int(rng.integers(1, 4))
        out.append(measurement_product(source_record(tx, capture, limit=size), tx, False, capture, m, memory)["quality"])
    return size, out


def test_quality_updates_have_a_fixed_size():
    """Like a zero-span sweep: 512 new symbols plus the matched filters'
    transient, or 262,144 samples at high oversampling, not what fits in a
    frame (which left no headroom)."""
    from vectorrf.engine import QUALITY_UPDATE_SAMPLES, _quality_samples
    qpsk = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3)
    assert _quality_samples(qpsk, None, MeasurementSpec(quality=True)) == (2 * qpsk.filter_span_symbols + 512) * 20
    fast = TxConfig(waveform="qpsk", sample_rate=100e6, symbol_rate=50e3)
    assert _quality_samples(fast, None, MeasurementSpec(quality=True)) <= QUALITY_UPDATE_SAMPLES
    oqpsk = TxConfig(waveform="oqpsk", sample_rate=1e6, symbol_rate=50e3)
    assert _quality_samples(oqpsk, None, MeasurementSpec(quality=True)) == (2 * qpsk.filter_span_symbols + 1024) * 20


@pytest.mark.parametrize("waveform, rate, symbol_rate", [("64qam", 50e6, 1e6), ("16qam", 200e6, 5e6), ("32apsk", 10e6, 200e3)])
def test_dense_constellations_stay_locked_across_real_time_gaps(waveform, rate, symbol_rate):
    """Short tracked blocks a few captures apart: each block continues the
    previous one's scale (a block's own fit could settle on another match of
    the rings, 0.81 of the scale) and refines the carrier phase only across
    real gaps (a fraction of a degree over one symbol became hundreds of
    hertz); APSK probes 256 symbols, enough of its inner ring.  64-QAM used
    to lose lock within a few blocks and re-acquire, at 50-200 ms an update."""
    tx = TxConfig(waveform=waveform, sample_rate=rate, symbol_rate=symbol_rate, duration=min(524288, int(rate / 30)) / rate,
                  snr_db=40, data_source="pn", pn_order=15, pn_bits=32767)
    _, updates = _gapped_updates(tx, 30)
    blocks = [q["metrics"]["evm_all_percent"] for q in updates[3:]]
    assert max(blocks) < 1.6 and not any(q["telemetry"].get("reacquired") for q in updates[1:])
    # The record joins the blocks fitted to the constellation: it reads what each block reads.
    assert updates[-1]["metrics"]["symbol_count"] == 2048
    assert updates[-1]["metrics"]["evm_rms_percent"] < 1.3


def test_fast_slicers_decide_like_the_exhaustive_search():
    from vectorrf.modem import _nearest, reference_constellation
    from vectorrf.sync import _nearest_ref, decision
    rng = np.random.default_rng(3)
    for mod in ("bpsk", "qpsk", "8psk", "16qam", "64qam", "16apsk", "32apsk"):
        refs = reference_constellation(mod)
        z = (rng.standard_normal(5000) + 1j * rng.standard_normal(5000)) * 0.8
        np.testing.assert_array_equal(_nearest(z, refs), np.argmin(np.abs(z[:, None] - refs[None, :]), axis=1))
        nearest, points = decision(refs), refs.astype(np.complex128).tolist()
        assert all(nearest(complex(v)) == _nearest_ref(complex(v), points) for v in z[:1000]), mod


@pytest.mark.parametrize("n, up, down", [(70_000, 16, 1250), (262_000, 1, 125), (2_080_000, 1, 2500)])
def test_parallel_resampling_is_the_single_call(n, up, down):
    from scipy import signal as sig
    from vectorrf.dsp import resample
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    np.testing.assert_array_equal(resample(x, up, down), sig.resample_poly(x, up, down, window=("kaiser", 10.0)))
