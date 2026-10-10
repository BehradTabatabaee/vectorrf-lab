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
