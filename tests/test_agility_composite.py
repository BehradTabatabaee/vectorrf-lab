"""Frequency agility (hopping / sweeping) and composite (multi-signal) transmitters."""
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from vectorrf.agility import FrequencyAgility, MAX_HOPS, hop_count, hop_frequencies, oscillator, schedule, summary
from vectorrf.bands import band_warnings, design_plan, nominal_band
from vectorrf.engine import calibrated_analyze
from vectorrf.models import SpectrumConfig, TxConfig
from vectorrf.signal import component_seed, generate
from vectorrf.source import source_record


EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples" / "designs").glob("*.json"))


def example(name):
    return json.loads((Path(__file__).resolve().parents[1] / "examples" / "designs" / f"{name}.json").read_text(encoding="utf-8"))["tx"]


def inst_freq(x, fs):
    return np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * np.pi)


HOP = dict(enabled=True, mode="hop", start_hz=-100e3, stop_hz=100e3, step_hz=50e3, dwell_s=1e-3, off_s=0.2e-3)


# ---------------------------------------------------------------- agility --

def test_hop_grid_is_inclusive_and_exact_for_the_900mhz_example():
    cfg = FrequencyAgility(enabled=True, start_hz=-50e6, stop_hz=50e6, step_hz=100e3)
    assert hop_count(cfg) == 1001
    f = hop_frequencies(cfg)
    assert f[0] == -50e6 and f[-1] == pytest.approx(50e6) and np.allclose(np.diff(f), 100e3)


@pytest.mark.parametrize("order, expected", [("up", [-1, 0, 1]), ("down", [1, 0, -1]), ("up-down", [-1, 0, 1, 0])])
def test_hop_orders(order, expected):
    cfg = FrequencyAgility(enabled=True, start_hz=-1, stop_hz=1, step_hz=1, order=order)
    assert hop_frequencies(cfg).tolist() == expected


def test_random_plan_is_a_seeded_permutation_of_the_grid():
    base = FrequencyAgility(enabled=True, start_hz=0, stop_hz=63e3, step_hz=1e3, plan="random", seed=5)
    f = hop_frequencies(base)
    assert sorted(f.tolist()) == [k * 1e3 for k in range(64)] and not np.all(np.diff(f) > 0)
    assert np.array_equal(f, hop_frequencies(base))
    assert not np.array_equal(f, hop_frequencies(base.model_copy(update={"seed": 6})))


def test_each_dwell_sits_exactly_on_its_hop_frequency_and_off_time_is_gated():
    fs = 1e6
    lo, gate = oscillator(FrequencyAgility(**HOP), fs, 6000)
    for h, f in enumerate([-100e3, -50e3, 0.0, 50e3, 100e3]):
        dwell = slice(h * 1200, h * 1200 + 1000)
        assert np.allclose(inst_freq(lo[dwell], fs), f, atol=1e-6)
        assert np.all(gate[dwell] == 1) and np.all(gate[h * 1200 + 1000:(h + 1) * 1200] == 0)


def test_oscillator_is_identical_however_the_timeline_is_split():
    cfg = FrequencyAgility(**HOP)
    whole, g_whole = oscillator(cfg, 1e6, 9000, 12345)
    parts = [oscillator(cfg, 1e6, n, 12345 + a) for a, n in ((0, 1), (1, 4321), (4322, 4678))]
    assert np.array_equal(np.concatenate([p[0] for p in parts]), whole)
    assert np.array_equal(np.concatenate([p[1] for p in parts]), g_whole)


def test_continuous_phase_has_no_steps_at_hop_boundaries():
    cfg = FrequencyAgility(enabled=True, start_hz=-100e3, stop_hz=100e3, step_hz=50e3, dwell_s=1e-3)
    lo, _ = oscillator(cfg, 1e6, 5000)
    f = inst_freq(lo, 1e6)
    # The instantaneous frequency only ever takes the hop values: no phase jumps.
    assert np.all(np.min(np.abs(f[:, None] - np.array([-100e3, -50e3, 0, 50e3, 100e3])[None, :]), axis=1) < 1e-6)


def test_coherent_hops_have_the_phase_of_free_running_oscillators():
    cfg = FrequencyAgility(enabled=True, start_hz=-100e3, stop_hz=100e3, step_hz=50e3, dwell_s=1e-3, phase="coherent")
    lo, _ = oscillator(cfg, 1e6, 3000, 7_000_000)
    k = 7_000_000 + np.arange(3000)
    # Absolute sample 7e6 lies in hop (7e6 // 1000) % 5 = 0 → -100 kHz for the first 1000 samples.
    assert np.allclose(lo[:1000], np.exp(-2j * np.pi * 100e3 * k[:1000] / 1e6), atol=1e-9)


def test_phase_stays_exact_after_an_enormous_number_of_periods():
    cfg = FrequencyAgility(enabled=True, start_hz=-123_456.7, stop_hz=98_765.4, step_hz=11_111.1, dwell_s=3.7e-4, off_s=1.1e-4)
    s = schedule(cfg, 2e6)
    near, _ = oscillator(cfg, 2e6, 4000, 0)
    far, _ = oscillator(cfg, 2e6, 4000, s.period * 10**12)
    # One period advances the DDS by an exact integer phase; the frequency
    # pattern repeats exactly and the phase differs by a constant rotation.
    rotation = far[0] / near[0]
    assert abs(abs(rotation) - 1) < 1e-12 and np.allclose(far, near * rotation, atol=1e-9)


@pytest.mark.parametrize("order, first, last", [("up", -200e3, 200e3), ("down", 200e3, -200e3)])
def test_sweeps_are_linear_and_continuous(order, first, last):
    cfg = FrequencyAgility(enabled=True, mode="sweep", start_hz=-200e3, stop_hz=200e3, dwell_s=2e-3, order=order)
    lo, gate = oscillator(cfg, 1e6, 2000)
    f = inst_freq(lo, 1e6)
    # The DDS ramps by (last - first) / 2000 per sample and reaches ``last``
    # at the end of the dwell.
    assert np.allclose(f, first + (last - first) * np.arange(len(f)) / 2000, atol=1e-6) and np.all(gate == 1)


def test_triangle_sweep_returns_to_its_start():
    cfg = FrequencyAgility(enabled=True, mode="sweep", start_hz=-200e3, stop_hz=200e3, dwell_s=1e-3, order="up-down")
    f = inst_freq(oscillator(cfg, 1e6, 2001)[0], 1e6)
    assert f[999] == pytest.approx(200e3, abs=500) and f[1999] == pytest.approx(-200e3, abs=500)
    assert summary(cfg, 1e6)["sweep_rate_hz_s"] == pytest.approx(400e3 / 1e-3)


def test_raised_cosine_ramps_shape_both_dwell_edges():
    cfg = FrequencyAgility(**{**HOP, "ramp_s": 50e-6})
    _, gate = oscillator(cfg, 1e6, 1200)
    assert gate[0] < 0.01 and gate[49] > 0.99 and np.all(gate[50:950] == 1) and gate[999] < 0.01
    assert np.allclose(gate[:50], gate[999:949:-1])


def test_schedule_offset_delays_the_pattern():
    cfg = FrequencyAgility(**{**HOP, "offset_s": 0.5e-3})
    lo, gate = oscillator(cfg, 1e6, 2000)
    assert np.all(gate[:500] == 0) and np.allclose(inst_freq(lo[500:1500], 1e6), -100e3)


@pytest.mark.parametrize("bad", [
    dict(enabled=True, mode="sweep", start_hz=5, stop_hz=5),
    dict(enabled=True, plan="list", frequencies_hz=[]),
    dict(enabled=True, start_hz=0, stop_hz=1e9, step_hz=1),
])
def test_agility_validation(bad):
    with pytest.raises(ValidationError):
        FrequencyAgility(**bad)


def test_hop_limit_constant_matches_validation():
    FrequencyAgility(enabled=True, start_hz=0, stop_hz=MAX_HOPS - 1, step_hz=1)
    with pytest.raises(ValidationError):
        FrequencyAgility(enabled=True, start_hz=0, stop_hz=MAX_HOPS, step_hz=1)


# ------------------------------------------------------- agile transmitter --

def test_hopping_tone_reads_its_configured_power_while_on_and_noise_stays_continuous():
    cfg = TxConfig(waveform="tone", sample_rate=1e6, duration=6e-3, power_dbm=-7, snr_db=20, agility=HOP)
    g = generate(cfg)
    x = g.samples.astype(complex)
    # Within one dwell the tone power equals the configured on-power reference …
    dwell = x[1200:2200]
    tone = np.abs(np.mean(dwell * np.exp(2j * np.pi * 50e3 * np.arange(1200, 2200) / 1e6)))
    assert 20 * np.log10(tone / g.metadata["power_reference_rms"]) == pytest.approx(0, abs=0.1)
    # … while the transmitter AWGN continues through the off-time.
    off = x[1000:1200]
    assert np.mean(np.abs(off) ** 2) == pytest.approx(g.metadata["power_reference_rms"] ** 2 / 100, rel=0.4)
    assert g.metadata["agility"]["segments"] == 5 and g.metadata["agility"]["duty"] == pytest.approx(1 / 1.2)


def test_hopping_continues_across_live_captures():
    cfg = TxConfig(waveform="tone", sample_rate=1e6, duration=1.7e-3, snr_db=None,
                   agility=dict(enabled=True, start_hz=-100e3, stop_hz=100e3, step_hz=50e3, dwell_s=1e-3))
    a, b = source_record(cfg, 0).samples, source_record(cfg, 1).samples
    joined = np.r_[a, b].astype(complex)
    whole = generate(cfg.model_copy(update={"duration": 3.4e-3})).samples.astype(complex)
    assert np.allclose(joined, whole, atol=1e-5)


def test_agility_extent_must_stay_inside_nyquist():
    with pytest.raises(ValidationError, match="Nyquist"):
        TxConfig(sample_rate=1e6, center_hz=200e3, agility=dict(enabled=True, mode="sweep", start_hz=0, stop_hz=400e3))


# --------------------------------------------------------------- composite --

TWO = [{"label": "Tone", "waveform": "tone", "power_dbm": -10.0},
       {"label": "QPSK", "waveform": "qpsk", "power_dbm": -20.0, "center_hz": 200e3, "symbol_rate": 50e3, "data_source": "pn"}]


def test_composite_components_read_their_own_absolute_powers():
    cfg = TxConfig(waveform="composite", sample_rate=1e6, duration=0.05, snr_db=None, components=TWO)
    g = generate(cfg)
    sp = calibrated_analyze(g.samples, g, SpectrumConfig(rbw_hz=100, max_fft=1 << 16, averaging=8))
    assert sp.peak_hz == 0 and sp.peak_db == pytest.approx(-10, abs=0.05)
    band = calibrated_analyze(g.samples, g, SpectrumConfig(rbw_hz=100, max_fft=1 << 16, averaging=8,
                                                           channels=[{"center_hz": 200e3, "width_hz": 100e3}])).channels[0]
    assert band["power_db"] == pytest.approx(-20, abs=0.3)
    assert g.metadata["total_power_dbm"] == pytest.approx(10 * np.log10(10 ** -1 + 10 ** -2))
    assert [c["label"] for c in g.metadata["components"]] == ["Tone", "QPSK"]


def test_components_share_the_parent_clock():
    cfg = TxConfig(waveform="composite", sample_rate=2e6, duration=0.01,
                   components=[{"waveform": "tone", "sample_rate": 5e5, "duration": 1.0}])
    assert cfg.components[0].sample_rate == 2e6 and cfg.components[0].duration == 0.01


def test_a_copied_composite_keeps_one_clock():
    """``model_copy`` skips validation, so the parent clock must still reach every component."""
    cfg = TxConfig(waveform="composite", sample_rate=1e6, duration=0.002,
                   components=[{"waveform": "tone"}, {"waveform": "qpsk", "center_hz": 100e3, "symbol_rate": 50e3}])
    longer = cfg.model_copy(update={"duration": 0.01})
    assert len(generate(longer).samples) == 10_000


def test_identical_components_carry_independent_data_and_noise():
    same = {"waveform": "qpsk", "symbol_rate": 50e3, "data_source": "pn"}
    assert component_seed(7, 0, 7) != component_seed(7, 1, 7)
    cfg = TxConfig(waveform="composite", sample_rate=1e6, duration=0.01, snr_db=None, components=[same, same])
    g = generate(cfg)
    # Coherent addition would double the amplitude (+6 dB); independent data adds in power.
    assert np.mean(np.abs(g.samples) ** 2) == pytest.approx(cfg.amplitude ** 2, rel=0.1)


def test_disabled_components_are_skipped_and_one_must_remain():
    on = TxConfig(waveform="composite", components=[TWO[0], {**TWO[1], "enabled": False}])
    assert len(generate(on).metadata["components"]) == 1
    with pytest.raises(ValidationError, match="Enable at least one"):
        TxConfig(waveform="composite", components=[{**TWO[0], "enabled": False}])


@pytest.mark.parametrize("components, message", [
    ([], "at least one component"),
    ([{"waveform": "composite", "components": [{"waveform": "tone"}]}], "cannot themselves be composite"),
])
def test_composite_validation(components, message):
    with pytest.raises(ValidationError, match=message):
        TxConfig(waveform="composite", components=components)


def test_composite_scale_is_deterministic_across_captures():
    cfg = TxConfig(waveform="composite", sample_rate=1e6, duration=0.004, snr_db=None,
                   components=[TWO[0], {**TWO[1], "burst_enabled": True, "burst_period_s": 0.003, "burst_duty": 0.3}])
    refs = {source_record(cfg, k).metadata["power_reference_rms"] for k in range(4)}
    assert len(refs) == 1


def test_composite_tone_is_continuous_across_captures():
    cfg = TxConfig(waveform="composite", sample_rate=1e6, duration=1e-3, snr_db=None,
                   components=[{"waveform": "tone", "center_hz": 123_456.0}])
    joined = np.r_[source_record(cfg, 0).samples, source_record(cfg, 1).samples].astype(complex)
    assert np.allclose(inst_freq(joined, 1e6), 123_456.0, atol=1e-3)


def test_composite_with_hopping_component_and_swept_parent():
    cfg = TxConfig(waveform="composite", sample_rate=2e6, duration=0.004, snr_db=None,
                   agility=dict(enabled=True, mode="sweep", start_hz=-200e3, stop_hz=200e3, dwell_s=2e-3),
                   components=[{"waveform": "tone", "agility": dict(enabled=True, start_hz=-100e3, stop_hz=100e3,
                                                                     step_hz=100e3, dwell_s=1e-3)}])
    f = inst_freq(generate(cfg).samples.astype(complex), 2e6)
    # Frequencies add: hop (-100 kHz) + sweep start (-200 kHz) at t = 0.
    assert f[0] == pytest.approx(-300e3, abs=500)
    plan = design_plan(cfg)
    assert plan["components"][0]["agility"]["segments"] == 3 and plan["agility"]["mode"] == "sweep"


def test_composite_carrier_extent_includes_component_hops():
    with pytest.raises(ValidationError, match="Nyquist"):
        TxConfig(waveform="composite", sample_rate=1e6, center_hz=100e3,
                 components=[{"waveform": "tone", "center_hz": 300e3,
                              "agility": dict(enabled=True, start_hz=0, stop_hz=150e3, step_hz=50e3)}])


# ---------------------------------------------- plan, bands, example designs --

def test_nominal_bands_follow_textbook_widths():
    qpsk = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, rolloff=0.35, center_hz=100e3)
    assert nominal_band(qpsk) == pytest.approx((100e3 - 33_750, 100e3 + 33_750))
    fm = TxConfig(waveform="fm", message_freq_hz=5e3, fm_deviation_hz=25e3)
    assert nominal_band(fm) == pytest.approx((-30e3, 30e3))
    hop = TxConfig(waveform="tone", agility=dict(enabled=True, start_hz=-100e3, stop_hz=100e3, step_hz=50e3))
    assert nominal_band(hop) == pytest.approx((-100e3, 100e3))


def test_band_warnings_flag_aliasing_components():
    cfg = TxConfig(waveform="composite", sample_rate=1e6, components=[
        {"label": "Edge", "waveform": "qpsk", "symbol_rate": 100e3, "center_hz": 450e3}])
    assert any("Edge" in w and "aliases" in w for w in band_warnings(cfg))


def test_design_plan_reports_totals_and_normalized_config():
    plan = design_plan(TxConfig(waveform="composite", components=TWO))
    assert plan["total_power_dbm"] == pytest.approx(10 * np.log10(0.11))
    assert plan["config"]["components"][1]["sample_rate"] == plan["sample_rate"]
    assert plan["components"][1]["band_hz"] == pytest.approx([200e3 - 33_750, 200e3 + 33_750])


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_every_example_design_validates_and_generates(path):
    cfg = TxConfig.model_validate(json.loads(path.read_text(encoding="utf-8"))["tx"])
    g = generate(cfg)
    assert np.isfinite(g.samples).all() and len(g.samples) == round(cfg.sample_rate * cfg.duration)
    assert not band_warnings(cfg), band_warnings(cfg)


def test_fhss_example_is_the_requested_900mhz_to_1ghz_plan():
    cfg = TxConfig.model_validate(example("fhss-qpsk-900m"))
    plan = design_plan(cfg)
    assert plan["agility"]["segments"] == 1001
    assert cfg.rf_center_hz + plan["agility"]["min_hz"] == pytest.approx(900e6)
    assert cfg.rf_center_hz + plan["agility"]["max_hz"] == pytest.approx(1e9)
    assert plan["agility"]["dwell_s"] == pytest.approx(100e-6) and plan["agility"]["off_s"] == pytest.approx(20e-6)


def test_tone_plus_qpsk_example_places_a_10mhz_qpsk_at_20mhz():
    cfg = TxConfig.model_validate(example("tone-plus-qpsk"))
    band = design_plan(cfg)["components"][1]["band_hz"]
    assert (band[0] + band[1]) / 2 == pytest.approx(20e6) and band[1] - band[0] == pytest.approx(10e6, rel=1e-9)
    assert band[1] < cfg.sample_rate / 2  # the whole band (and its filter skirt) clears Nyquist


@pytest.mark.parametrize("second, expected_db", [
    ({"waveform": "tone"}, 20 * np.log10(2)),                                   # a duplicated tone: amplitudes add
    ({"waveform": "tone", "phase_offset_deg": 90}, 10 * np.log10(2)),           # in quadrature: powers add
    ({"waveform": "tone", "phase_offset_deg": 180, "power_dbm": -6}, 20 * np.log10(1 - 10 ** (-6 / 20))),
    ({"waveform": "tone", "center_hz": 100e3}, 10 * np.log10(2)),               # another frequency: independent
    ({"waveform": "qpsk", "data_source": "pn"}, None),
])
def test_composite_total_power_is_what_the_output_measures(second, expected_db):
    """Independent components add in power; identical deterministic copies add
    as amplitudes (a duplicated tone reads +6 dB, not +3 dB)."""
    first = {"waveform": second["waveform"], **({"data_source": "pn"} if second["waveform"] == "qpsk" else {})}
    cfg = TxConfig(waveform="composite", sample_rate=1e6, duration=0.02, components=[first, second])
    total = design_plan(cfg)["total_power_dbm"]
    g = generate(cfg)
    measured = 10 * np.log10(np.mean(np.abs(g.samples.astype(complex)) ** 2) / g.metadata["power_reference_rms"] ** 2) + g.metadata["power_dbm"]
    assert total == pytest.approx(measured, abs=0.05)
    if expected_db is not None:
        assert total == pytest.approx(expected_db, abs=0.01)
