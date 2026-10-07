import numpy as np
import pytest

from vectorrf.blind import analyze
from vectorrf.models import TxConfig
from vectorrf.signal import generate


@pytest.mark.parametrize("waveform, expected", [
    ("awgn", "awgn"),
    ("qpsk", "qpsk"),
    ("16qam", "16qam"),
    ("2fsk", "2fsk"),
    ("msk", "msk"),
    ("ofdm", "ofdm"),
    ("lfm", "lfm"),
    ("barker", "barker/pn-phase-coded"),
    ("fmcw", "fmcw"),
])
def test_blind_representative_generated_waveforms(waveform, expected):
    # Random payloads: the classifier assumes data that visits every
    # constellation point (a short repeating text uses only a subset of 16-QAM).
    cfg = TxConfig(waveform=waveform, sample_rate=500_000, symbol_rate=25_000, duration=.06, data_source="pn",
                   snr_db=None if waveform == "awgn" else 30, sweep_bw_hz=100_000,
                   pulse_width_s=40e-6, pri_s=200e-6)
    g = generate(cfg)
    r = analyze(g.samples, g.sample_rate)
    assert r.modulation == expected
    if waveform in {"qpsk", "16qam", "2fsk", "msk"}:
        assert r.symbol_rate_baud == pytest.approx(25_000, rel=.03)


def test_blind_surfaces_tone_ssb_ambiguity():
    cfg = TxConfig(waveform="tone", sample_rate=500_000, duration=.04, snr_db=35)
    r = analyze(generate(cfg).samples, cfg.sample_rate)
    assert r.modulation == "single-tone/ssb-like"
    assert any("indistinguishable" in n for n in r.notes)


# Regression matrix for the 1.3 classifier hardening.  Before it, carrier
# offsets turned PSK/QAM into "64qam", nested constellations made BPSK read as
# 8-PSK, 64-QAM was mistaken for OFDM, and noisy CPFSK fell into the linear
# branch.  Random (PN) payloads; rates scaled to show sample-rate invariance.
import itertools

LINEAR_CASES = list(itertools.product(["bpsk", "qpsk", "8psk", "16qam", "64qam"], [(1e6, 20), (4e6, 8)], [0.0, 0.004, 0.05]))


@pytest.mark.parametrize("mod, rate, offset", LINEAR_CASES)
def test_blind_linear_matrix_with_carrier_offset(mod, rate, offset):
    fs, sps = rate
    cfg = TxConfig(waveform=mod, sample_rate=fs, symbol_rate=fs / sps, duration=60_000 / fs, data_source="pn",
                   snr_db=20, center_hz=offset * fs)
    r = analyze(generate(cfg).samples, fs)
    assert r.modulation == mod, r.notes
    assert r.symbol_rate_baud == pytest.approx(fs / sps, rel=0.02)


CPFSK_CASES = list(itertools.product(["2fsk", "4fsk", "msk", "gmsk"], [(1e6, 20), (2e6, 8), (4e6, 40)], [15, 30]))


@pytest.mark.parametrize("mod, rate, snr", CPFSK_CASES)
def test_blind_cpfsk_matrix(mod, rate, snr):
    fs, sps = rate
    rs = fs / sps
    cfg = TxConfig(waveform=mod, sample_rate=fs, symbol_rate=rs, duration=.06, data_source="pn", snr_db=snr,
                   fsk_deviation_hz=rs * (0.75 if mod == "4fsk" else 1.0))
    r = analyze(generate(cfg).samples, fs)
    assert r.modulation == mod, r.notes
    assert r.symbol_rate_baud == pytest.approx(rs, rel=0.02)


@pytest.mark.parametrize("mod", ["fm", "pm"])
@pytest.mark.parametrize("snr", [15, 30])
def test_blind_analog_angle_modulation_is_not_reported_as_fsk(mod, snr):
    cfg = TxConfig(waveform=mod, sample_rate=1e6, duration=.06, message_freq_hz=7_000, fm_deviation_hz=50_000,
                   pm_deviation_rad=2, snr_db=snr)
    assert analyze(generate(cfg).samples, 1e6).modulation == "fm/pm-like"


@pytest.mark.parametrize("waveform", ["qpsk", "2fsk", "msk"])
def test_blind_text_payload_does_not_lock_to_data_periodicity(waveform):
    """Repeating ASCII creates genuine lines at submultiples of the symbol rate."""
    cfg = TxConfig(waveform=waveform, sample_rate=1e6, symbol_rate=50_000, duration=.06, data_source="text",
                   message="VECTORRF LAB ", snr_db=25)
    r = analyze(generate(cfg).samples, 1e6)
    assert r.modulation == waveform
    assert r.symbol_rate_baud == pytest.approx(50_000, rel=0.02)


def test_blind_reports_multiple_emitters_and_isolates_each():
    fs = 2_048_000
    qpsk = generate(TxConfig(waveform="qpsk", sample_rate=fs, symbol_rate=64_000, duration=.1, center_hz=-400e3,
                             data_source="pn", snr_db=None)).samples
    fsk = generate(TxConfig(waveform="2fsk", sample_rate=fs, symbol_rate=64_000, duration=.1, center_hz=250e3,
                            fsk_deviation_hz=40e3, data_source="pn", snr_db=None)).samples
    gate = (np.arange(len(fsk)) // (fs // 20)) % 2 == 0  # bursty second emitter
    x = 0.3 * qpsk + 0.15 * fsk * gate
    x = x + (np.random.default_rng(1).normal(size=len(x)) + 1j * np.random.default_rng(2).normal(size=len(x))) * 0.01
    combined = analyze(x, fs)
    assert len(combined.features["bands"]) == 2 and "separate occupied bands" in combined.notes[0]
    assert combined.confidence <= 0.6
    strongest = combined.features["bands"][0]
    q = analyze(x, fs, (strongest["center_hz"], strongest["width_hz"] * 1.1))
    assert q.modulation == "qpsk" and q.symbol_rate_baud == pytest.approx(64_000, rel=0.02)
    other = combined.features["bands"][1]
    b = analyze(x, fs, (other["center_hz"], other["width_hz"] * 1.1))
    assert b.modulation == "2fsk" and any("longest burst" in n for n in b.notes)
