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
    cfg = TxConfig(waveform=waveform, sample_rate=500_000, symbol_rate=25_000, duration=.06,
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
