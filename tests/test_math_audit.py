"""Regression tests for the 1.3 mathematical audit.

Each test pins an analytic property (code sidelobes, calibration, ENBW,
alignment, estimator bias) rather than a snapshot of current output, so a
future change that breaks the mathematics fails loudly.
"""
import binascii
from fractions import Fraction

import numpy as np
import pytest
from scipy import signal

from vectorrf.dsp import phase_cycles, rrc_taps
from vectorrf.measurements import zero_span
from vectorrf.models import ChannelConfig, RxConfig, SpectrumConfig, TxConfig, WaveformKind as W
from vectorrf.modem import (bits_per_symbol, evm_rms, fit_constellation, map_bits, reference_constellation,
                            slice_symbols, _int_to_bits)
from vectorrf.rx import demodulate, receive_frame
from vectorrf.signal import _BARKER, _costas_permutation, _polyphase_code, generate
from vectorrf.models import ChannelBand
from vectorrf.spectrum import _enbw_bins, _window, analyze, bandwidth_metrics
from vectorrf.sync import apply_channel


def _psl_db(code):
    c = np.asarray(code, complex)
    ac = np.abs(np.correlate(c, c, "full"))
    return 20 * np.log10(np.max(np.delete(ac, len(c) - 1)) / ac[len(c) - 1])


# ---------------------------------------------------------------------------
# Radar codes


@pytest.mark.parametrize("m", [3, 4, 5, 6, 8])
def test_p1_matches_literature_definition_and_frank_sidelobes(m):
    """P1 (Lewis & Kretschmer): phi = -(pi/M)[M-(2g+1)][gM+k], group-major order."""
    cfg = TxConfig(waveform="p1", polyphase_order=m)
    code = _polyphase_code(W.P1, cfg)
    g, k = np.divmod(np.arange(m * m), m)
    expected = np.exp(-1j * np.pi / m * (m - (2 * g + 1)) * (g * m + k))
    np.testing.assert_allclose(code, expected, atol=1e-12)
    # P1 and Frank codes share aperiodic autocorrelation magnitude.
    frank = _polyphase_code(W.FRANK, cfg)
    assert _psl_db(code) == pytest.approx(_psl_db(frank), abs=1e-9)


@pytest.mark.parametrize("m", [2, 4, 6, 8])
def test_frank_and_p2_sidelobes_and_definition(m):
    cfg = TxConfig(waveform="frank", polyphase_order=m)
    frank = _polyphase_code(W.FRANK, cfg)
    g, k = np.divmod(np.arange(m * m), m)
    np.testing.assert_allclose(frank, np.exp(2j * np.pi * g * k / m), atol=1e-12)
    p2 = _polyphase_code(W.P2, cfg)
    assert np.allclose(np.abs(p2), 1)
    assert _psl_db(p2) <= _psl_db(frank) + 1e-9


@pytest.mark.parametrize("n", [16, 31, 64, 127])
def test_p3_p4_zadoff_chu_are_unimodular_and_zc_is_cazac(n):
    base = TxConfig(waveform="p3", phase_code_chips=n)
    for kind in (W.P3, W.P4, W.ZADOFF_CHU):
        code = _polyphase_code(kind, base.model_copy(update={"zc_root": 1 if n % 2 == 0 else 5}))
        assert np.allclose(np.abs(code), 1.0)
    # Zadoff-Chu has ideal periodic autocorrelation for gcd(u, N) = 1.
    zc = _polyphase_code(W.ZADOFF_CHU, base.model_copy(update={"zc_root": 1 if n % 2 == 0 else 3}))
    periodic = np.abs(np.fft.ifft(np.abs(np.fft.fft(zc)) ** 2))
    assert np.max(periodic[1:]) < 1e-9 * periodic[0]


def test_barker_codes_have_unit_sidelobes():
    for length, code in _BARKER.items():
        ac = np.correlate(code, code, "full")
        assert np.max(np.abs(np.delete(ac, length - 1))) == 1


@pytest.mark.parametrize("requested", range(4, 33))
def test_welch_costas_arrays_are_valid_and_not_longer_than_requested(requested):
    p = list(_costas_permutation(requested))
    assert sorted(p) == list(range(len(p)))
    assert len(p) <= requested
    for d in range(1, len(p)):
        diffs = [p[i + d] - p[i] for i in range(len(p) - d)]
        assert len(set(diffs)) == len(diffs)


# ---------------------------------------------------------------------------
# Modem mapping, slicing, EVM


@pytest.mark.parametrize("mod", ["bpsk", "qpsk", "8psk", "16qam", "64qam", "16apsk", "32apsk"])
def test_constellations_are_unit_power_and_bits_roundtrip(mod):
    ref = reference_constellation(mod)
    assert np.mean(np.abs(ref) ** 2) == pytest.approx(1.0, rel=1e-6)
    assert len(np.unique(np.round(ref, 6))) == 1 << bits_per_symbol(mod)
    bits = np.random.default_rng(1).integers(0, 2, bits_per_symbol(mod) * 3000).astype(np.uint8)
    _, out = slice_symbols(map_bits(bits, mod), mod)
    np.testing.assert_array_equal(out, bits)


@pytest.mark.parametrize("mod", ["qpsk", "8psk", "16qam", "64qam"])
def test_psk_qam_maps_are_gray_coded(mod):
    ref = reference_constellation(mod)
    k = bits_per_symbol(mod)
    labels = _int_to_bits(np.arange(1 << k), k).reshape(-1, k)
    d = np.abs(ref[:, None] - ref[None, :])
    np.fill_diagonal(d, np.inf)
    neighbours = np.argwhere(np.isclose(d, d.min(), atol=1e-6))
    assert all(np.sum(labels[a] != labels[b]) == 1 for a, b in neighbours)


@pytest.mark.parametrize("mod", ["qpsk", "16qam", "64qam", "32apsk"])
def test_evm_of_ideal_symbols_is_zero_for_any_finite_data_mix(mod):
    """RMS normalization alone left 0.03 % on 16-QAM; the LS gain fit is exact.

    The common-phase residual is kept realistic (post carrier recovery): a
    rotation large enough to push outer QAM points across decision boundaries
    cannot be removed by any decision-directed EVM measurement.
    """
    bits = np.random.default_rng(4).integers(0, 2, bits_per_symbol(mod) * 64).astype(np.uint8)
    s = map_bits(bits, mod) * 3.7 * np.exp(1j * 0.02)
    assert evm_rms(s, mod)["evm_rms_percent"] < 1e-4


@pytest.mark.parametrize("snr_db", [30, 20, 15])
def test_evm_tracks_snr_without_normalization_bias(snr_db):
    rng = np.random.default_rng(9)
    s = map_bits(rng.integers(0, 2, 2 * 40000), "qpsk")
    n = (rng.normal(size=s.size) + 1j * rng.normal(size=s.size)) * np.sqrt(10 ** (-snr_db / 10) / 2)
    measured = evm_rms(s + n, "qpsk")["evm_rms_percent"]
    assert measured == pytest.approx(100 * 10 ** (-snr_db / 20), rel=0.03)


def test_constellation_fit_scale_only_keeps_phase_errors_visible():
    s = map_bits(np.random.default_rng(2).integers(0, 2, 2000), "qpsk")
    z, idx = fit_constellation(s * 2 * np.exp(1j * np.deg2rad(30)), reference_constellation("qpsk"), complex_gain=False)
    assert np.mean(np.abs(z)) == pytest.approx(1.0, rel=1e-6)  # amplitude exact despite the rotation
    assert np.allclose(np.angle(z * np.conj(s)), np.deg2rad(30), atol=1e-6)


def test_rrc_is_unit_energy_symmetric_and_nyquist_after_matching():
    for sps in (2, 4, 7, 16):
        for beta in (0.2, 0.35, 0.5, 1.0):
            h = rrc_taps(sps, beta, 24).astype(float)
            assert np.sum(h * h) == pytest.approx(1.0, rel=1e-5)
            np.testing.assert_allclose(h, h[::-1], atol=1e-7)
            rc = np.convolve(h, h)
            c = len(rc) // 2
            assert np.max(np.abs(rc[c::sps][1:8])) < 2e-3 * rc[c]


# ---------------------------------------------------------------------------
# Spectrum calibration and resolution


@pytest.mark.parametrize("window, enbw", [("rect", 1.0), ("hann", 1.5), ("blackmanharris", 2.0044), ("flattop", 3.7702)])
def test_window_enbw_matches_textbook_values(window, enbw):
    assert _enbw_bins(_window(window, 8192)) == pytest.approx(enbw, rel=2e-3)


@pytest.mark.parametrize("snr_db", [None, 30, 10, 0])
def test_transmit_power_is_signal_power_independent_of_added_noise(snr_db):
    """A 0 dBm tone at 0 dB SNR used to read -3 dBm because noise entered the reference."""
    g = generate(TxConfig(waveform="tone", sample_rate=1e6, duration=.05, power_dbm=0, snr_db=snr_db))
    r = analyze(g.samples, 1e6, SpectrumConfig(rbw_hz=1000, averaging=16),
                reference_power_dbm=0, reference_rms=g.metadata["power_reference_rms"])
    assert r.peak_db == pytest.approx(0.0, abs=0.05)


def test_burst_reads_configured_power_while_on():
    tx = TxConfig(waveform="tone", sample_rate=1e6, duration=.02, power_dbm=7, snr_db=None,
                  burst_enabled=True, burst_period_s=.002, burst_duty=.25)
    g = generate(tx)
    on = g.samples[np.abs(g.samples) > 0][:1000]
    r = analyze(on, 1e6, SpectrumConfig(rbw_hz=2000, averaging=1), reference_power_dbm=7,
                reference_rms=g.metadata["power_reference_rms"])
    assert r.peak_db == pytest.approx(7.0, abs=0.05)
    assert np.mean(np.abs(g.samples) > 0) == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize("n, rbw", [(4000, 10), (4000, 100), (20000, 10), (512, 1)])
def test_short_capture_never_zero_pads_under_the_window(n, rbw):
    """A 4 ms capture with 10 Hz RBW used to read a 0 dBm tone as -74.6 dBm."""
    x = np.ones(n, np.complex64)
    r = analyze(x, 1e6, SpectrumConfig(rbw_hz=rbw, averaging=1, window="hann"), reference_power_dbm=0, reference_rms=1)
    assert r.peak_db == pytest.approx(0.0, abs=0.01)
    assert r.rbw_limited_by_capture
    assert r.nfft <= n  # the window never extends past the data (the FFT beyond it is zero-padded)
    # The reported RBW is the window's equivalent noise bandwidth (Hann: 1.5 bins of the window length).
    k = 1.5 * 1e6 / r.nfft
    assert r.rbw_actual_hz == pytest.approx(k, rel=1e-6)


def test_noise_psd_and_power_per_rbw_are_calibrated():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=1 << 19) + 1j * rng.normal(size=1 << 19)) / np.sqrt(2)
    psd = analyze(x, 1e6, SpectrumConfig(rbw_hz=1000, averaging=128))
    assert np.mean(10 ** (psd.psd_db / 10)) == pytest.approx(1e-6, rel=0.02)  # unit power / 1 MHz
    per_bin = analyze(x, 1e6, SpectrumConfig(rbw_hz=1000, averaging=128), reference_power_dbm=0, reference_rms=1)
    assert 10 * np.log10(np.mean(10 ** (per_bin.psd_db / 10))) == pytest.approx(10 * np.log10(per_bin.rbw_actual_hz / 1e6), abs=0.1)


def test_bandwidth_metrics_interpolate_crossings():
    f = np.linspace(-1e6, 1e6, 4001)
    sigma = 50e3
    p = np.exp(-f ** 2 / (2 * sigma ** 2))
    bw3, bw99, peak_hz, _ = bandwidth_metrics(f, p)
    assert bw3 == pytest.approx(2 * sigma * np.sqrt(2 * np.log(10 ** 0.3)), rel=2e-3)
    assert bw99 == pytest.approx(2 * 2.5758293 * sigma, rel=5e-3)
    assert peak_hz == 0


def test_frequency_axes_keep_float64_precision_at_ghz_rates():
    fs = 4e9
    x = np.exp(2j * np.pi * 1.234567891e9 * np.arange(1 << 16) / fs).astype(np.complex64)
    r = analyze(x, fs, SpectrumConfig(rbw_hz=200e3, averaging=1, max_fft=1 << 16))
    assert r.freq_hz.dtype == np.float64
    # float64 near 2 GHz resolves 2.4e-7 Hz (float32: 256 Hz, more than the bin spacing at finer RBW).
    assert np.all(np.diff(r.freq_hz) == pytest.approx(fs / r.fft_size, rel=1e-10))


# ---------------------------------------------------------------------------
# Exact time / phase


def test_phase_cycles_are_exact_rational_at_long_simulated_time():
    fs, f = 2e9, 7.3e8 + 0.123
    start = 10_000_000 * 2_000_000  # 10^4 s of simulated time at 2 GS/s
    got = phase_cycles(f, fs, 8, start)
    exact = [float((Fraction(f) * (start + k) / Fraction(fs)) % 1) for k in range(8)]
    np.testing.assert_allclose(got, exact, atol=1e-12)


def test_carrier_offset_and_analog_messages_are_continuous_across_captures():
    for waveform in ("tone", "fm", "am", "pm", "ssb", "multitone"):
        cfg = TxConfig(waveform=waveform, sample_rate=1e6, duration=.003, center_hz=123_456.7, snr_db=None,
                       normalize_output=False, amplitude=1, message_freq_hz=3_333.3, tones_hz=[-71_000, 13_000])
        a = generate(cfg, start_sample=0).samples
        b = generate(cfg, start_sample=len(a)).samples
        full = generate(cfg.model_copy(update={"duration": .006})).samples
        if waveform == "multitone":
            # RMS normalization is per capture; compare shape only.
            a, b, full = a / np.std(a), b / np.std(b), full / np.std(full)
        np.testing.assert_allclose(np.r_[a, b], full, atol=2e-5, err_msg=waveform)


def test_fm_uses_the_exact_phase_integral():
    cfg = TxConfig(waveform="fm", sample_rate=1e6, duration=.004, fm_deviation_hz=20_000,
                   message_freq_hz=1_000, snr_db=None, normalize_output=False, amplitude=1)
    y = generate(cfg).samples
    t = np.arange(len(y)) / 1e6
    expected = np.exp(1j * (20_000 / 1_000) * (1 - np.cos(2 * np.pi * 1_000 * t)))
    np.testing.assert_allclose(y, expected, atol=2e-6)
    inst = zero_span(y, 1e6, duration=.004)["frequency_hz"]
    assert np.max(np.abs(inst[1:])) == pytest.approx(20_000, rel=2e-3)


def test_ssb_is_one_sided_without_record_edge_ringing():
    cfg = TxConfig(waveform="ssb", sample_rate=1e6, duration=.004, message_freq_hz=10_000, snr_db=None)
    for sideband, sign in (("usb", 1), ("lsb", -1)):
        y = generate(cfg.model_copy(update={"sideband": sideband})).samples
        f = np.angle(y[1:] * np.conj(y[:-1])) * 1e6 / (2 * np.pi)
        np.testing.assert_allclose(f, sign * 10_000, atol=0.05)


def test_channel_cfo_is_phase_continuous_between_captures():
    ch = ChannelConfig(snr_db=None, freq_offset_hz=1234.5, phase_offset_deg=10)
    x = np.ones(5000, np.complex64)
    a = apply_channel(x, 1e6, ch, start_sample=0)
    b = apply_channel(x, 1e6, ch, start_sample=5000)
    full = apply_channel(np.ones(10000, np.complex64), 1e6, ch, start_sample=0)
    np.testing.assert_allclose(np.r_[a, b], full, atol=1e-5)


# ---------------------------------------------------------------------------
# Demodulator alignment


@pytest.mark.parametrize("mod", ["2fsk", "4fsk", "msk", "gmsk"])
@pytest.mark.parametrize("sps", [2, 3, 4, 8])
def test_cpfsk_demodulator_is_error_free_at_low_samples_per_symbol(mod, sps):
    """A one-sample discriminator misalignment gave 25 % noise-free BER at 2 sps."""
    fs = 200_000
    rs = fs / sps
    dev = rs * (0.75 if mod == "4fsk" else 0.5)
    tx = TxConfig(waveform=mod, sample_rate=fs, symbol_rate=rs, duration=.02, data_source="pn",
                  pn_order=9, snr_db=None, fsk_deviation_hz=dev)
    g = generate(tx)
    r = demodulate(g.samples, RxConfig(modulation=mod, sample_rate=fs, symbol_rate=rs, fsk_deviation_hz=dev))
    n = min(len(r.bits), len(g.tx_bits)) - 8
    assert n > 100
    assert np.mean(r.bits[:n] != g.tx_bits[:n]) == 0


@pytest.mark.parametrize("payload", ["bpsk", "qpsk", "8psk", "16qam", "oqpsk", "dqpsk"])
def test_frame_blocks_decode_every_bit_in_place(payload):
    """Linear frame blocks used to lose their first filter_span/2 symbols."""
    bits = 96 if payload != "8psk" else 99
    blocks = [{"name": "Sync", "role": "preamble", "modulation": "bpsk", "bit_count": 32,
               "bits": "10101010101010101101001110010001", "symbol_rate": 50_000},
              {"name": "Payload", "role": "payload", "modulation": payload, "bit_count": bits, "symbol_rate": 50_000}]
    tx = TxConfig(waveform="qpsk", sample_rate=500_000, symbol_rate=50_000, duration=.01, frame_enabled=True,
                  frame_mode="blocks", frame_blocks=blocks, snr_db=None, data_source="pn", pn_order=9)
    g = generate(tx)
    rx = RxConfig(modulation="qpsk", sample_rate=500_000, symbol_rate=50_000, frame_enabled=True, frame_blocks=blocks,
                  energy_threshold_dbm=-60, preamble_threshold=.6)
    acq = receive_frame(g.samples, rx, power_dbm=0, reference_rms=g.metadata["power_reference_rms"])
    sync, pay = acq["decoded_blocks"]
    assert sync["bits"] == list(g.tx_bits[:32])
    assert len(pay["bits"]) == bits
    # Interior symbols are exact; the two edge symbols see truncated-pulse ISI.
    k = bits_per_symbol({"oqpsk": "qpsk", "dqpsk": "qpsk"}.get(payload, payload))
    inner = slice(k, bits - k)
    assert np.array_equal(np.asarray(pay["bits"])[inner], g.tx_bits[32:32 + bits][inner])


def test_negative_fsk_timing_phase_wraps_instead_of_clamping():
    tx = TxConfig(waveform="2fsk", sample_rate=200_000, symbol_rate=20_000, duration=.01, data_source="pn",
                  snr_db=None, fsk_deviation_hz=10_000)
    g = generate(tx)
    base = RxConfig(modulation="2fsk", sample_rate=200_000, symbol_rate=20_000, fsk_deviation_hz=10_000)
    a = demodulate(g.samples, base.model_copy(update={"timing_phase": -0.2}))
    b = demodulate(g.samples, base.model_copy(update={"timing_phase": 0.8 - 1.0}))
    np.testing.assert_array_equal(a.bits, b.bits)
    assert not np.array_equal(a.bits[:50], demodulate(g.samples, base).bits[:50])


def test_frame_crc_matches_ccitt_false_reference():
    cfg = TxConfig(data_source="bits", custom_bits="1011", frame_enabled=True, preamble_bits="11", payload_bits=24, frame_crc=True)
    from vectorrf.signal import _data_bits
    bits = _data_bits(cfg, 2 + 24 + 16, np.random.default_rng(0))
    payload = np.packbits(bits[2:26]).tobytes()
    assert np.packbits(bits[26:42]).tobytes() == binascii.crc_hqx(payload, 0xFFFF).to_bytes(2, "big")
    assert binascii.crc_hqx(b"123456789", 0xFFFF) == 0x29B1  # CRC-16/CCITT-FALSE check value


def test_awgn_noise_bandwidth_has_no_filter_startup_transient():
    g = generate(TxConfig(waveform="awgn", sample_rate=1e6, duration=.02, noise_bandwidth_hz=200e3))
    x = g.samples
    head = np.mean(np.abs(x[:64]) ** 2)
    body = np.mean(np.abs(x[2000:]) ** 2)
    assert head == pytest.approx(body, rel=0.5)
    f, p = signal.welch(x, fs=1e6, nperseg=1024, return_onesided=False)
    inband = np.mean(p[np.abs(f) < 80e3])
    outband = np.mean(p[np.abs(f) > 200e3])
    assert 10 * np.log10(inband / outband) > 40


@pytest.mark.parametrize("window", ["hann", "blackmanharris", "kaiser"])
def test_a_tone_reads_its_level_and_shape_wherever_it_falls_between_bins(window):
    """The FFT is zero-padded 4x: a tone between bins loses < 0.1 dB (Hann used
    to lose up to 1.42 dB) and draws the same RBW shape at any frequency."""
    fs, n = 1e6, 33_333
    widths = []
    for f0 in (125_000.0, 123_456.0, 100_488.3, 200_000.0):
        x = np.exp(2j * np.pi * f0 * np.arange(n) / fs).astype(np.complex64)
        r = analyze(x, fs, SpectrumConfig(rbw_hz=2000, window=window, averaging=2), reference_power_dbm=0, reference_rms=1)
        assert r.fft_size == 4 * r.nfft
        assert r.peak_db == pytest.approx(0.0, abs=0.1), (window, f0)
        assert abs(r.peak_hz - f0) <= fs / r.fft_size / 2 + 1e-6
        widths.append(r.bandwidth_3db_hz)
    assert max(widths) / min(widths) < 1.05, widths  # the same -3 dB width at every frequency


def test_zero_padding_keeps_integrated_noise_calibrated():
    """Zero padding multiplies the bins per RBW; integrated noise stays exact."""
    rng = np.random.default_rng(3)
    x = ((rng.normal(size=1 << 18) + 1j * rng.normal(size=1 << 18)) / np.sqrt(2)).astype(np.complex64)
    band = ChannelBand(center_hz=0, width_hz=200e3)
    r = analyze(x, 1e6, SpectrumConfig(rbw_hz=1000, averaging=64, channels=[band]), reference_power_dbm=0, reference_rms=1)
    assert r.channels[0]["power_db"] == pytest.approx(10 * np.log10(0.2), abs=0.15)
