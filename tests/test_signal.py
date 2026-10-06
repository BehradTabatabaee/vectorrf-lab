import numpy as np
import pytest

from vectorrf.models import TxConfig, WaveformKind
from vectorrf.signal import generate


@pytest.mark.parametrize("waveform", list(WaveformKind))
def test_all_waveforms_generate_finite_iq(waveform):
    cfg = TxConfig(
        waveform=waveform,
        sample_rate=200_000,
        duration=0.012,
        symbol_rate=10_000,
        message_freq_hz=2_000,
        fsk_deviation_hz=8_000,
        fm_deviation_hz=8_000,
        sweep_bw_hz=50_000,
        chirp_start_hz=-30_000,
        chirp_stop_hz=30_000,
        pulse_width_s=40e-6,
        pri_s=200e-6,
        tones_hz=[-80_000, 15_000, 45_000],
        snr_db=25 if waveform != WaveformKind.AWGN else None,
        components=[{"waveform": "qpsk", "symbol_rate": 10_000}, {"waveform": "tone", "center_hz": 40_000}]
        if waveform == WaveformKind.COMPOSITE else [],
    )
    g = generate(cfg)
    assert g.samples.dtype == np.complex64
    assert len(g.samples) == round(cfg.sample_rate * cfg.duration)
    assert np.isfinite(g.samples.real).all()
    assert np.isfinite(g.samples.imag).all()


def test_python310_compatible_enum_behavior():
    assert isinstance(WaveformKind.QPSK.value, str)
    assert WaveformKind('qpsk') is WaveformKind.QPSK


def test_radar_code_and_train_metadata():
    cfg = TxConfig(
        waveform='p4', sample_rate=500_000, duration=0.01,
        pulse_width_s=80e-6, pri_s=300e-6, phase_code_chips=31,
        sweep_bw_hz=100_000, pri_mode='staggered', pri_stagger=[1.0, 1.1],
        snr_db=None,
    )
    g = generate(cfg)
    assert g.metadata['code_length'] == 31
    assert g.metadata['pri_mode'] == 'staggered'
    assert g.metadata['pulse_count'] >= 2


def test_fmcw_triangle_is_finite():
    cfg = TxConfig(
        waveform='fmcw', sample_rate=500_000, duration=0.005,
        pri_s=200e-6, sweep_bw_hz=100_000, fmcw_shape='triangle', snr_db=None,
    )
    g = generate(cfg)
    assert g.metadata['fmcw_shape'] == 'triangle'
    assert np.isfinite(g.samples).all()


def test_arbitrary_radar_segment_builder():
    cfg = TxConfig(
        waveform='radar-arbitrary', sample_rate=500_000, duration=0.005,
        arbitrary_duration_us=[20, 20, 40],
        arbitrary_start_freq_hz=[-50_000, 50_000, 0],
        arbitrary_stop_freq_hz=[50_000, -50_000, 0],
        arbitrary_amplitude=[1.0, 0.5, 0.0],
        arbitrary_phase_deg=[0, 90, 0],
        snr_db=None,
    )
    g = generate(cfg)
    assert g.metadata['segment_count'] == 3
    assert g.metadata['radar_family'] == 'arbitrary'
    assert np.isfinite(g.samples).all()
