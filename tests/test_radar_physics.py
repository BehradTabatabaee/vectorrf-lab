"""Radar simulation physics: every quantity checked against its closed form.

Conventions under test (``radar_sim``):

* targets move as p(t) = p0 + v0·t + ½·a·t², v(t) = v0 + a·t;
* radial velocity is v·u with u pointing from the radar to the target, so a
  receding target is positive and its Doppler shift f_d = -2·v_r/λ negative;
* the monostatic radar equation P_r = P_t·G_t·G_r·λ²·σ / ((4π)³·R⁴) in watts,
  thermal noise k·T₀·B·F with B = sample rate;
* leakage and clutter are relative to the transmitted signal (dBc).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vectorrf.radar_processing import RadarProcessing
from vectorrf.radar_sim import ArrayModel, RadarExperiment, SceneTarget, simulate
from vectorrf.programs import WaveProgram

C = 299_792_458.0
K_BOLTZMANN = 1.380649e-23

ISOTROPIC_1 = ArrayModel(geometry='linear', columns=1, element='isotropic')
ISOTROPIC_12 = ArrayModel(geometry='linear', columns=12, element='isotropic')


def experiment(**update) -> RadarExperiment:
    base = dict(antenna=ISOTROPIC_1, noise_db=-120, targets=[])
    base.update(update)
    return RadarExperiment(**base)


def _state(target: SceneTarget, t: float):
    p0, v0, a = (np.asarray(x, float) for x in (target.position, target.velocity, target.acceleration))
    return p0 + v0 * t + 0.5 * a * t * t, v0 + a * t


# --------------------------------------------------------------- kinematics

@pytest.mark.parametrize('t', [0.0, 0.37, 12.5, 600.0])
def test_truth_follows_constant_acceleration_kinematics(t):
    target = SceneTarget(position=[4000, -1500, 300], velocity=[-35, 12, -1], acceleration=[0.8, -0.3, 0.05], gain_db=0)
    d = simulate(experiment(targets=[target], time_s=t, max_range_m=40_000))
    p, v = _state(target, t)
    truth = d['truth'][0]
    assert truth['position'] == pytest.approx(p.tolist(), abs=1e-6)
    r = float(np.linalg.norm(p))
    assert truth['range_m'] == pytest.approx(r, rel=1e-12)
    assert truth['velocity_mps'] == pytest.approx(float(v @ (p / r)), abs=1e-9)
    assert truth['az_deg'] == pytest.approx(math.degrees(math.atan2(p[1], p[0])), abs=1e-9)
    assert truth['el_deg'] == pytest.approx(math.degrees(math.asin(p[2] / r)), abs=1e-9)


def test_receding_target_is_positive_and_approaching_negative():
    out = simulate(experiment(targets=[SceneTarget(position=[3000, 0, 0], velocity=[30, 0, 0], gain_db=0)], noise_db=-100))
    inc = simulate(experiment(targets=[SceneTarget(position=[3000, 0, 0], velocity=[-30, 0, 0], gain_db=0)], noise_db=-100))
    assert out['truth'][0]['velocity_mps'] == pytest.approx(30)
    assert inc['truth'][0]['velocity_mps'] == pytest.approx(-30)
    assert out['peak_velocity_mps'] > 0 > inc['peak_velocity_mps']


def test_tangential_motion_has_no_radial_velocity():
    d = simulate(experiment(targets=[SceneTarget(position=[3000, 0, 0], velocity=[0, 40, 0], gain_db=0)], noise_db=-100))
    assert d['truth'][0]['velocity_mps'] == pytest.approx(0, abs=1e-12)
    assert abs(d['peak_velocity_mps']) <= 5 + 1e-9  # the nearest Doppler bin (5 m/s spacing)


def test_range_changes_by_radial_velocity_times_elapsed_time():
    target = SceneTarget(position=[5000, 0, 0], velocity=[-50, 0, 0], gain_db=0)
    a = simulate(experiment(targets=[target], time_s=0.0))['truth'][0]['range_m']
    b = simulate(experiment(targets=[target], time_s=2.0))['truth'][0]['range_m']
    assert a - b == pytest.approx(100.0, abs=1e-9)


# ---------------------------------------------------------- range / Doppler

@pytest.mark.parametrize('fs', [1e6, 4e6, 20e6])
@pytest.mark.parametrize('rng', [600.0, 2345.0, 7777.0, 12_000.0])
def test_correlation_range_is_within_one_range_bin(fs, rng):
    d = simulate(experiment(sample_rate=fs, samples=8192, noise_db=-100,
                            targets=[SceneTarget(position=[rng, 0, 0], velocity=[0, 0, 0], gain_db=0)], max_range_m=15_000))
    bin_m = C / (2 * fs)
    assert abs(d['peak_range_m'] - rng) <= bin_m, (d['peak_range_m'], rng, bin_m)


@pytest.mark.parametrize('vr', [-80.0, -35.0, -10.0, 0.0, 10.0, 35.0, 80.0])
def test_correlation_doppler_matches_radial_velocity(vr):
    d = simulate(experiment(noise_db=-100, samples=16384,
                            targets=[SceneTarget(position=[3000, 0, 0], velocity=[vr, 0, 0], gain_db=0)]))
    spacing = 2 * 100 / (41 - 1)
    assert abs(d['peak_velocity_mps'] - vr) <= spacing / 2 + 1e-9


@pytest.mark.parametrize('method', ['pulse-doppler', 'fmcw'])
@pytest.mark.parametrize('vr', [-45.0, -15.0, 15.0, 45.0])
def test_coherent_processors_measure_signed_velocity(method, vr):
    pri, slope, width = 0.0002, 2e9, 0.00008
    phase = 'exp(j*2*pi*(-100000*t+1000000000*t*t))'
    expression = f'where(t < {width}, {phase}, 0)' if method == 'pulse-doppler' else phase
    cfg = experiment(samples=32768, program=WaveProgram(period_s=pri, expression=expression), noise_db=-100,
                     targets=[SceneTarget(position=[3000, 0, 0], velocity=[vr, 0, 0], gain_db=0)], max_range_m=12_000, max_velocity_mps=80,
                     processing=RadarProcessing(method=method, pri_s=pri, pulse_width_s=width, chirp_slope_hz_s=slope, doppler_window='hann'))
    d = simulate(cfg)
    resolution = d['processor_info']['velocity_resolution_mps']
    assert abs(d['peak_velocity_mps'] - vr) <= resolution, (d['peak_velocity_mps'], vr, resolution)
    # FMCW range carries the range-Doppler coupling c·f_d/(2k); both stay within a few range bins.
    assert d['peak_range_m'] == pytest.approx(3000, abs=450)


def test_two_targets_are_resolved_at_their_own_ranges():
    d = simulate(experiment(noise_db=-100, samples=8192, sample_rate=4e6, targets=[
        SceneTarget(name='A', position=[2000, 0, 0], velocity=[0, 0, 0], gain_db=0),
        SceneTarget(name='B', position=[6000, 0, 0], velocity=[0, 0, 0], gain_db=0)]))
    found = sorted(x['range_m'] for x in d['detections'][:4])
    bin_m = C / (2 * 4e6)
    assert any(abs(r - 2000) <= 1.5 * bin_m for r in found) and any(abs(r - 6000) <= 1.5 * bin_m for r in found), found


@pytest.mark.parametrize('fs', [1e6, 4e6, 20e6])
@pytest.mark.parametrize('rng', [900.0, 2000.0, 6000.0, 13_000.0])
def test_cfar_detects_a_lone_target_at_any_range_and_rate(fs, rng):
    """No blind zone at the profile ends, and guard cells cover the mainlobe at any oversampling."""
    d = simulate(experiment(sample_rate=fs, samples=16384, noise_db=-70, max_range_m=15_000,
                            targets=[SceneTarget(position=[rng, 0, 0], velocity=[0, 0, 0], gain_db=0)]))
    bin_m = C / (2 * fs)
    assert d['detections'], (fs, rng, d['warnings'])
    assert abs(d['detections'][0]['range_m'] - rng) <= max(1.5 * bin_m, 10.0), (d['detections'][0], rng)


@pytest.mark.parametrize('mode', ['ca-cfar', 'go-cfar', 'so-cfar', 'os-cfar'])
@pytest.mark.parametrize('pfa', [1e-2, 1e-3])
def test_cfar_false_alarm_rate_on_noise_is_the_requested_one(mode, pfa):
    """Each detector's multiplier is solved exactly for exponential (square-law) noise."""
    from vectorrf.radar_processing import cfar_threshold
    rng = np.random.default_rng(4)
    noise = rng.exponential(size=(300, 2000))
    threshold = cfar_threshold(noise, mode, 2, 16, pfa)
    inside = slice(18, 2000 - 18)
    assert np.mean(noise[:, inside] > threshold[:, inside]) == pytest.approx(pfa, rel=0.12)
    # Short sides near the ends keep the rate with the cells that exist.
    assert np.mean(noise[:, :18] > threshold[:, :18]) == pytest.approx(pfa, rel=0.5)


@pytest.mark.parametrize('detector', ['ca-cfar', 'go-cfar', 'so-cfar', 'os-cfar'])
def test_every_detector_finds_two_targets_with_their_velocities(detector):
    d = simulate(experiment(sample_rate=4e6, samples=16384, noise_db=-70, max_range_m=15_000,
                            processing=RadarProcessing(detector=detector), targets=[
        SceneTarget(name='A', position=[2000, 0, 0], velocity=[-20, 0, 0], gain_db=0),
        SceneTarget(name='B', position=[6000, 0, 0], velocity=[30, 0, 0], gain_db=0)]))
    bin_m = C / (2 * 4e6)
    found = [(x['range_m'], x['velocity_mps']) for x in d['detections'][:4]]
    for r, v in [(2000, -20), (6000, 30)]:
        assert any(abs(fr - r) <= 1.5 * bin_m and abs(fv - v) <= 6 for fr, fv in found), (detector, found)
    assert all(x['margin_db'] > 0 for x in d['detections'])


# ------------------------------------------------------------------ angles

@pytest.mark.parametrize('az', [-40.0, -17.0, 0.0, 12.0, 33.0])
def test_azimuth_estimate_matches_target_bearing(az):
    a = math.radians(az)
    target = SceneTarget(position=[3000 * math.cos(a), 3000 * math.sin(a), 0], velocity=[0, 0, 0], gain_db=0)
    d = simulate(experiment(antenna=ISOTROPIC_12, noise_db=-100, targets=[target]))
    assert d['azimuth_deg'] == pytest.approx(az, abs=1.01)  # the 1° search grid


def test_beam_steering_attenuates_off_axis_echo_by_the_array_factor():
    target = SceneTarget(position=[3000, 0, 0], velocity=[0, 0, 0], gain_db=0)
    on = simulate(experiment(antenna=ISOTROPIC_12, targets=[target], noise_db=-150))
    steered = ISOTROPIC_12.model_copy(update={'steer_az_deg': 30})
    off = simulate(experiment(antenna=steered, targets=[target], noise_db=-150))
    # Uniform 12-element λ/2 array: AF(θ) = |sin(Nψ/2)/(N sin(ψ/2))| with ψ = π(sin θ − sin θ₀).
    psi = math.pi * (0 - math.sin(math.radians(30)))
    af = abs(math.sin(12 * psi / 2) / (12 * math.sin(psi / 2)))
    # The transmit pattern and the receive beam both apply the array factor.
    ratio = off['rx_rms'] / on['rx_rms']
    assert ratio == pytest.approx(af * af, abs=0.02)


# ---------------------------------------------------------- power budgets

def test_radar_equation_echo_power_is_exact():
    target = SceneTarget(position=[4000, 0, 0], velocity=[0, 0, 0], rcs_m2=5.0, gain_db=0)
    cfg = experiment(power_model='radar-equation', power_w=2500, tx_gain_db=25, rx_gain_db=22, loss_db=3,
                     carrier_hz=5.6e9, targets=[target], noise_figure_db=0)
    lam = C / cfg.carrier_hz
    expected = 2500 * 10 ** ((25 + 22 - 3) / 10) * lam ** 2 * 5.0 / ((4 * math.pi) ** 3 * 4000 ** 4)
    d = simulate(cfg)
    assert d['truth'][0]['echo_power_db'] == pytest.approx(10 * math.log10(expected), abs=1e-9)


def test_radar_equation_follows_inverse_fourth_power():
    a = simulate(experiment(power_model='radar-equation', targets=[SceneTarget(position=[3000, 0, 0], velocity=[0, 0, 0], gain_db=0)]))
    b = simulate(experiment(power_model='radar-equation', targets=[SceneTarget(position=[6000, 0, 0], velocity=[0, 0, 0], gain_db=0)]))
    assert a['truth'][0]['echo_power_db'] - b['truth'][0]['echo_power_db'] == pytest.approx(40 * math.log10(2), abs=1e-9)


def test_received_echo_power_equals_the_budget():
    """With noise far below it, the beamformed echo carries P_r (unit-RMS waveform)."""
    target = SceneTarget(position=[3000, 0, 0], velocity=[0, 0, 0], gain_db=0)
    cfg = experiment(targets=[target], power_model='radar-equation', noise_figure_db=0, sample_rate=1e3 * 1e3,
                     program=WaveProgram(expression='exp(j*2*pi*10000*t)', period_s=0.001))
    d = simulate(cfg)
    expected_db = d['truth'][0]['echo_power_db']
    measured_db = 20 * math.log10(d['rx_rms'])
    assert measured_db == pytest.approx(expected_db, abs=0.05)


def test_thermal_noise_is_ktb_times_noise_figure():
    for nf in (0.0, 6.0):
        cfg = experiment(power_model='radar-equation', noise_figure_db=nf, sample_rate=2e6, samples=65536 // 8)
        d = simulate(cfg)
        expected = K_BOLTZMANN * 290 * 2e6 * 10 ** (nf / 10)
        assert d['rx_rms'] ** 2 == pytest.approx(expected, rel=0.06)


def test_relative_model_noise_and_r4_scaling():
    d = simulate(experiment(noise_db=-40, samples=16384))
    assert 20 * math.log10(d['rx_rms']) == pytest.approx(-40, abs=0.2)
    near = simulate(experiment(noise_db=-200, targets=[SceneTarget(position=[3000, 0, 0], velocity=[0, 0, 0], gain_db=0)]))
    far = simulate(experiment(noise_db=-200, targets=[SceneTarget(position=[6000, 0, 0], velocity=[0, 0, 0], gain_db=0)]))
    assert far['rx_rms'] / near['rx_rms'] == pytest.approx(0.25, rel=0.02)


def test_default_radar_equation_scene_detects_its_target():
    """Regression: −100 dB leakage and clutter used to be absolute watts and buried the echo at range 0."""
    d = simulate(RadarExperiment(power_model='radar-equation'))
    truth = d['truth'][0]
    assert abs(d['peak_range_m'] - truth['range_m']) <= 2 * C / (2 * 1e6)
    assert any(abs(x['range_m'] - truth['range_m']) <= 2 * C / (2 * 1e6) for x in d['detections'])


@pytest.mark.parametrize('model', ['relative', 'radar-equation'])
def test_leakage_is_relative_to_the_transmitted_signal(model):
    cfg = experiment(power_model=model, leakage_db=-60, noise_db=-200, noise_figure_db=0,
                     program=WaveProgram(expression='exp(j*2*pi*10000*t)', period_s=0.001))
    if model == 'radar-equation':
        cfg = cfg.model_copy(update={'sample_rate': 1e6})
    d = simulate(cfg)
    reference = cfg.power_w if model == 'radar-equation' else 1.0
    expected = reference * 1e-6  # unit-RMS waveform, −60 dBc
    noise = K_BOLTZMANN * 290 * cfg.sample_rate if model == 'radar-equation' else 1e-20
    assert d['rx_rms'] ** 2 == pytest.approx(expected + noise, rel=0.01)


def test_leakage_and_clutter_are_off_by_default():
    a = simulate(experiment(noise_db=-200))
    assert a['rx_rms'] < 1e-9
    assert RadarExperiment().leakage_db is None and RadarExperiment().clutter_db is None


def test_strong_leakage_is_reported():
    d = simulate(experiment(leakage_db=-3, targets=[SceneTarget(position=[3000, 0, 0], velocity=[0, 0, 0], gain_db=-20)]))
    assert any('leakage' in w for w in d['warnings'])


def test_swerling_one_has_unit_mean_power():
    target = SceneTarget(position=[3000, 0, 0], velocity=[0, 0, 0], gain_db=0, fluctuation='swerling1')
    steady = simulate(experiment(targets=[target.model_copy(update={'fluctuation': 'none'})], samples=512))['truth'][0]['echo_power_db']
    powers = [10 ** (simulate(experiment(targets=[target], capture_id=k, samples=512))['truth'][0]['echo_power_db'] / 10) for k in range(300)]
    mean = float(np.mean(powers)) / 10 ** (steady / 10)
    assert mean == pytest.approx(1.0, abs=0.17)  # 300 exponential draws: σ ≈ 0.058
    # Exponential: the variance equals the squared mean.
    assert float(np.std(powers)) / float(np.mean(powers)) == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------- determinism / edges

def test_same_capture_is_bit_identical_and_next_capture_is_not():
    cfg = experiment(noise_db=-40, targets=[SceneTarget()])
    a, b = simulate(cfg), simulate(cfg)
    assert a['rx'] == b['rx'] and a['rd_db'] == b['rd_db']
    c = simulate(cfg.model_copy(update={'capture_id': 1}))
    assert c['rx'] != a['rx']


@pytest.mark.parametrize('position', [[0, 0, 0], [1e-3, 0, 0], [0, 0, 5000], [-3000, 0, 0], [1e8, 0, 0]])
def test_degenerate_geometries_never_produce_nan(position):
    d = simulate(experiment(antenna=ISOTROPIC_12, targets=[SceneTarget(position=position, velocity=[1, 2, 3], gain_db=0)]))
    for key in ('rd_db', 'range_db', 'psd_db', 'angle_db', 'pattern_db'):
        assert np.all(np.isfinite(d[key])), key
    assert math.isfinite(d['peak_range_m']) and math.isfinite(d['peak_velocity_mps'])


def test_twelve_targets_maximum():
    RadarExperiment(targets=[SceneTarget()] * 12)
    with pytest.raises(ValueError):
        RadarExperiment(targets=[SceneTarget()] * 13)


@pytest.mark.parametrize('bad', [{'position': [float('nan'), 0, 0]}, {'velocity': [0, 0]}, {'rcs_m2': 0}, {'acceleration': [0, 0, 0, 0]}])
def test_invalid_targets_are_rejected(bad):
    with pytest.raises(ValueError):
        SceneTarget(**bad)


def test_a_delayed_periodic_record_is_band_limited_not_linearly_interpolated():
    """A Studio design, OFDM frame or recording echo starts between samples:
    the windowed-sinc delay is accurate up to 0.8 Nyquist (linear
    interpolation was off by -4.5 dB at 0.74 Nyquist)."""
    from vectorrf.radar_sim import periodic_slice
    fs, m = 1e6, 4096
    base = np.exp(2j * np.pi * 0.37 * np.arange(m))
    start = 123.456789 / fs
    ideal = np.exp(2j * np.pi * 0.37 * fs * (start + np.arange(1000) / fs))
    error = np.max(np.abs(periodic_slice(base, fs, fs, start, 1000) - ideal))
    assert 20 * np.log10(error) < -55
