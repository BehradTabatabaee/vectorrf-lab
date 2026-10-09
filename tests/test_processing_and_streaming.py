import numpy as np
import pytest
from scipy import signal
from vectorrf.models import TxConfig,RxConfig
from vectorrf.ofdm import OFDMDesign
from vectorrf.signal import generate
from vectorrf.rx import demodulate
from vectorrf.radar_processing import RadarProcessing,cancel_clutter
from vectorrf.radar_sim import RadarExperiment,SceneTarget,ArrayModel,simulate
from vectorrf.programs import WaveProgram


def test_ofdm_custom_output_fir_is_actual_convolution_across_captures():
    d=OFDMDesign(enabled=True,window_samples=4,window_kind='custom',window_ramp=[.05,.25,.7,.95],output_filter='custom',filter_i=[.1,.2,.4,.2,.1])
    cfg=TxConfig(waveform='ofdm',duration=.002,ofdm_design=d,normalize_output=False,amplitude=1,snr_db=None)
    raw=generate(cfg.model_copy(update={'ofdm_design':d.model_copy(update={'output_filter':'off'}),'duration':.004}),start_s=.01).samples
    shaped=generate(cfg,start_s=.012).samples
    np.testing.assert_allclose(shaped,signal.lfilter(d.filter_i,[1],raw)[2000:],atol=1e-6)


def test_ofdm_filter_delay_can_be_tuned_out_in_receiver():
    d=OFDMDesign(enabled=True,output_filter='custom',filter_i=[0,0,1])
    cfg=TxConfig(waveform='ofdm',ofdm_design=d,snr_db=None)
    r=demodulate(generate(cfg).samples,RxConfig(modulation='ofdm',ofdm_design=d,ofdm_timing_offset=2))
    assert r.metrics['evm_rms_percent']<.01


def test_custom_window_rejects_wrong_length():
    with pytest.raises(ValueError,match='ramp'):generate(TxConfig(waveform='ofdm',ofdm_design=OFDMDesign(enabled=True,window_samples=4,window_kind='custom',window_ramp=[.2])))


@pytest.mark.parametrize('method',['pulse-doppler','fmcw'])
def test_radar_processor_measures_range_and_velocity_from_samples(method):
    pri=.0002;slope=2e9;width=.00008
    phase='exp(j*2*pi*(-100000*t+1000000000*t*t))'
    expression=f'where(t < {width}, {phase}, 0)' if method=='pulse-doppler' else phase
    cfg=RadarExperiment(samples=16384,program=WaveProgram(period_s=pri,expression=expression),targets=[SceneTarget(position=[3000,0,0],velocity=[20,0,0],gain_db=0)],antenna=ArrayModel(geometry='linear',columns=1,element='isotropic'),noise_db=-100,max_range_m=12000,max_velocity_mps=80,processing=RadarProcessing(method=method,pri_s=pri,pulse_width_s=width,chirp_slope_hz_s=slope,doppler_window='hann'))
    d=simulate(cfg)
    assert d['peak_range_m']==pytest.approx(3000,abs=450)
    assert d['peak_velocity_mps']==pytest.approx(20,abs=5)
    assert d['processor_info']['complete_intervals']>=80


@pytest.mark.parametrize('mode',['mean','two-pulse','three-pulse'])
def test_clutter_filters_suppress_stationary_but_keep_doppler(mode):
    constant=np.ones((24,64),complex);moving=np.exp(.6j*np.arange(24))[:,None]*np.ones((1,64))
    assert np.max(abs(cancel_clutter(constant,mode)))<1e-12
    assert np.sqrt(np.mean(abs(cancel_clutter(moving,mode))**2))>.1


def test_radar_range_gate_and_manual_detector():
    cfg=RadarExperiment(processing=RadarProcessing(min_range_m=4500,detector='manual',threshold_db=50))
    d=simulate(cfg)
    assert min(d['range_m'])>=4500
    assert d['peak_range_m']>=4500
    assert not d['detections']


def test_radar_processing_requires_coherent_intervals():
    with pytest.raises(ValueError,match='four complete'):
        simulate(RadarExperiment(samples=512,processing=RadarProcessing(method='pulse-doppler',pri_s=.001)))


def test_acknowledged_stream_accepts_processing_changes_between_frames():
    """An edit reaches every product computed after it arrives.

    The server keeps up to MAX_IN_FLIGHT frames unacknowledged, so products
    started before the edit arrived still answer the previous revision; each
    product names the client revision it answers.
    """
    from fastapi.testclient import TestClient
    from vectorrf.app import app
    import json
    with TestClient(app).websocket_connect('/ws/live') as ws:
        ws.receive_json()
        ws.send_json({'action':'start','tx':{'duration':.002},'fps':30,'acknowledge_frames':True,'pipeline':{'blocks':[]}})
        seen,updated=[],False
        for _ in range(500):
            m=ws.receive()
            if not m.get('text'):continue
            d=json.loads(m['text'])
            if d['type']=='pipeline':seen.append(d)
            if d['type']=='frame_end':
                if not updated:
                    ws.send_json({'action':'update','client_revision':1,'tx':{'duration':.002},'pipeline':{'blocks':[{'kind':'gain','gain_db':-6}]}})
                    updated=True
                ws.send_json({'action':'ack','seq':d['seq']})
            if sum(x['client_revision']==1 for x in seen)>=2:break
    before=[d for d in seen if d['client_revision']==0]
    after=[d for d in seen if d['client_revision']==1]
    assert before and all(not d['stages'] for d in before)
    assert len(after)==2 and all(d['stages'][0]['kind']=='gain' for d in after)
    assert seen.index(after[0])==len(before)                     # revisions never interleave
    ids=[d['capture_id'] for d in seen]
    assert all(b>a for a,b in zip(ids,ids[1:])),ids
