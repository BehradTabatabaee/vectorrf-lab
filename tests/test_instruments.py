import numpy as np
import pytest
from fastapi.testclient import TestClient
from vectorrf.app import app
from vectorrf.models import TxConfig,RxConfig,ChannelConfig
from vectorrf.signal import generate,_data_bits
from vectorrf.modem import reference_constellation
from vectorrf.source import source_record
from vectorrf.rx import demodulate
from vectorrf.sync import apply_channel,coarse_carrier,cma_equalizer
from vectorrf.measurements import zero_span

client=TestClient(app)


def test_payload_modes_are_exclusive_and_repeatable():
    pn=TxConfig(data_source='pn',message='ignored',pn_order=7,pn_bits=127)
    a=generate(pn).tx_bits
    b=generate(pn.model_copy(update={'message':'also ignored'})).tx_bits
    np.testing.assert_array_equal(a,b)
    assert len(np.unique(a))==2
    raw=TxConfig(data_source='bits',custom_bits='101100',message='ignored')
    bits=_data_bits(raw,24,np.random.default_rng(7))
    np.testing.assert_array_equal(bits,np.tile([1,0,1,1,0,0],4))
    with pytest.raises(ValueError):TxConfig(data_source='bits',custom_bits='10z')


@pytest.mark.parametrize('mod',['dbpsk','dqpsk','pi4-dqpsk','oqpsk','16apsk','32apsk','gmsk'])
def test_added_modulations_recover_known_payload(mod):
    tx=TxConfig(waveform=mod,message='HELLO VECTORRF ',duration=.06,snr_db=None)
    g=generate(tx)
    r=demodulate(g.samples,RxConfig(modulation=mod,sample_rate=tx.sample_rate,symbol_rate=tx.symbol_rate))
    assert r.text.startswith('HELLO VECTORRF'),(mod,r.text[:30])
    assert np.isfinite(g.samples).all()


def test_pi4_differential_steps_and_oqpsk_offset():
    tx=TxConfig(waveform='pi4-dqpsk',data_source='pn',snr_db=None)
    z=generate(tx).tx_symbols
    angles=np.angle(z[1:]*z[:-1].conj())
    units=np.mod(np.rint(angles/(np.pi/4)),8).astype(int)
    assert set(units).issubset({1,3,5,7})
    q=generate(TxConfig(waveform='qpsk',pulse_shape='rect',sample_rate=210000,symbol_rate=10000,snr_db=None))
    o=generate(TxConfig(waveform='oqpsk',pulse_shape='rect',sample_rate=210000,symbol_rate=10000,snr_db=None))
    # Fractional half-symbol offset at odd 21 samples/symbol.
    expected=np.interp(np.arange(len(q.samples))-10.5,np.arange(len(q.samples)),q.samples.imag,left=0)
    np.testing.assert_allclose(o.samples.imag[100:2000]/np.std(o.samples.real),expected[100:2000]/np.std(q.samples.real),atol=.015)


def test_apsk_rings():
    ref=reference_constellation('32apsk',2.8,5.2)
    radii,counts=np.unique(np.round(np.abs(ref),5),return_counts=True)
    np.testing.assert_array_equal(counts,[4,12,16])
    assert radii[1]/radii[0]==pytest.approx(2.8,rel=1e-4)
    assert np.mean(abs(ref)**2)==pytest.approx(1,rel=1e-6)


def test_gardner_recovers_fractional_timing_and_logs_state():
    tx=TxConfig(waveform='qpsk',message='HELLO VECTORRF ',duration=.06,snr_db=None)
    g=generate(tx)
    y=apply_channel(g.samples,g.sample_rate,ChannelConfig(snr_db=30,timing_offset_samples=.72))
    r=demodulate(y,RxConfig(timing_recovery='gardner',sample_rate=tx.sample_rate,symbol_rate=tx.symbol_rate))
    assert r.text.startswith('HELLO VECTORRF')
    assert r.metrics['evm_rms_percent']<5
    assert len(r.telemetry['timing_error'])>100


def test_coarse_cfo_ground_truth():
    tx=TxConfig(waveform='qpsk',data_source='pn',snr_db=None,duration=.04)
    g=generate(tx)
    y=apply_channel(g.samples,g.sample_rate,ChannelConfig(snr_db=30,freq_offset_hz=7200))
    assert coarse_carrier(y,g.sample_rate,4)==pytest.approx(7200,abs=50)


def test_cma_reduces_constant_modulus_error():
    rng=np.random.default_rng(32)
    x=np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0,4,16000)))
    y=x+.35*np.r_[0,x[:-1]]
    z,error,weights=cma_equalizer(y,gain=.02)
    assert np.mean(error[-2000:]**2)<np.mean(error[:2000]**2)*.75
    assert np.isfinite(weights).all()


def test_zero_span_frequency_before_plot_reduction():
    """Detection runs on every sample; a display bucket shows the extremes of
    its samples, so a steady 170 kHz reads 170 kHz however few points are drawn."""
    fs=1e6;f=170000;t=np.arange(16000)/fs
    z=np.exp(2j*np.pi*f*t)
    d=zero_span(z,fs,duration=.016,full_band=True,points=100)
    assert len(d['frequency_hz'])==len(d['frequency_hz_max'])==len(d['time_s'])==100
    np.testing.assert_allclose(d['frequency_hz'][1:],f,atol=.01)
    np.testing.assert_allclose(d['frequency_hz_max'],f,atol=.01)
    assert d['time_s'][0]==pytest.approx(79.5/fs) and d['time_s'][-1]==pytest.approx(15919.5/fs)


def test_source_is_shared_immutable_and_revisioned():
    cfg=TxConfig(duration=.06)
    a=source_record(cfg);b=source_record(cfg)
    assert a is b and not a.samples.flags.writeable
    c=source_record(cfg.model_copy(update={'phase_offset_deg':20}))
    assert a.metadata['source_revision']!=c.metadata['source_revision']
    body=cfg.model_dump(mode='json')
    rx=client.post('/api/rx/loopback',json={'tx':body}).json()
    blind=client.post('/api/blind/generate',json=body).json()
    measured=client.post('/api/measurements',json={'tx':body}).json()
    assert rx['tx_metadata']['source_revision']==blind['source_revision']==measured['source_revision']==a.metadata['source_revision']


def test_large_records_and_sequences_are_rejected():
    assert client.post('/api/tx/preview',json={'sample_rate':10000000,'duration':5}).status_code==422
    assert client.post('/api/studio/radar',json={'source':'legacy','legacy':{'sequence':[{'repeat':100000,'pri_s':1}]}}).status_code==422


def test_ws_starts_without_edit_and_recovers_invalid_update():
    with client.websocket_connect('/ws/live') as ws:
        assert ws.receive_json()['type']=='hello'
        assert ws.receive_json()['type']=='meta'
        first=ws.receive_json();assert first['type']=='metrics'
        assert ws.receive_bytes()[:4]==b'VRF1'
        ws.send_json({'action':'update','tx':{'sample_rate':-1}})
        seen_error=False
        for _ in range(10):
            msg=ws.receive()
            if 'text' in msg:
                import json
                d=json.loads(msg['text'])
                if d['type']=='error':seen_error=True;break
        assert seen_error
        ws.send_json({'action':'update','tx':{'waveform':'awgn','duration':.06}})
        for _ in range(10):
            msg=ws.receive()
            if 'text' in msg:
                import json
                d=json.loads(msg['text'])
                if d['type']=='meta':break
        assert d['waveform']=='awgn'


def test_generic_frame_crc_and_source_separation():
    import binascii
    cfg=TxConfig(data_source='bits',custom_bits='11001010',frame_enabled=True,preamble_bits='1010',payload_bits=16,frame_crc=True)
    bits=_data_bits(cfg,36,np.random.default_rng(1))
    assert ''.join(map(str,bits[:4]))=='1010'
    payload=np.packbits(bits[4:20]).tobytes()
    expected=binascii.crc_hqx(payload,0xffff).to_bytes(2,'big')
    assert np.packbits(bits[20:36]).tobytes()==expected
    with pytest.raises(ValueError):TxConfig(data_source='text',message='')


def test_segmented_radar_timing_and_frequency():
    from vectorrf.radar_studio import RadarStudioConfig,generate_radar
    cfg=RadarStudioConfig(mode='segments',duration_s=.004,segments=[{'duration_us':40,'start_hz':20000,'stop_hz':20000,'amplitude':1},{'duration_us':30,'start_hz':0,'stop_hz':0,'amplitude':0}])
    d=generate_radar(cfg)
    x=d.samples.astype(complex)
    assert d.metadata['logical_sequence_s']==pytest.approx(70e-6)
    inst=np.angle(x[1:]*np.conj(x[:-1]))*cfg.sample_rate/(2*np.pi)
    assert np.median(inst[2:35])==pytest.approx(20000,rel=.01)
    assert np.min(np.abs(x))==0


def test_full_radar_record_and_short_pulse_supported():
    from vectorrf.radar_studio import RadarStudioConfig,generate_radar
    d=generate_radar(RadarStudioConfig(duration_s=.02,sequence=[{'kind':'pulse','pri_s':100e-6,'pulse':{'waveform':'pulse','width_s':5e-6}}]))
    assert len(d.samples)==round(.02*1e6)
    assert np.count_nonzero(np.abs(d.samples)>0)==pytest.approx(.05*len(d.samples),rel=.02)


def test_receiver_exposes_demodulated_bits_and_hex():
    r=client.post('/api/rx/loopback',json={'tx':{'message':'HELLO '},'channel':{'snr_db':None},'rx':{'use_costas':False}})
    assert r.status_code==200
    d=r.json()
    assert set(d['bits'])=={0,1}
    assert d['hex'].startswith('48 45 4c 4c 4f')


def test_display_buckets_do_not_change_frequency_when_peak_moves():
    from vectorrf.engine import reduce_spectrum
    f=np.arange(4000)
    a=np.zeros(4000);b=np.zeros(4000);a[101]=10;b[102]=10
    fa,pa=reduce_spectrum(f,a,100)
    fb,pb=reduce_spectrum(f,b,100)
    np.testing.assert_array_equal(fa,fb)
    assert pa.max()==pb.max()==10


def test_manual_receiver_rejects_inconsistent_format_parameters():
    for settings in ({'modulation':'ofdm','ofdm_fft':16,'ofdm_cp':16},
                     {'modulation':'32apsk','apsk_ratio2':4,'apsk_ratio3':3},
                     {'symbol_rate':11}):
        response=client.post('/api/rx/loopback',json={'rx':settings})
        assert response.status_code==422
