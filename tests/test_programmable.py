import numpy as np
import pytest
from fastapi.testclient import TestClient
from vectorrf.app import app
from vectorrf.models import TxConfig,RxConfig
from vectorrf.source import source_record
from vectorrf.signal import generate
from vectorrf.rx import demodulate
from vectorrf.programs import WaveProgram,program_samples
from vectorrf.expressions import evaluate
from vectorrf.ofdm import OFDMDesign,resource_grid
from vectorrf.pipeline import Pipeline,process_chain
from vectorrf.radar_sim import RadarExperiment,SceneTarget,ArrayModel,simulate,transmitter


def test_live_captures_change_data_and_are_reproducible():
    tx=TxConfig(waveform='qpsk',snr_db=20)
    a=source_record(tx,20);b=source_record(tx,21)
    assert not np.array_equal(a.samples,b.samples)
    assert a is source_record(tx,20)
    assert a.metadata['source_revision']==b.metadata['source_revision']
    assert b.metadata['simulation_time_s']>a.metadata['simulation_time_s']


def test_expression_absolute_time_and_capture_continuity():
    p=WaveProgram(expression='exp(j*2*pi*12345*time)',period_s=.001)
    a=program_samples(p,1e6,100,0,0);b=program_samples(p,1e6,100,.0001,1)
    expected=np.exp(2j*np.pi*12345*np.arange(200)/1e6)
    np.testing.assert_allclose(np.r_[a,b],expected,atol=1e-6)


@pytest.mark.parametrize('expr',["__import__('os')","x.__class__",'exp(x, out=x)','[1]*1000000000','2**1000000'])
def test_expression_rejects_code_and_unbounded_operations(expr):
    with pytest.raises(ValueError):evaluate(expr,{'x':np.ones(20)},20)


def test_control_point_frequency_integral_and_iq_interpolation():
    p=WaveProgram(mode='points',period_s=.001,points=[{'time_us':0,'frequency_hz':20000},{'time_us':1000,'frequency_hz':20000}])
    y=program_samples(p,1e6,500,.0001)
    np.testing.assert_allclose(np.angle(y[1:]*y[:-1].conj())*1e6/(2*np.pi),20000,atol=.02)
    p=WaveProgram(mode='iq',samples_i=[1,0,-1,0],samples_q=[0,1,0,-1],table_rate=10000,table_interp='hold')
    np.testing.assert_allclose(program_samples(p,10000,8),[1,1j,-1,-1j,1,1j,-1,-1j])


@pytest.mark.parametrize('mod',['bpsk','qpsk','8psk','16qam','64qam'])
def test_custom_ofdm_known_payload_with_pilots_and_variable_cp(mod):
    d=OFDMDesign(enabled=True,modulation=mod,cp_lengths=[16,12,8],window_samples=4)
    tx=TxConfig(waveform='ofdm',data_source='text',message='CUSTOM OFDM ',ofdm_design=d,snr_db=None)
    g=generate(tx)
    r=demodulate(g.samples,RxConfig(modulation='ofdm',ofdm_design=d))
    assert r.text.startswith('CUSTOM OFDM')
    assert r.metrics['evm_rms_percent']<.01


def test_arbitrary_ofdm_grid_values_nulls_and_allocation():
    d=OFDMDesign(enabled=True,active_carriers=[-4,-2,1,5],pilot_carriers=[-2],symbols=2,cells=[{'symbol':0,'carrier':5,'kind':'value','i':.3,'q':-.8},{'symbol':1,'carrier':1,'kind':'null'}])
    grid,roles,_,_=resource_grid(64,16,52,d,lambda n:np.zeros(n,dtype=np.uint8),2)
    assert grid[0,5]==pytest.approx(.3-.8j)
    assert grid[1,1]==0 and roles[1,1]==0
    assert np.count_nonzero(grid[0])==4
    assert grid[0,-2]==1 and grid[1,-2]==-1


def test_pilot_equalizer_corrects_known_multipath():
    d=OFDMDesign(enabled=True,pilot_carriers=[-24,-16,-8,8,16,24],equalizer='pilot-ls')
    tx=TxConfig(waveform='ofdm',ofdm_design=d,data_source='pn',snr_db=None)
    x=generate(tx).samples;y=x+.25*np.r_[0,x[:-1]]
    on=demodulate(y,RxConfig(modulation='ofdm',ofdm_design=d)).metrics['evm_rms_percent']
    off=demodulate(y,RxConfig(modulation='ofdm',ofdm_design=d.model_copy(update={'equalizer':'off'}))).metrics['evm_rms_percent']
    assert on<off*.5


def test_pipeline_mixer_fir_resampling_and_probe():
    fs=100000;t=np.arange(10000)/fs;x=np.exp(2j*np.pi*20000*t)
    cfg=Pipeline.model_validate({'probe':0,'blocks':[{'kind':'mixer','frequency_hz':15000},{'kind':'lowpass','cutoff_hz':10000,'taps':63},{'kind':'resample','down':2}]})
    y,outfs,stages,probe=process_chain(x,fs,cfg)
    assert outfs==50000 and len(y)==5000 and len(stages)==3
    assert np.median(np.angle(y[100:]*y[99:-1].conj())*outfs/(2*np.pi))==pytest.approx(5000,abs=1)
    assert probe['time_s'][-1]==pytest.approx(.09999)


def test_radar_full_waveform_range_doppler_and_angle_are_measured():
    r=RadarExperiment(targets=[SceneTarget(position=[3000,600,200],velocity=[0,0,0])],noise_db=-80)
    d=simulate(r);truth=d['truth'][0]
    assert abs(d['peak_range_m']-truth['range_m'])<2*299792458/(2*r.sample_rate)
    assert d['peak_velocity_mps']==0
    assert d['azimuth_deg']==pytest.approx(truth['az_deg'],abs=1.1)
    assert d['elevation_deg']==pytest.approx(truth['el_deg'],abs=1.6)
    no=simulate(r.model_copy(update={'targets':[]}))
    assert no['rx_rms']<d['rx_rms']/100


def test_radar_frames_refresh_samples_and_measured_power_tracks_range():
    r=RadarExperiment(targets=[SceneTarget(position=[3000,0,0],velocity=[20,0,0])],noise_db=-80)
    a=simulate(r);b=simulate(r.model_copy(update={'capture_id':1,'time_s':.13}))
    assert a['rx']['i']!=b['rx']['i'] and a['psd_db']!=b['psd_db'] and a['rd_db']!=b['rd_db']
    assert a['peak_velocity_mps']==pytest.approx(20,abs=5)
    far=simulate(r.model_copy(update={'targets':[SceneTarget(position=[6000,0,0],velocity=[20,0,0])]}))
    assert far['rx_rms']/a['rx_rms']==pytest.approx(.25,rel=.12)


def test_antenna_steering_changes_received_echo():
    r=RadarExperiment(antenna=ArrayModel(geometry='linear',columns=12,element='isotropic'),targets=[SceneTarget(position=[3000,0,0],velocity=[0,0,0])],noise_db=-100)
    on=simulate(r);off=simulate(r.model_copy(update={'antenna':r.antenna.model_copy(update={'steer_az_deg':30})}))
    assert off['rx_rms']<on['rx_rms']*.05
    assert on['elevation_deg'] is None


def test_complete_legacy_sequence_and_all_custom_sources_feed_returns():
    r=RadarExperiment(source='legacy',targets=[])
    a=transmitter(r,0,4096)
    cfg=r.legacy.model_copy(deep=True);cfg.sequence[0].pulse.sweep_bw_hz=400000
    b=transmitter(r.model_copy(update={'legacy':cfg}),0,4096)
    assert not np.allclose(a,b)
    assert simulate(r)['rd_shape'][0]==r.doppler_bins
    assert simulate(RadarExperiment(source='ofdm',samples=1024))['tx']['i']


def test_studio_api_validation_and_same_capture_observers():
    c=TestClient(app);tx={'waveform':'custom','program':{'expression':'exp(j*2*pi*10000*time)'}}
    assert c.post('/api/measurements',json={'tx':tx,'capture_id':8}).json()['capture_id']==8
    assert c.post('/api/studio/radar',json={'program':{'expression':'not valid'}}).status_code==422
    assert c.post('/api/studio/pipeline',json={'pipeline':{'blocks':[{'kind':'expression','expression':'x * 2'}]},'capture_id':3}).status_code==200


def test_streaming_filter_state_equals_one_continuous_filter():
    from scipy.signal import lfilter
    rng=np.random.default_rng(5);x=rng.normal(size=400)+1j*rng.normal(size=400)
    cfg=Pipeline.model_validate({'blocks':[{'kind':'fir','taps_i':[.2,.5,.3]}]})
    memory={};a=process_chain(x[:173],10000,cfg,memory=memory)[0];b=process_chain(x[173:],10000,cfg,memory=memory)[0]
    np.testing.assert_allclose(np.r_[a,b],lfilter([.2,.5,.3],[1],x),atol=1e-12)


def test_custom_python_extension_produces_waveform_and_carries_state():
    from vectorrf.programs import extension_catalog
    assert {'polynomial_chirp','iir_notch'}<={x['name'] for x in extension_catalog()}
    p=WaveProgram(mode='extension',extension='polynomial_chirp',parameters={'f0':20000,'f1':0,'width_s':.001},period_s=.002)
    x=program_samples(p,100000,1000)
    assert np.max(abs(x))==pytest.approx(1)
    cfg=Pipeline.model_validate({'blocks':[{'kind':'extension','extension':'iir_notch','parameters':{'frequency_hz':20000,'Q':10}}]})
    memory={};y,_,_,_=process_chain(np.exp(2j*np.pi*20000*np.arange(2000)/100000),100000,cfg,memory=memory)
    assert np.mean(abs(y[-500:]))<.01
    assert 'zi' in memory['0']


def test_websocket_transports_processing_for_each_capture():
    client=TestClient(app)
    with client.websocket_connect('/ws/live') as ws:
        ws.receive_json()
        ws.send_json({'action':'start','tx':{'waveform':'custom','duration':.002,'program':{'expression':'exp(j*2*pi*10000*time)'}},'pipeline':{'blocks':[{'kind':'fir','taps_i':[.5,.5]}]},'channel':{'snr_db':None}})
        import json
        products=[]
        for _ in range(20):
            raw=ws.receive()
            if 'text' in raw:
                d=json.loads(raw['text'])
                if d['type']=='pipeline':products.append(d)
                if len(products)==2:break
        assert products[1]['capture_id']==products[0]['capture_id']+1
        assert products[1]['stages'][0]['state']=='carried'


def test_mixed_ofdm_cell_mapping_decodes_with_pilot_scale():
    d=OFDMDesign(enabled=True,cells=[{'symbol':0,'carrier':1,'kind':'data','modulation':'64qam','gain':.5},
        {'symbol':1,'carrier':2,'kind':'data','modulation':'bpsk','gain':1.5}])
    cfg=TxConfig(waveform='ofdm',data_source='pn',ofdm_design=d,snr_db=None)
    g=generate(cfg);r=demodulate(g.samples,RxConfig(modulation='ofdm',ofdm_design=d))
    assert r.metrics['evm_rms_percent']<.01
    assert len(r.bits)>100


@pytest.mark.parametrize('start_s',[0,4.38])
def test_ofdm_cp_acquisition_corrects_fractional_cfo(start_s):
    d=OFDMDesign(enabled=True,modulation='16qam',cp_lengths=[16,12,8],window_samples=4)
    cfg=TxConfig(waveform='ofdm',ofdm_design=d,snr_db=None,message='RECOVER CFO ')
    x=generate(cfg,start_s=start_s).samples*np.exp(2j*np.pi*2200*(start_s+np.arange(int(cfg.sample_rate*cfg.duration))/cfg.sample_rate))
    r=demodulate(x,RxConfig(modulation='ofdm',ofdm_design=d,coarse_cfo=True,ofdm_start_s=start_s))
    assert r.telemetry['ofdm_cfo_hz']==pytest.approx(2200,abs=.1)
    assert r.metrics['evm_rms_percent']<.01
    if start_s==0:assert r.text.startswith('RECOVER CFO')


def test_ofdm_large_frame_advances_to_later_rows_across_captures():
    d=OFDMDesign(enabled=True,symbols=128,cells=[{'symbol':100,'carrier':1,'kind':'value','i':3,'q':-2}],pilot_alternating=False)
    cfg=TxConfig(waveform='ofdm',duration=.001,ofdm_design=d,normalize_output=False,amplitude=1,snr_db=None)
    g=source_record(cfg,8)  # sample 8000 = OFDM symbol 100 at N+CP=80
    fft=np.fft.fft(g.samples[16:80])/8
    assert fft[1]==pytest.approx(3-2j,abs=1e-5)
    rx=demodulate(g.samples,RxConfig(modulation='ofdm',ofdm_design=d,ofdm_start_s=.008))
    assert rx.metrics['evm_rms_percent']<.01


def test_custom_ofdm_capture_boundaries_with_overlap_match_long_signal():
    d=OFDMDesign(enabled=True,cp_lengths=[16,12,8],window_samples=4)
    cfg=TxConfig(waveform='ofdm',duration=.001,ofdm_design=d,normalize_output=False,amplitude=1,snr_db=None)
    a=generate(cfg,start_s=0).samples;b=generate(cfg,start_s=.001).samples
    full=generate(cfg.model_copy(update={'duration':.002})).samples
    np.testing.assert_allclose(np.r_[a,b],full,atol=1e-6)
