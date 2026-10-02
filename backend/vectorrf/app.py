from __future__ import annotations

from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .blind import analyze as blind_analyze
from .engine import live_session
from .models import (
    ChannelConfig, RxConfig, SpectrumConfig, TxConfig, WaveformKind,
    MAX_SAMPLE_RATE, MAX_SPECTRUM_SPAN, RECOMMENDED_LIVE_SAMPLE_RATE, RADAR_WAVEFORMS,
)
from .rx import demodulate, receive_frame
from .signal import generate
from .source import source_record
from .measurements import zero_span
from .spectrum import analyze as spectrum_analyze
from .sync import apply_channel
from .radar_studio import RadarStudioConfig, radar_products
from .radar_scene import SceneRequest, scene_products

ROOT=Path(__file__).resolve().parents[2]
PROJECT_WEB=ROOT/"web"
PACKAGE_WEB=Path(__file__).resolve().parent/"web"
WEB=PROJECT_WEB if PROJECT_WEB.exists() else PACKAGE_WEB

app=FastAPI(title="VectorRF Lab",version=__version__,description="Interactive vector signal and DSP laboratory")


@app.exception_handler(ValueError)
async def invalid_design(request: Request, exc: ValueError):
    return JSONResponse(status_code=422,content={'detail':str(exc)[:1000]})


class SpectrumRequest(BaseModel):
    tx: TxConfig = Field(default_factory=TxConfig)
    spectrum: SpectrumConfig = Field(default_factory=SpectrumConfig)


from .pipeline import Pipeline,process_chain,channel_samples

class LoopbackRequest(BaseModel):
    pipeline:Pipeline=Field(default_factory=Pipeline)
    tx: TxConfig = Field(default_factory=TxConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    rx: RxConfig = Field(default_factory=RxConfig)
    capture_id: int = Field(default=0,ge=0,le=2147483647)


def _complex_points(x: np.ndarray,max_points:int=2000):
    if len(x)>max_points:x=x[::max(1,len(x)//max_points)][:max_points]
    return {"i":x.real.astype(float).tolist(),"q":x.imag.astype(float).tolist()}


def _real_points(x: np.ndarray,max_points:int=4000):
    y=np.asarray(x).real
    if len(y)>max_points:y=y[::max(1,len(y)//max_points)][:max_points]
    return y.astype(float).tolist()


@app.get("/api/health")
def health():
    return {"ok":True,"version":__version__,"max_sample_rate":MAX_SAMPLE_RATE,"max_spectrum_span":MAX_SPECTRUM_SPAN,"recommended_live_sample_rate":RECOMMENDED_LIVE_SAMPLE_RATE}


@app.get("/api/catalog")
def catalog():
    return {
        "modes":["tx","spectrum","rx","blind","radar"],
        "waveforms":[
            {"group":"Fundamentals","items":["tone","multitone","awgn","chirp"]},
            {"group":"Digital / PSK & QAM","items":["bpsk","qpsk","dbpsk","dqpsk","pi4-dqpsk","oqpsk","8psk","16qam","64qam","16apsk","32apsk"]},
            {"group":"Digital / FSK & MSK","items":["2fsk","4fsk","msk","gmsk"]},
            {"group":"Digital / Multicarrier","items":["ofdm"]},
            {"group":"Analog","items":["am","fm","pm","ssb","dsb-sc"]},
        ],
        "radar": {
            "continuous":["cw","fmcw","pmcw","stepped-cw"],
            "pulses":["pulse","lfm","nlfm","barker","frank","p1","p2","p3","p4","zadoff-chu","custom-phase","costas","stepped-frequency"],
        },
        "spectrum":{"windows":["hann","blackmanharris","flattop","kaiser","rect"],"detectors":["average","peak","sample"],"max_span_hz":MAX_SPECTRUM_SPAN},
    }


@app.post("/api/tx/preview")
def tx_preview(cfg:TxConfig):
    g=generate(cfg)
    return {"metadata":g.metadata,"samples":_complex_points(g.samples),"bit_count":int(g.tx_bits.size) if g.tx_bits is not None else 0,
            "symbol_count":int(g.tx_symbols.size) if g.tx_symbols is not None else 0}


@app.post("/api/spectrum")
def spectrum(req:SpectrumRequest):
    g=source_record(req.tx)
    s=spectrum_analyze(g.samples,g.sample_rate,req.spectrum,
        reference_power_dbm=g.metadata.get("power_dbm"), reference_rms=g.metadata.get("power_reference_rms"))
    return {"freq_hz":s.freq_hz.astype(float).tolist(),"psd_db":s.psd_db.astype(float).tolist(),"nfft":s.nfft,
            "rbw_actual_hz":s.rbw_actual_hz,"bandwidth_3db_hz":s.bandwidth_3db_hz,"bandwidth_99_hz":s.bandwidth_99_hz,
            "peak_hz":s.peak_hz,"peak_db":s.peak_db,"power_unit":s.unit,"metadata":g.metadata}


@app.post("/api/blind/generate")
def blind_generate(cfg:TxConfig, capture_id: int = Query(default=0,ge=0,le=2147483647)):
    g=source_record(cfg,capture_id)
    x=g.samples
    result=blind_analyze(x,g.sample_rate).model_dump()
    a=np.abs(x)
    hist,edges=np.histogram(a,bins=40)
    result['source_revision']=g.metadata['source_revision']
    result['capture_id']=capture_id
    result['features'].update({'papr_db':float(10*np.log10((np.max(a*a)+1e-20)/(np.mean(a*a)+1e-20))),
        'envelope_cv':float(np.std(a)/(np.mean(a)+1e-20)), 'sample_count':len(x),
        'amplitude_histogram':hist.tolist(),'amplitude_bins':edges[:-1].tolist()})
    return result


@app.post("/api/rx/loopback")
def rx_loopback(req:LoopbackRequest):
    g=source_record(req.tx,req.capture_id)
    y=channel_samples(g,req.channel,req.capture_id)
    rx=req.rx.model_copy(update={"sample_rate":g.sample_rate,"ofdm_start_s":req.capture_id*req.tx.duration})
    y,processed_rate,_,_=process_chain(y,g.sample_rate,req.pipeline,req.capture_id*req.tx.duration,req.capture_id)
    rx=rx.model_copy(update={"sample_rate":processed_rate})
    r=demodulate(y,rx)
    frame_info=receive_frame(y,rx,power_dbm=float(g.metadata.get("power_dbm",0.0)),
        reference_rms=float(g.metadata.get("power_reference_rms",1.0) or 1.0)) if rx.frame_enabled else None
    syms=_complex_points(r.symbols[:2500],max_points=2500) if r.symbols is not None else None
    if syms and r.symbols is not None and len(r.symbols):
        syms["scale"]=float(np.sqrt(np.mean(np.abs(r.symbols)**2)))
    # Prefer the matched-filter symbol stream immediately before carrier recovery
    # for the receiver's "before sync" view.  Fall back to channel samples for
    # demodulators that do not expose that stage.
    telemetry=dict(r.telemetry)
    pre_i=telemetry.pop("pre_costas_i",None)
    pre_q=telemetry.pop("pre_costas_q",None)
    if pre_i is not None and pre_q is not None:
        raw_symbols={"i":pre_i,"q":pre_q}
    else:
        raw_sps=max(1,int(round(g.sample_rate/max(rx.symbol_rate,1.0))))
        raw_symbols=_complex_points(y[::raw_sps],max_points=2500)
    # Reference-aided FIR estimate at the symbol stage, with explicit provenance.
    if g.tx_symbols is not None and r.symbols is not None and len(r.symbols)>64 and req.tx.waveform.value in {"bpsk","qpsk","8psk","16qam","64qam"}:
        n=min(len(r.symbols),len(g.tx_symbols),4096)
        ref=np.asarray(g.tx_symbols[:n],complex)
        measured=np.asarray(r.symbols[:n],complex)
        from scipy.linalg import convolution_matrix
        A=convolution_matrix(ref,5,mode='full')[:n]
        h, *_=np.linalg.lstsq(A[32:],measured[32:],rcond=None)
        telemetry['channel_estimate_taps']=np.abs(h).tolist()
        telemetry['channel_estimate_phase_deg']=np.rad2deg(np.angle(h)).tolist()
        telemetry['channel_estimate_kind']='reference-aided, symbol-aligned 5-tap least squares'
    return JSONResponse({"tx_metadata":{k:v for k,v in g.metadata.items() if not k.startswith('grid_')},"channel":req.channel.model_dump(),"metrics":r.metrics,"text":r.text[:4096] if r.text else r.text,
            "symbols":syms,"raw_symbols":raw_symbols,
            "bits":r.bits[:2048].astype(int).tolist() if r.bits is not None else [],
            "hex":np.packbits(r.bits[:2048]).tobytes().hex(' ') if r.bits is not None else "","recovered":_real_points(r.recovered,1600),"telemetry":telemetry,"frame":frame_info})


class MeasurementRequest(BaseModel):
    tx: TxConfig = Field(default_factory=TxConfig)
    offset_s: float = Field(default=0, ge=0)
    duration_s: float = Field(default=.002, gt=0, le=1)
    quality: bool = False
    capture_id: int = Field(default=0,ge=0,le=2147483647)


@app.post("/api/measurements")
def measurements(req: MeasurementRequest):
    g=source_record(req.tx,req.capture_id)
    out=zero_span(g.samples,g.sample_rate,req.offset_s,req.duration_s)
    out['source_revision']=g.metadata['source_revision']
    out['capture_id']=req.capture_id
    out['quality']=None
    if req.quality:
        # Mixed frames intentionally do not pretend that one demodulator applies
        # to the whole record.  Their detailed quality belongs in RX framing.
        mixed = req.tx.frame_enabled and req.tx.frame_mode == "blocks"
        allowed=set(RxConfig.model_fields['modulation'].annotation.__args__)
        r=None
        x=np.asarray(g.samples,dtype=np.complex64)
        if abs(req.tx.center_hz)>1e-12:
            t=np.arange(len(x))/g.sample_rate
            x=x*np.exp(-2j*np.pi*req.tx.center_hz*t)
        if not mixed and req.tx.waveform.value in allowed:
            rx=RxConfig(modulation=req.tx.waveform.value,sample_rate=req.tx.sample_rate,
                symbol_rate=req.tx.symbol_rate,rolloff=req.tx.rolloff,pulse_shape=req.tx.pulse_shape,
                filter_span_symbols=req.tx.filter_span_symbols,apsk_ratio2=req.tx.apsk_ratio2,
                apsk_ratio3=req.tx.apsk_ratio3,ofdm_fft=req.tx.ofdm_fft,ofdm_cp=req.tx.ofdm_cp,
                ofdm_occupied=req.tx.ofdm_occupied,ofdm_design=req.tx.ofdm_design,ofdm_start_s=req.capture_id*req.tx.duration,fsk_deviation_hz=req.tx.fsk_deviation_hz,
                fm_deviation_hz=max(1,req.tx.fm_deviation_hz),coarse_cfo=True)
            r=demodulate(x,rx)
        if r is not None and "evm_rms_percent" in r.metrics and r.symbols is not None:
            out['quality']={'kind':'evm','metrics':r.metrics,'telemetry':r.telemetry,
                'symbols':_complex_points(r.symbols),
                'method':'Decision-directed EVM with known transmitter modulation, rate, filter and baseband tuning; not a standards conformance result.'}
        else:
            # Fundamentals, analog, FSK/MSK, and mixed frames still get a useful
            # quality page: actual I/Q trajectory, envelope, phase and frequency.
            amp=np.abs(x).astype(np.float64)
            phase=np.unwrap(np.angle(x).astype(np.float64))
            freq=np.r_[0.0,np.angle(x[1:]*np.conj(x[:-1]))*g.sample_rate/(2*np.pi)] if len(x)>1 else np.zeros(len(x))
            take=max(1,len(x)//2000)
            out['quality']={
                'kind':'basic',
                'metrics':{
                    'rms':float(np.sqrt(np.mean(amp*amp))) if len(amp) else 0.0,
                    'peak':float(np.max(amp)) if len(amp) else 0.0,
                    'mean_frequency_hz':float(np.mean(freq[1:])) if len(freq)>1 else 0.0,
                    'frequency_std_hz':float(np.std(freq[1:])) if len(freq)>1 else 0.0,
                },
                'telemetry':{
                    'amplitude':amp[::take].astype(float).tolist(),
                    'phase_deg':np.rad2deg(phase[::take]).astype(float).tolist(),
                    'frequency_hz':freq[::take].astype(float).tolist(),
                },
                'symbols':_complex_points(x[::take],max_points=2000),
                'method':'Direct baseband waveform quality: I/Q trajectory, envelope, phase and instantaneous frequency. EVM is not defined for this signal type.' if not mixed else 'Mixed-modulation frame overview. Use Receiver → Framing for per-block acquisition and decoding.',
            }
    return JSONResponse(out)


@app.post("/api/radar/preview")
def radar_preview(cfg: TxConfig):
    if cfg.waveform not in RADAR_WAVEFORMS:
        raise HTTPException(400, "waveform is not a radar waveform")
    g = generate(cfg)
    x = np.asarray(g.samples, dtype=np.complex128)
    # Keep browser products bounded while retaining full-rate DSP in Python.
    max_points = 4000
    stride = max(1, len(x) // max_points)
    xd = x[::stride][:max_points]
    envelope = np.abs(xd)
    phase = np.unwrap(np.angle(xd))
    if len(xd) > 1:
        instf = np.r_[0.0, np.diff(phase) * (g.sample_rate / stride) / (2.0 * np.pi)]
    else:
        instf = np.zeros_like(phase)

    spec_cfg = SpectrumConfig(rbw_hz=max(100.0, g.sample_rate / 4096.0), averaging=1, max_fft=32768)
    sp = spectrum_analyze(x, g.sample_rate, spec_cfg)

    # Zero-Doppler matched-filter/autocorrelation cut. Limit to one PRI and cap FFT size.
    ref_n = min(len(x), max(64, int(round(cfg.pri_s * g.sample_rate))))
    ref = x[:ref_n]
    active = np.flatnonzero(np.abs(ref) > 0.05 * max(1e-12, float(np.max(np.abs(ref)))))
    if active.size:
        ref = ref[max(0, int(active[0])):min(len(ref), int(active[-1]) + 1)]
    if len(ref) > 8192:
        ref = ref[::max(1, len(ref) // 8192)]
    if len(ref) >= 2:
        ac = np.correlate(ref, ref, mode="full")[len(ref)-1:]
        ac_mag = np.abs(ac)
        ac_mag /= max(1e-12, float(ac_mag[0]))
        ac_db = 20.0 * np.log10(np.maximum(ac_mag, 1e-6))
    else:
        ac_db = np.zeros(1)

    c0 = 299_792_458.0
    nominal_range_resolution = None
    if cfg.waveform in {WaveformKind.LFM, WaveformKind.NLFM, WaveformKind.FMCW, WaveformKind.COSTAS}:
        bw = float(g.metadata.get("sweep_bw_hz", cfg.sweep_bw_hz))
        if bw > 0:
            nominal_range_resolution = c0 / (2.0 * bw)
    elif cfg.waveform == WaveformKind.STEPPED_FREQ:
        bw = max(0.0, (cfg.stepped_steps - 1) * cfg.stepped_spacing_hz)
        if bw > 0:
            nominal_range_resolution = c0 / (2.0 * bw)
    elif cfg.waveform == WaveformKind.PULSE:
        nominal_range_resolution = c0 * cfg.pulse_width_s / 2.0
    elif cfg.waveform in {WaveformKind.BARKER, WaveformKind.FRANK, WaveformKind.P1, WaveformKind.P2, WaveformKind.P3, WaveformKind.P4, WaveformKind.ZADOFF_CHU, WaveformKind.CUSTOM_PHASE}:
        code_len = max(1, int(g.metadata.get("code_length", 1)))
        nominal_range_resolution = c0 * (cfg.pulse_width_s / code_len) / 2.0
    elif cfg.waveform == WaveformKind.PMCW:
        nominal_range_resolution = c0 / (2.0 * cfg.pmcw_chip_rate)
    return {
        "metadata": g.metadata,
        "time_s": (np.arange(len(xd)) * stride / g.sample_rate).astype(float).tolist(),
        "envelope": envelope.astype(float).tolist(),
        "inst_freq_hz": instf.astype(float).tolist(),
        "phase_deg": np.rad2deg(np.angle(xd)).astype(float).tolist(),
        "i": xd.real.astype(float).tolist(),
        "q": xd.imag.astype(float).tolist(),
        "freq_hz": sp.freq_hz.astype(float).tolist(),
        "psd_db": sp.psd_db.astype(float).tolist(),
        "autocorr_db": ac_db.astype(float).tolist(),
        "metrics": {
            "prf_hz": 1.0 / cfg.pri_s,
            "duty_cycle": min(1.0, cfg.pulse_width_s / cfg.pri_s),
            "time_bandwidth_product": cfg.pulse_width_s * cfg.sweep_bw_hz,
            "nominal_range_resolution_m": nominal_range_resolution,
        },
    }




@app.post("/api/radar/studio")
def radar_studio(cfg: RadarStudioConfig):
    return radar_products(cfg)


@app.post("/api/radar/scene")
def radar_scene(req: SceneRequest):
    return scene_products(req)


@app.websocket("/ws/live")
async def ws_live(ws:WebSocket):
    await live_session(ws)


if WEB.exists():
    app.mount("/assets",StaticFiles(directory=WEB),name="assets")

    @app.get("/")
    def index(): return FileResponse(WEB/"index.html")

    @app.get("/help")
    def help_page(): return FileResponse(WEB/"help.html")

# Programmable studio APIs. New frames have explicit simulation time and identity.
from .radar_sim import RadarExperiment, simulate as radar_simulate
from .pipeline import Pipeline, process_chain, preview as pipeline_preview
from .programs import extension_catalog

class PipelineRequest(BaseModel):
    tx: TxConfig = Field(default_factory=TxConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    pipeline: Pipeline = Field(default_factory=Pipeline)
    capture_id: int = Field(default=0,ge=0,le=2147483647)

@app.get('/api/studio/extensions')
def studio_extensions():
    return {'extensions':extension_catalog()}

@app.post('/api/studio/radar')
def studio_radar(req: RadarExperiment):
    return JSONResponse(radar_simulate(req))

@app.post('/api/studio/pipeline')
def studio_pipeline(req: PipelineRequest):
    g=source_record(req.tx,req.capture_id)
    y=channel_samples(g,req.channel,req.capture_id)
    z,fs,stages,probe=process_chain(y,g.sample_rate,req.pipeline,req.capture_id*req.tx.duration,req.capture_id)
    spec=spectrum_analyze(z,fs,SpectrumConfig(rbw_hz=fs/1024,averaging=2,max_fft=2048))
    return {'capture_id':req.capture_id,'input':pipeline_preview(y,g.sample_rate),'output':probe,
            'stages':stages,'output_rate':fs,'freq_hz':spec.freq_hz.tolist(),'psd_db':spec.psd_db.tolist(),
            'rms':float(np.sqrt(np.mean(abs(z)**2))),'papr_db':float(10*np.log10((np.max(abs(z)**2)+1e-20)/(np.mean(abs(z)**2)+1e-20)))}

@app.post('/api/studio/ofdm')
def studio_ofdm(req: TxConfig):
    if req.waveform.value!='ofdm':raise ValueError('Select OFDM')
    cfg=req.model_copy(update={'duration':min(req.duration,req.ofdm_design.symbols*(req.ofdm_fft+req.ofdm_cp)/req.sample_rate+.001)})
    return generate(cfg).metadata
