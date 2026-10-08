from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import json
from fastapi import Body, FastAPI, WebSocket, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from . import __version__
from .blind import analyze as blind_analyze
from .engine import MeasurementSpec, calibrated_analyze, live_session, measurement_product
from .models import (
    ChannelConfig, RxConfig, SpectrumConfig, TxConfig,
    MAX_SAMPLE_RATE, MAX_SPECTRUM_SPAN, MAX_CAPTURE_SAMPLES, MAX_EXPORT_SAMPLES,
    RECOMMENDED_LIVE_SAMPLE_RATE, MAX_COMPONENTS,
)
from .agility import MAX_HOPS
from .rx import demodulate, receive_frame
from .signal import generate
from .source import acquire, source_record
from .spectrum import analyze as spectrum_analyze
from .recordings import FileSource
from . import iqfile, recordings
from .bands import component_problems, design_plan
from .errors import describe, errors_of, message as describe_message
from .dsp import display_complex, display_real

ROOT=Path(__file__).resolve().parents[2]
PROJECT_WEB=ROOT/"web"
PACKAGE_WEB=Path(__file__).resolve().parent/"web"
WEB=PROJECT_WEB if PROJECT_WEB.exists() else PACKAGE_WEB

app=FastAPI(title="VectorRF Lab",version=__version__,description="Interactive vector signal and DSP laboratory")


def _invalid(exc: Exception, data=None) -> JSONResponse:
    """422 with one readable sentence per problem (``detail``) and the raw locations (``errors``)."""
    errors=[{"loc":[str(p) for p in e.get("loc",())],
             "msg":describe_message(str(e.get("msg","")),next((k for k in reversed(e.get("loc",())) if isinstance(k,str)),None))}
            for e in errors_of(exc)]
    return JSONResponse(status_code=422,content={'detail':describe(exc,data)[:2000],'errors':errors[:50]})


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    return _invalid(exc, exc.body)


@app.exception_handler(ValueError)
async def invalid_design(request: Request, exc: ValueError):
    return _invalid(exc)


@app.middleware("http")
async def revalidate_ui_assets(request: Request, call_next):
    """Make browsers revalidate the UI shell and scripts on every load.

    Without an explicit policy, browsers apply heuristic caching and can keep
    serving a previous release's scripts after an upgrade, mixing versions.
    ``no-cache`` still allows cheap ETag/Last-Modified revalidation.
    """
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/help") or path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class SpectrumRequest(BaseModel):
    tx: TxConfig = Field(default_factory=TxConfig)
    spectrum: SpectrumConfig = Field(default_factory=SpectrumConfig)
    source: FileSource | None = None
    capture_id: int = Field(default=0,ge=0,le=2147483647)


from .pipeline import Pipeline,process_chain,channel_samples,dehop

class LoopbackRequest(BaseModel):
    pipeline:Pipeline=Field(default_factory=Pipeline)
    tx: TxConfig = Field(default_factory=TxConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    rx: RxConfig = Field(default_factory=RxConfig)
    source: FileSource | None = None
    capture_id: int = Field(default=0,ge=0,le=2147483647)


def _capture(tx: TxConfig, source: FileSource | None, capture_id: int):
    """The capture every observer of (source, capture_id) shares."""
    return acquire(tx, source, capture_id)


def _public_metadata(g):
    return {k:v for k,v in g.metadata.items() if not k.startswith('grid_')}


@app.get("/api/health")
def health():
    return {"ok":True,"version":__version__,"max_sample_rate":MAX_SAMPLE_RATE,"max_spectrum_span":MAX_SPECTRUM_SPAN,
            "max_capture_samples":MAX_CAPTURE_SAMPLES,"max_export_samples":MAX_EXPORT_SAMPLES,
            "recommended_live_sample_rate":RECOMMENDED_LIVE_SAMPLE_RATE}


@app.get("/api/catalog")
def catalog():
    return {
        "modes":["tx","files","spectrum","rx","blind","radar"],
        "sources":["transmitter","file"],
        "waveforms":[
            {"group":"Fundamentals","items":["tone","multitone","awgn","chirp","custom"]},
            {"group":"Digital / PSK & QAM","items":["bpsk","qpsk","dbpsk","dqpsk","pi4-dqpsk","oqpsk","8psk","16qam","64qam","16apsk","32apsk"]},
            {"group":"Digital / FSK & MSK","items":["2fsk","4fsk","msk","gmsk"]},
            {"group":"Digital / Multicarrier","items":["ofdm"]},
            {"group":"Analog","items":["am","fm","pm","ssb","dsb-sc"]},
            {"group":"Composite","items":["composite"]},
        ],
        "agility":{"modes":["hop","sweep"],"plans":["linear","list","random"],"phase":["continuous","coherent"],
                   "max_hops":MAX_HOPS,"max_components":MAX_COMPONENTS},
        "radar": {
            "continuous":["cw","fmcw","pmcw","stepped-cw"],
            "pulses":["pulse","lfm","nlfm","barker","frank","p1","p2","p3","p4","zadoff-chu","custom-phase","costas","stepped-frequency"],
        },
        "spectrum":{"windows":["hann","blackmanharris","flattop","kaiser","rect"],"detectors":["average","peak","sample"],"max_span_hz":MAX_SPECTRUM_SPAN},
        "files":{
            "containers":["raw","wav","sigmf"],
            "sample_formats":list(iqfile.RAW_FORMATS),
            "extensions":sorted(iqfile.LIBRARY_EXTENSIONS),
            "export_formats":["cf32_le","ci16_le","ci8","cu8","cf64_le"],
        },
    }


@app.get("/api/defaults")
def defaults():
    """Default values of every editable model, so the browser never duplicates them."""
    from .agility import FrequencyAgility
    from .models import ChannelBand, FrameBlock
    from .pipeline import DSPBlock
    from .radar_processing import RadarProcessing
    from .radar_sim import ArrayModel, SceneTarget
    from .radar_studio import RadarPulseConfig, RadarSegment, RadarSequenceItem, RadarStudioConfig
    models = {
        "tx": TxConfig, "rx": RxConfig, "channel": ChannelConfig, "spectrum": SpectrumConfig,
        "channel_band": ChannelBand, "agility": FrequencyAgility, "frame_block": FrameBlock,
        "pipeline": Pipeline, "dsp_block": DSPBlock, "radar_experiment": RadarExperiment,
        "radar_design": RadarStudioConfig, "radar_pulse": RadarPulseConfig, "radar_item": RadarSequenceItem,
        "radar_segment": RadarSegment, "radar_target": SceneTarget, "radar_array": ArrayModel,
        "radar_processing": RadarProcessing,
    }
    return {name: model().model_dump(mode="json") for name, model in models.items()}


@app.post("/api/tx/plan")
def tx_plan(body: dict = Body(...)):
    """Validate a transmitter design and describe its bands, power and hop/sweep schedules.

    A rejected composite also reports which components are at fault
    (``components``: one message or ``None`` per component), so the editor can
    mark them.
    """
    try:
        cfg = TxConfig.model_validate(body)
    except ValidationError as exc:
        response = _invalid(exc, body)
        content = json.loads(response.body)
        content["components"] = component_problems(body)
        return JSONResponse(status_code=422, content=content)
    return JSONResponse(design_plan(cfg))


@app.post("/api/tx/preview")
def tx_preview(cfg:TxConfig):
    g=generate(cfg)
    return {"metadata":g.metadata,"samples":display_complex(g.samples),"bit_count":int(g.tx_bits.size) if g.tx_bits is not None else 0,
            "symbol_count":int(g.tx_symbols.size) if g.tx_symbols is not None else 0}


@app.post("/api/spectrum")
def spectrum(req:SpectrumRequest):
    g=_capture(req.tx,req.source,req.capture_id)
    s=calibrated_analyze(g.samples,g,req.spectrum)
    return {"freq_hz":s.freq_hz.astype(float).tolist(),"psd_db":s.psd_db.astype(float).tolist(),"nfft":s.nfft,
            "rbw_actual_hz":s.rbw_actual_hz,"rbw_limited_by_capture":s.rbw_limited_by_capture,
            "bandwidth_3db_hz":s.bandwidth_3db_hz,"bandwidth_99_hz":s.bandwidth_99_hz,
            "peak_hz":s.peak_hz,"peak_db":s.peak_db,"power_unit":s.unit,"center_hz":s.center_hz,"span_hz":s.span_hz,
            "decimation":s.decimation,"analysis_rate":s.analysis_rate,"channels":s.channels,"metadata":_public_metadata(g)}


def _blind_product(g, capture_id, band=None):
    x=g.samples
    result=blind_analyze(x,g.sample_rate,band).model_dump()
    a=np.abs(x)
    hist,edges=np.histogram(a,bins=40)
    result['source_revision']=g.metadata['source_revision']
    result['capture_id']=capture_id
    result['source_kind']=g.metadata.get('source_kind','transmitter')
    result['features'].update({'papr_db':float(10*np.log10((np.max(a*a)+1e-20)/(np.mean(a*a)+1e-20))),
        'envelope_cv':float(np.std(a)/(np.mean(a)+1e-20)), 'sample_count':len(x),
        'amplitude_histogram':hist.tolist(),'amplitude_bins':edges[:-1].tolist()})
    return result


@app.post("/api/blind/generate")
def blind_generate(cfg:TxConfig, capture_id: int = Query(default=0,ge=0,le=2147483647)):
    return _blind_product(source_record(cfg,capture_id),capture_id)


@app.get("/api/blind/features")
def blind_features():
    """Definitions of the features the classifier's rules test."""
    from .blind import FEATURES
    return {"features": {key: {"label": label, "description": description} for key, (label, description) in FEATURES.items()}}


class BlindRequest(BaseModel):
    tx: TxConfig = Field(default_factory=TxConfig)
    source: FileSource | None = None
    capture_id: int = Field(default=0,ge=0,le=2147483647)
    band_center_hz: float | None = None
    band_width_hz: float | None = Field(default=None, gt=0)


@app.post("/api/blind/analyze")
def blind_capture(req:BlindRequest):
    band=(req.band_center_hz or 0.0,req.band_width_hz) if req.band_width_hz else None
    return _blind_product(_capture(req.tx,req.source,req.capture_id),req.capture_id,band)


@app.post("/api/rx/loopback")
def rx_loopback(req:LoopbackRequest):
    g=_capture(req.tx,req.source,req.capture_id)
    y=dehop(channel_samples(g,req.channel,req.capture_id),req.rx.dehop,g.sample_rate,g.metadata.get("start_sample"))
    start=float(g.metadata.get("simulation_time_s",0.0))
    y,processed_rate,_,_=process_chain(y,g.sample_rate,req.pipeline,start,req.capture_id,start_sample=g.metadata.get("start_sample"))
    rx=req.rx.model_copy(update={"sample_rate":processed_rate,"ofdm_start_s":start})
    aligned_start = g.metadata.get("start_sample") if abs(processed_rate - g.sample_rate) < 1e-9 * g.sample_rate else None
    r=demodulate(y,rx,int(aligned_start) if aligned_start is not None else None)
    frame_info=receive_frame(y,rx,power_dbm=float(g.metadata.get("power_dbm",0.0)),
        reference_rms=float(g.metadata.get("power_reference_rms",1.0) or 1.0)) if rx.frame_enabled else None
    syms=display_complex(r.symbols[:2500],maximum=2500) if r.symbols is not None else None
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
        raw_sps=max(1,int(round(processed_rate/max(rx.symbol_rate,1.0))))
        raw_symbols=display_complex(y[::raw_sps],maximum=2500)
    # Reference-aided FIR estimate at the symbol stage, with explicit provenance.
    if req.source is None and g.tx_symbols is not None and r.symbols is not None and len(r.symbols)>64 and req.tx.waveform.value in {"bpsk","qpsk","8psk","16qam","64qam"}:
        from .rx import channel_symbols, reference_channel_estimate
        telemetry.update(reference_channel_estimate(g.tx_symbols,channel_symbols(r)) or {})
    return JSONResponse({"tx_metadata":_public_metadata(g),"channel":req.channel.model_dump(),"metrics":r.metrics,"text":r.text[:4096] if r.text else r.text,
            "symbols":syms,"raw_symbols":raw_symbols,"capture_id":req.capture_id,
            "bits":r.bits[:2048].astype(int).tolist() if r.bits is not None else [],
            "hex":np.packbits(r.bits[:2048]).tobytes().hex(' ') if r.bits is not None else "","recovered":display_real(r.recovered,1600),"telemetry":telemetry,"frame":frame_info})


class MeasurementRequest(BaseModel):
    tx: TxConfig = Field(default_factory=TxConfig)
    source: FileSource | None = None
    offset_s: float = Field(default=0, ge=0)
    duration_s: float = Field(default=.002, gt=0, le=1)
    # Zero-span tuning: mix ``center_hz`` to DC and limit to ``bandwidth_hz``
    # (the zero-span resolution bandwidth) before AM/PM/FM detection.
    center_hz: float = 0.0
    bandwidth_hz: float | None = Field(default=None, gt=0)
    quality: bool = False
    capture_id: int = Field(default=0,ge=0,le=2147483647)


@app.post("/api/measurements")
def measurements(req: MeasurementRequest):
    g=_capture(req.tx,req.source,req.capture_id)
    out=measurement_product(g,req.tx,req.source is not None,req.capture_id,MeasurementSpec(**req.model_dump(include=set(MeasurementSpec.model_fields))))
    return JSONResponse(out)


@app.websocket("/ws/live")
async def ws_live(ws:WebSocket):
    # Browsers do not apply CORS to WebSockets: a page from another origin could
    # otherwise drive the engine of a lab running on this machine.
    origin = ws.headers.get("origin")
    if origin and urlsplit(origin).netloc != ws.headers.get("host"):
        await ws.close(code=1008)
        return
    await live_session(ws)


# ---------------------------------------------------------------------------
# Recording library, import and export


@app.get("/api/files")
def files_list():
    return {"directory":str(recordings.library_dir()),"files":recordings.list_library(),
            "max_upload_bytes":recordings.MAX_UPLOAD_BYTES}


@app.post("/api/files/inspect")
def files_inspect(source: FileSource):
    return recordings.describe(source)


class OverviewRequest(BaseModel):
    source: FileSource
    rows: int = Field(default=192, ge=8, le=1024)
    nfft: int = Field(default=1024, ge=64, le=16384)


@app.post("/api/files/overview")
def files_overview(req: OverviewRequest):
    return JSONResponse(recordings.overview(req.source, req.rows, req.nfft))


@app.put("/api/files/upload")
async def files_upload(request: Request, name: str = Query(min_length=1, max_length=200), overwrite: bool = False):
    """Stream a request body straight to the capture library (no multipart, no RAM copy)."""
    saved = await recordings.save_upload(name, request.stream(), overwrite)
    return {**saved, "file": next((f for f in recordings.list_library() if f["name"] == saved["name"]), None)}


@app.delete("/api/files/{name}")
def files_delete(name: str):
    return {"removed": recordings.delete_recording(name)}


class ExportRequest(BaseModel):
    """Write a capture to disk formats used by SDR tools.

    ``tx`` is validated here with the export sample limit (a longer contiguous
    record than one live capture).  With ``source`` set, the window
    ``[start_sample, start_sample + samples)`` of a recording is converted.
    """
    tx: dict = Field(default_factory=dict)
    source: FileSource | None = None
    capture_id: int = Field(default=0, ge=0, le=2147483647)
    duration_s: float | None = Field(default=None, gt=0)
    samples: int | None = Field(default=None, ge=32)
    format: Literal["cf32_le", "cf64_le", "ci16_le", "ci8", "cu8"] = "cf32_le"
    container: Literal["raw", "wav", "sigmf"] = "sigmf"
    scaling: Literal["peak", "unity"] = "peak"
    headroom_db: float = Field(default=1.0, ge=0.0, le=60.0)
    destination: Literal["download", "library"] = "download"
    name: str | None = Field(default=None, max_length=160)


_RAW_EXTENSIONS = {"cf32_le": ".cf32", "cf64_le": ".cf64", "ci16_le": ".cs16", "ci8": ".cs8", "cu8": ".cu8"}


@app.post("/api/files/export")
def files_export(req: ExportRequest):
    if req.source is not None:
        reader = recordings.open_reader(req.source)
        count = int(req.samples or req.source.capture_samples)
        if count > MAX_EXPORT_SAMPLES:
            raise ValueError(f"Export is limited to {MAX_EXPORT_SAMPLES:,} samples (VECTORRF_MAX_EXPORT_SAMPLES)")
        x = reader.read(req.source.start_sample, count)
        if not len(x):
            raise ValueError("The requested window lies beyond the end of the recording")
        fs = float(reader.info.sample_rate or req.source.sample_rate or 0)
        if not fs:
            raise ValueError("Set the recording sample rate before exporting")
        center = reader.info.center_frequency_hz
        stem = Path(reader.info.path.name).stem.replace(".sigmf", "") + f"_{req.source.start_sample}"
        extra = {"vectorrf:source_recording": reader.info.path.name, "vectorrf:source_start_sample": int(req.source.start_sample)}
        description = f"Excerpt of {reader.info.path.name}"
        power = None
    else:
        payload = dict(req.tx)
        if req.duration_s is not None:
            payload["duration"] = req.duration_s
        elif req.samples is not None:
            payload["duration"] = req.samples / float(payload.get("sample_rate", TxConfig().sample_rate))
        try:
            tx = TxConfig.model_validate(payload, context={"max_samples": MAX_EXPORT_SAMPLES})
        except ValidationError as exc:
            raise ValueError(str(exc.errors()[0].get("msg", exc))) from exc
        effective = tx.model_copy(update={"seed": (tx.seed + req.capture_id * 104729) % (2**32)})
        n = max(32, int(round(tx.duration * tx.sample_rate)))
        g = generate(effective, frame=req.capture_id, start_sample=req.capture_id * n)
        x, fs = g.samples, tx.sample_rate
        stem = f"vectorrf_{tx.waveform.value}_{int(tx.sample_rate)}sps"
        power = (float(g.metadata["power_dbm"]), float(g.metadata["power_reference_rms"]))
        center = tx.rf_center_hz or None
        extra = {"vectorrf:waveform": tx.waveform.value, "vectorrf:capture_id": req.capture_id,
                 "vectorrf:start_sample": int(req.capture_id * n), "vectorrf:power_dbm": power[0],
                 # The complete design, so an exported file documents exactly how it was made.
                 "vectorrf:tx_design": tx.model_dump(mode="json")}
        description = f"VectorRF Lab {tx.waveform.value} capture"
    fmt = iqfile.SampleFormat.parse(req.format)
    scale = iqfile.peak_scale(x, req.headroom_db) if req.scaling == "peak" else 1.0
    if req.scaling == "unity" and fmt.kind != "f" and np.max(np.abs(np.r_[x.real, x.imag]), initial=0) > 1:
        raise ValueError("Unity scaling would clip this capture in an integer format; choose peak scaling or a float format")
    if power is not None:
        # A reader that applies the stored offset recovers absolute dBm.
        extra["vectorrf:full_scale_dbm"] = power[0] - 20 * np.log10(max(power[1] * scale, 1e-30))
    extra["vectorrf:export_scale"] = scale
    stem = Path(req.name).stem if req.name else stem
    container = req.container
    if container == "wav":
        data, clipped = iqfile.encode_recording(x, fmt, fs, "wav", scale, center)
        filename = stem + ".wav"
    else:
        data, clipped = iqfile.encode_recording(x, fmt, fs, "raw", scale)
        if container == "sigmf":
            meta = iqfile.sigmf_meta(fmt, fs, center, description, extra)
            if req.destination == "library":
                meta_name = recordings.unique_name(stem + ".sigmf-meta")
                base = meta_name[: -len(".sigmf-meta")]
                recordings.resolve(base + ".sigmf-data", must_exist=False).write_bytes(data)
                import json as _json
                recordings.resolve(meta_name, must_exist=False).write_text(_json.dumps(meta, indent=2), encoding="utf-8")
                return {"saved": meta_name, "samples": len(x), "bytes": len(data), "clipped": clipped, "scale": scale}
            data = iqfile.sigmf_archive(stem, meta, data)
            filename = stem + ".sigmf"
        else:
            filename = stem + _RAW_EXTENSIONS[fmt.name]
    if req.destination == "library":
        saved = recordings.write_library_file(filename, data)
        return {"saved": saved, "samples": len(x), "bytes": len(data), "clipped": clipped, "scale": scale}
    headers = {"Content-Disposition": f'attachment; filename="{filename}"', "X-VectorRF-Clipped": str(clipped),
               "X-VectorRF-Scale": f"{scale:.9g}", "X-VectorRF-Samples": str(len(x))}
    return Response(content=data, media_type="application/octet-stream", headers=headers)


if WEB.exists():
    # The UI is ES modules, which browsers only run with a JavaScript MIME
    # type; some Windows registries map .js to text/plain.
    mimetypes.add_type("text/javascript", ".js")
    app.mount("/assets",StaticFiles(directory=WEB),name="assets")

    @app.get("/")
    def index(): return FileResponse(WEB/"index.html")

    @app.get("/help")
    def help_page(): return FileResponse(WEB/"help.html")

# Programmable studio APIs. New frames have explicit simulation time and identity.
from .radar_sim import RadarExperiment, simulate as radar_simulate
from .pipeline import preview as pipeline_preview
from .programs import extension_catalog

class PipelineRequest(BaseModel):
    tx: TxConfig = Field(default_factory=TxConfig)
    source: FileSource | None = None
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
    g=_capture(req.tx,req.source,req.capture_id)
    y=channel_samples(g,req.channel,req.capture_id)
    z,fs,stages,probe=process_chain(y,g.sample_rate,req.pipeline,float(g.metadata.get('simulation_time_s',0.0)),req.capture_id,start_sample=g.metadata.get('start_sample'))
    spec=spectrum_analyze(z,fs,SpectrumConfig(rbw_hz=fs/1024,averaging=2,max_fft=2048))
    return {'capture_id':req.capture_id,'input':pipeline_preview(y,g.sample_rate),'output':probe,
            'stages':stages,'output_rate':fs,'freq_hz':spec.freq_hz.tolist(),'psd_db':spec.psd_db.tolist(),
            'rms':float(np.sqrt(np.mean(abs(z)**2))),'papr_db':float(10*np.log10((np.max(abs(z)**2)+1e-20)/(np.mean(abs(z)**2)+1e-20)))}

@app.post('/api/studio/ofdm')
def studio_ofdm(req: TxConfig):
    if req.waveform.value!='ofdm':raise ValueError('Select OFDM')
    cfg=req.model_copy(update={'duration':min(req.duration,req.ofdm_design.symbols*(req.ofdm_fft+req.ofdm_cp)/req.sample_rate+.001)})
    return generate(cfg).metadata
