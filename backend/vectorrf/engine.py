from __future__ import annotations
import asyncio
import json
import time
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from .models import TxConfig, SpectrumConfig, ChannelConfig, RxConfig, RECOMMENDED_LIVE_SAMPLE_RATE
from .source import source_record
from .pipeline import Pipeline,pipeline_product,channel_samples,process_chain,preview
from .spectrum import analyze, choose_nfft
from .stream import FrameType, pack_f32
from .rx import demodulate, receive_frame


def reduce_spectrum(freq, db, maximum=2048):
    """Fixed frequency buckets preserve trace-hold alignment across frames."""
    if len(db) <= maximum:
        return freq, db
    edges = np.linspace(0, len(db), maximum+1, dtype=int)
    centers=(freq[edges[:-1]]+freq[edges[1:]-1])/2
    return centers.astype(np.float32), np.maximum.reduceat(db,edges[:-1])



def frame_product(g, spec, cursor):
    nfft, _ = choose_nfft(g.sample_rate, spec.rbw_hz, spec.window, spec.max_fft)
    count = min(len(g.samples), max(4096, nfft * max(1,spec.averaging)))
    # Never wrap a DSP window across the record boundary.  The former circular
    # splice created a discontinuity that appeared as occasional broadband
    # "pulses" on PSK/QAM spectra.
    if count >= len(g.samples):
        chunk = g.samples
        next_cursor = 0
    else:
        max_start = len(g.samples) - count
        cursor = int(np.clip(cursor, 0, max_start))
        chunk = g.samples[cursor:cursor+count]
        next_cursor = (cursor + max(1, count//3)) % (max_start + 1)
    result = analyze(
        chunk, g.sample_rate, spec,
        reference_power_dbm=g.metadata.get('power_dbm'),
        reference_rms=g.metadata.get('power_reference_rms'),
    )
    f, p = reduce_spectrum(result.freq_hz, result.psd_db)
    return result, f, p, next_cursor


def _complex_points(x, maximum=2000):
    if x is None:
        return None
    x=np.asarray(x)
    if len(x)>maximum:
        ix=np.unique(np.linspace(0,len(x)-1,maximum,dtype=int));x=x[ix]
    return {'i':np.asarray(x).real.astype(float).tolist(),'q':np.asarray(x).imag.astype(float).tolist()}

def _trim_telemetry(value, maximum=2200):
    if isinstance(value, np.ndarray):
        value=value.tolist()
    if isinstance(value, list) and len(value)>maximum:
        stride=max(1,len(value)//maximum)
        return value[::stride][:maximum]
    if isinstance(value, dict):
        return {k:_trim_telemetry(v,maximum) for k,v in value.items()}
    return value

def receiver_product(g, channel, pipeline, rx, frame, memory=None):
    """One real source -> channel -> user pipeline -> demod product for live UI."""
    y=channel_samples(g,channel,frame)
    start=float(g.metadata.get('simulation_time_s',frame*g.metadata.get('duration',0)))
    z,fs,stages,probe=process_chain(y,g.sample_rate,pipeline,start,frame,memory)
    cfg=rx.model_copy(update={'sample_rate':fs,'ofdm_start_s':start})
    r=demodulate(z,cfg)
    frame_info = receive_frame(
        z, cfg, power_dbm=float(g.metadata.get('power_dbm', 0.0)),
        reference_rms=float(g.metadata.get('power_reference_rms', 1.0) or 1.0),
    ) if cfg.frame_enabled else None
    telemetry={k:_trim_telemetry(v) for k,v in dict(r.telemetry).items()}
    pre_i=telemetry.pop('pre_costas_i',None);pre_q=telemetry.pop('pre_costas_q',None)
    if pre_i is not None and pre_q is not None:
        raw_symbols={'i':pre_i,'q':pre_q}
    else:
        raw_sps=max(1,int(round(fs/max(cfg.symbol_rate,1.0)))) if cfg.modulation not in {'tone','multitone','awgn','chirp','custom'} else max(1,len(z)//1600)
        raw_symbols=_complex_points(z[::raw_sps],2500)
    syms=_complex_points(r.symbols,2500)
    if syms is not None and r.symbols is not None and len(r.symbols):
        syms['scale']=float(np.sqrt(np.mean(np.abs(r.symbols)**2)))
    # Keep the reference-aided channel view available without creating a second capture.
    if g.tx_symbols is not None and r.symbols is not None and len(r.symbols)>64 and g.metadata.get('waveform') in {'bpsk','qpsk','8psk','16qam','64qam'}:
        n=min(len(r.symbols),len(g.tx_symbols),2048)
        ref=np.asarray(g.tx_symbols[:n],complex);measured=np.asarray(r.symbols[:n],complex)
        try:
            from scipy.linalg import convolution_matrix
            A=convolution_matrix(ref,5,mode='full')[:n]
            h,*_=np.linalg.lstsq(A[32:],measured[32:],rcond=None)
            telemetry['channel_estimate_taps']=np.abs(h).astype(float).tolist()
            telemetry['channel_estimate_phase_deg']=np.rad2deg(np.angle(h)).astype(float).tolist()
            telemetry['channel_estimate_kind']='reference-aided, symbol-aligned 5-tap least squares'
        except (ValueError,np.linalg.LinAlgError):
            pass
    scfg=SpectrumConfig(rbw_hz=max(1.0,fs/1024),averaging=1,max_fft=2048)
    sp=analyze(z,fs,scfg);freq,db=reduce_spectrum(sp.freq_hz,sp.psd_db,640)
    bits=r.bits[:2048].astype(int).tolist() if r.bits is not None else []
    packed=np.packbits(r.bits[:2048]).tobytes().hex(' ') if r.bits is not None else ''
    recovered=np.asarray(r.recovered)
    return {
        'capture_id':frame,
        'tx_metadata':{k:v for k,v in g.metadata.items() if not k.startswith('grid_')},
        'channel':channel.model_dump(),
        'metrics':r.metrics,'text':r.text[:4096] if r.text else r.text,
        'symbols':syms,'raw_symbols':raw_symbols,'bits':bits,'hex':packed,
        'recovered':np.asarray(recovered).real[::max(1,len(recovered)//1600)][:1600].astype(float).tolist() if len(recovered) else [],
        'recovered_complex':_complex_points(recovered,1600),
        'telemetry':telemetry,'frame':_trim_telemetry(frame_info) if frame_info is not None else None,
        'input':preview(y,g.sample_rate),'output':probe,'stages':stages,'output_rate':float(fs),
        'freq_hz':np.round(freq,3).tolist(),'psd_db':np.round(db,4).tolist(),
        'rms':float(np.sqrt(np.mean(np.abs(z)**2))) if len(z) else 0.0,
    }


async def live_session(ws: WebSocket) -> None:
    await ws.accept()
    tx, spec = TxConfig(duration=.06), SpectrumConfig(max_fft=32768, averaging=2)
    fps, seq, cursor, dirty, running = 30., 0, 0, True, True
    record = None
    require_ack=False;awaiting_ack=False;sent_at=0.
    pipeline=None;channel=ChannelConfig();rx=None;memory={};pipe_signature=None
    deadline = time.monotonic() + .05
    axis_signature = None
    await ws.send_json({'type':'hello','protocol':'VRF1','recommended_live_sample_rate':RECOMMENDED_LIVE_SAMPLE_RATE})
    try:
        while True:
            # Deadline is independent of incoming edits, so controls cannot starve the stream.
            try:
                message = await asyncio.wait_for(ws.receive_text(), timeout=max(.001, deadline-time.monotonic()))
                req = json.loads(message)
                if req.get('action')=='ack':
                    awaiting_ack=False
                elif req.get('action') == 'stop':
                    running = False
                elif req.get('action') in {'start','update'}:
                    if req.get('action')=='start' and 'capture_id' in req:
                        seq=max(0,min(2147483647,int(req['capture_id'])))
                    require_ack=bool(req.get('acknowledge_frames',require_ack))
                    new_pipeline=Pipeline.model_validate(req['pipeline']) if req.get('pipeline') is not None else None
                    new_channel=ChannelConfig.model_validate(req.get('channel',channel.model_dump()))
                    new_rx=RxConfig.model_validate(req['rx']) if req.get('rx') is not None else None
                    signature=json.dumps([new_pipeline.model_dump(exclude={'probe'}) if new_pipeline else None,new_channel.model_dump(),req.get('tx')],sort_keys=True)
                    new_tx = TxConfig.model_validate(req.get('tx',tx.model_dump()))
                    new_spec = SpectrumConfig.model_validate(req.get('spectrum',spec.model_dump()))
                    if signature!=pipe_signature:memory={};pipe_signature=signature
                    pipeline,channel,rx=new_pipeline,new_channel,new_rx
                    # Spectrum-only edits do not require regenerating the source.
                    dirty = dirty or new_tx != tx
                    tx, spec = new_tx, new_spec
                    fps = float(np.clip(req.get('fps',fps),2,60))
                    running = True
                    await ws.send_json({'type':'accepted'})
            except asyncio.TimeoutError:
                pass
            except (ValueError,TypeError) as exc:
                await ws.send_json({'type':'error','message':str(exc)[:500]})
                continue
            if time.monotonic() < deadline:
                continue
            deadline = time.monotonic()+1/fps
            if awaiting_ack and time.monotonic()-sent_at<2:
                continue
            if not running:
                continue
            try:
                if dirty or record is None:
                    record = await asyncio.to_thread(source_record, tx, seq)
                    cursor, dirty, axis_signature = 0, False, None
                    await ws.send_json({'type':'meta','sample_rate':tx.sample_rate,'waveform':record.metadata.get('waveform',tx.waveform.value),'metadata':{k:v for k,v in record.metadata.items() if not k.startswith('grid_')},'spectrum':spec.model_dump()})
                start = time.perf_counter()
                sr, f, p, cursor = await asyncio.to_thread(frame_product,record,spec,cursor)
                # Frequency coordinates are invariant until source-rate/analyzer
                # settings change.  Sending them once removes most live JSON work.
                current_axis=(float(record.sample_rate),spec.model_dump_json(),len(f),float(f[0]) if len(f) else 0.0,float(f[-1]) if len(f) else 0.0)
                axis_changed=current_axis!=axis_signature
                axis_signature=current_axis
                metrics={'type':'metrics','seq':seq,'source_revision':record.metadata['source_revision'],
                    'rbw_actual_hz':sr.rbw_actual_hz,'bw3_hz':sr.bandwidth_3db_hz,
                    'bw99_hz':sr.bandwidth_99_hz,'peak_hz':sr.peak_hz,'peak_db':sr.peak_db,'power_unit':sr.unit,
                    'capture_id':seq,'simulation_time_s':seq*tx.duration,'display_fps':fps,'compute_ms':1000*(time.perf_counter()-start),'cursor':cursor}
                if axis_changed: metrics['freq_hz']=f.tolist()
                await ws.send_json(metrics)
                await ws.send_bytes(pack_f32(FrameType.PSD,seq,p))
                if rx is not None:
                    product=await asyncio.to_thread(receiver_product,record,channel,pipeline or Pipeline(),rx,seq,memory)
                    await ws.send_json({'type':'receiver',**product})
                elif pipeline is not None:
                    product=await asyncio.to_thread(pipeline_product,record,channel,pipeline,seq,memory)
                    await ws.send_json({'type':'pipeline',**product})
                if require_ack:
                    await ws.send_json({'type':'frame_end','seq':seq})
                    awaiting_ack=True;sent_at=time.monotonic()
                seq=(seq+1)&0xffffffff
            except (ValueError,IndexError) as exc:
                # Keep the last valid stream alive after a bad transient edit.
                await ws.send_json({'type':'error','message':str(exc)[:500]})
                dirty=True
    except (WebSocketDisconnect, RuntimeError):
        return
