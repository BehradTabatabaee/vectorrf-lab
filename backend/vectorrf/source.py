"""Deterministic capture identity shared by observers; live captures advance."""
from functools import lru_cache
from hashlib import sha256
from threading import RLock
from .models import TxConfig
from .signal import generate
_lock=RLock()

@lru_cache(maxsize=12)
def _record(config_json,frame):
    cfg=TxConfig.model_validate_json(config_json)
    effective=cfg.model_copy(update={'seed':(cfg.seed+frame*104729)%(2**32)})
    g=generate(effective,start_s=frame*cfg.duration,frame=frame)
    g.samples.flags.writeable=False
    g.metadata.update(source_revision=sha256(config_json.encode()).hexdigest()[:12],
                      capture_id=frame,source_seed=cfg.seed,tx_symbol_rate=cfg.symbol_rate,simulation_time_s=frame*cfg.duration,
                      playback='advancing deterministic simulation captures')
    return g

def source_record(cfg,frame=0):
    if not 0<=frame<=2147483647:raise ValueError('Capture ID outside supported range')
    with _lock:return _record(cfg.model_dump_json(),int(frame))
