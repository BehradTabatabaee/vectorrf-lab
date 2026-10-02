from __future__ import annotations

import struct
from enum import IntEnum
import numpy as np

MAGIC=b"VRF1"
HEADER=struct.Struct("<4sHHII")

class FrameType(IntEnum):
    PSD=1
    WATERFALL=2
    CONSTELLATION=3
    TIME_IQ=4
    EYE_IQ=5
    LOOP_TELEMETRY=6


def pack_f32(frame_type: FrameType, seq: int, values: np.ndarray, flags: int=0) -> bytes:
    arr=np.asarray(values,dtype="<f4").reshape(-1)
    payload=arr.tobytes(order="C")
    return HEADER.pack(MAGIC,int(frame_type),flags,seq,len(payload))+payload


def interleave_complex(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x)
    out=np.empty(x.size*2,dtype=np.float32)
    out[0::2]=x.real.astype(np.float32,copy=False)
    out[1::2]=x.imag.astype(np.float32,copy=False)
    return out
