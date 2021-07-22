from __future__ import annotations
import numpy as np

def check_nd(x, ndim:int):
    if np.isscalar(x):
        x = (x,) * ndim
    elif len(x) != ndim:
        raise ValueError("length of parameter and dimension must match.")
    return x
    

def largest_zeros(shape) -> np.ndarray:
    try:
        out = np.zeros(shape, dtype=np.uint64)
    except MemoryError:
        try:
            out = np.zeros(shape, dtype=np.uint32)
        except MemoryError:
            out = np.zeros(shape, dtype=np.uint16)
    return out
