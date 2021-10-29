from __future__ import annotations
from typing import Sequence, TypeVar
import numpy as np

T = TypeVar("T")

def check_nd(x: T, ndim: int) -> Sequence[T]:
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
