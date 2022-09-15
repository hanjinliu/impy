from __future__ import annotations

from ._axis import AxisLike, Axis
import numpy as np


class Interpolator:
    def __init__(self, order: int = 3, mode: str = "constant", cval: float = 0.0):
        self._order = order
        self._mode = mode
        self._cval = cval
    
    def __getattr__(self, axis: str) -> _InterpolatorSlice:
        return _InterpolatorSlice(self, axis)

    def __call__(self, axis: str) -> _InterpolatorSlice:
        return _InterpolatorSlice(self, axis)
    

class _InterpolatorSlice:
    ...