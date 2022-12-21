from __future__ import annotations

import numpy as np
from .lazy import LazyImgArray

from impy._types import nDFloat, Coords, Iterable, Dims, PaddingMode
from impy._const import Const

class BigImgArray(LazyImgArray):
    def gaussian_filter(
        self, 
        sigma: nDFloat = 1,
        *, 
        dims: Dims = None, 
        update: bool = False,
    ) -> BigImgArray:
        out = super().gaussian_filter(sigma, dims=dims, update=update)
        out.release()
        return out
