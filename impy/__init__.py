__version__ = "2.5.1"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

import logging
from typing import TYPE_CHECKING

from impy._const import Const, SetConst, use  # noqa

from impy.collections import DataList, DataDict  # noqa
from impy.core import *  # noqa
from impy.binder import bind  # noqa
from impy.correlation import *  # noqa
from impy.arrays import ImgArray, LazyImgArray, BigImgArray, Label  # noqa
from impy import random, io, lazy  # noqa
from impy.axes import slicer  # noqa

# Inheritance
# -----------
#
# AxesMixin -----> LazyImgArray ------> BigImgArray
#     |
#     +-----> MetaArray --+--> LabeledArray --+--> ImgArray
#     |                   |                   |
# np.ndarray              +--> Label          +--> PhaseArray
#                         |
#                         +--> PropArray


logging.getLogger("skimage").setLevel(logging.ERROR)
logging.getLogger("tifffile").setLevel(logging.ERROR)


del logging

# dtypes
from numpy import (  # noqa
    uint8, uint16, uint32, uint64,
    int8, int16, int32, int64,
    float16, float32, float64,
    complex64, complex128,
    bool_,
)

# lazy loading

_VIEWER_CACHE = None
if TYPE_CHECKING:
    from impy.viewer import napariViewers

    gui: "napariViewers"

def __getattr__(key):
    global _VIEWER_CACHE

    if key == "gui":
        from impy.viewer import napariViewers

        if _VIEWER_CACHE is None:
            _VIEWER_CACHE = napariViewers()
        return _VIEWER_CACHE
    else:
        raise AttributeError(f"module 'impy' has no attribute '{key}'")
