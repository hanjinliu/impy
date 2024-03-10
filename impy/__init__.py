__version__ = "2.3.3"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

import logging

from ._const import Const, SetConst, use

from .collections import DataList, DataDict
from .core import *
from .binder import bind
from .viewer import gui
from .correlation import *
from .arrays import ImgArray, LazyImgArray, BigImgArray, Label  # for typing
from . import random, io, lazy
from .axes import slicer

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
from numpy import (
    uint8, uint16, uint32, uint64,
    int8, int16, int32, int64,
    float16, float32, float64,
    complex64, complex128,
    bool_,
)
