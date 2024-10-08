__version__ = "2.4.5"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

import logging

from ._const import Const, SetConst, use  # noqa

from .collections import DataList, DataDict  # noqa
from .core import *  # noqa
from .binder import bind  # noqa
from .viewer import gui  # noqa
from .correlation import *  # noqa
from .arrays import ImgArray, LazyImgArray, BigImgArray, Label  # noqa
from . import random, io, lazy  # noqa
from .axes import slicer  # noqa

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
