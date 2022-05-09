__version__ = "2.0.1.dev0"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

import logging

from ._const import Const, SetConst, use

from .collections import *
from .core import *
from .binder import bind
from .viewer import gui
from .correlation import *
from .arrays import ImgArray, LazyImgArray, Label  # for typing
from . import random

r"""
Inheritance
-----------
        np.ndarray _    _ AxesMixin _
                    \  /             \
            ____ MetaArray _     LazyImgArray
          /        /        \ 
  LabeledArray  Label    PropArray
    /     \
ImgArray PhaseArray

"""

logging.getLogger("skimage").setLevel(logging.ERROR)
logging.getLogger("tifffile").setLevel(logging.ERROR)


del logging

# dtypes
from numpy import (
	uint8, uint16, uint32, uint64, uint128, uint256,
	int8, int16, int32, int64, int128, int256,
	float16, float32, float64, float128, float256,
	complex64, complex128, complex256, complex512,
	bool_, bool8,
)