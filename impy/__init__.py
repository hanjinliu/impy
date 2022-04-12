__version__ = "2.0.0.alpha"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

import logging
from functools import wraps

from ._const import Const, SetConst, use

from .collections import *
from .core import *
from .binder import bind
from .viewer import gui
from .correlation import *
from .arrays import ImgArray, LazyImgArray  # for typing
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


del logging, wraps