__version__ = "1.26.1.dev0"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

import logging
from functools import wraps

from ._const import Const, SetConst, silent, use

from .collections import *
from .core import *
from .binder import bind
from .viewer import gui, GUIcanvas
from .correlation import *
from .arrays import ImgArray, LazyImgArray  # for typing
from . import random

r"""
Inheritance
-----------
        np.ndarray _    _ AxesMixin _
                    \  /             \
              __ MetaArray _     LazyImgArray
             /              \ 
       HistoryArray     PropArray
       /         \    
  LabeledArray   Label  
    /     \
ImgArray PhaseArray

"""

logging.getLogger("skimage").setLevel(logging.ERROR)
logging.getLogger("tifffile").setLevel(logging.ERROR)


del logging, wraps