__version__ = "1.24.4.dev0"

import logging
from functools import wraps

from ._const import Const, SetConst

from ._cupy import GPU_AVAILABLE
if GPU_AVAILABLE:
    Const._setitem_("RESOURCE", "cupy")
    Const["SCHEDULER"] = "single-threaded"
else:
    Const._setitem_("RESOURCE", "numpy")
del GPU_AVAILABLE

from .collections import *
from .core import *
from .binder import bind
from .viewer import gui, GUIcanvas
from .correlation import *
from .arrays import ImgArray, LazyImgArray # for typing
from . import random
import numpy as np

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