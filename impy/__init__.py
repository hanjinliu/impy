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

class Random:
    """
    This class enables practically any numpy.random functions to return ImgArray by such as the
    `ip.random.normal(size=(10, 256, 256))`.
    """
    def __init__(self):
        pass
    
    def __getattribute__(self, name: str):
        npfunc = getattr(np.random, name)
        @wraps(npfunc)
        def _func(*args, **kwargs):
            name = kwargs.pop("name", npfunc.__name__)
            axes = kwargs.pop("axes", None)
            out = npfunc(*args, **kwargs)
            return asarray(out, name=name, axes=axes)
        setattr(self, npfunc.__name__, _func)
        return _func
    
    def random(self, size):
        return asarray(np.random.random(size), name="random")
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        return asarray(np.random.normal(loc, scale, size), name="normal")
    
    def random_uint8(self, size):
        arr = np.random.randint(0, 255, size, dtype=np.uint8)
        return asarray(arr, name="random_uint16")
    
    def random_uint16(self, size):
        arr = np.random.randint(0, 65535, size, dtype=np.uint16)
        return asarray(arr, name="random_uint16")


random = Random()

del logging, wraps