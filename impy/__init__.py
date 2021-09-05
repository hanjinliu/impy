__version__ = "1.23.0"

import logging
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
    
    def __getattribute__(self, name:str):
        npfunc = getattr(np.random, name)
        def _func(*args, **kwargs):
            out = npfunc(*args, **kwargs)
            return array(out, name=npfunc.__name__)
        return _func

random = Random()

del logging