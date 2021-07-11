__version__ = "1.15.5"

# TODO
# - 3D Gabor filter
# - get lines/paths from shape layers

import warnings
from .datalist import DataList
from .core import *
from .binder import bind
from .viewer import gui
from .correlation import *
import numpy as np

r"""
Inheritance
-----------
              __ MetaArray _
             /              \ 
       HistoryArray     PropArray
       /         \    
  LabeledArray   Label  
    /     \
ImgArray PhaseArray

"""

# To silence Warnings in skimage
warnings.resetwarnings()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

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

del warnings