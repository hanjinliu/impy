__version__ = "1.12.3"

# TODO
# - nD Kalman filter
# - FSC, FRC
# - Colocalization ... https://note.com/sakulab/n/n0e2cf293cc1e#BGd2U
# - 3D Gabor filter
# - get line from shape layers

import warnings
from .core import (array, zeros, empty, imread, imread_stack, imread_collection, read_meta, 
                   set_cpu, set_verbose, sample_image)
from .binder import bind
from .arrays import ImgArray,  PropArray, Label, PhaseArray
from .frame import MarkerFrame, TrackFrame
from .viewer import gui
import numpy

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
        npfunc = getattr(numpy.random, name)
        def _func(*args, **kwargs):
            out = npfunc(*args, **kwargs)
            return array(out, name=npfunc.__name__)
        return _func

random = Random()
