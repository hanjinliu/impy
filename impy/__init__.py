__version__ = "1.10.3"

# TODO
# - crop image from shape
# - lucy lower bound
# - nD Kalman filter
# - FSC, FRC
# - Colocalization ... https://note.com/sakulab/n/n0e2cf293cc1e#BGd2U
# - 3D Gabor filter
# - get line from shape layers

import warnings
from .core import (array, zeros, zeros_like, empty, empty_like, 
                   imread, imread_stack, imread_collection, read_meta, 
                   stack, set_cpu, set_verbose, sample_image, squeeze,
                   bind)
from .imgarray import ImgArray
from .specials import PropArray, MarkerFrame, TrackFrame
from .label import Label
from .phasearray import PhaseArray
from .viewer import window
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

def __getattr__(key):
    """
    This builtin function enables practically any numpy functions to take string `axis` argument
    by such as `ip.mean(img, axis="z")`. Also, unlike `np.mean(img)`, ImgArray is converted to 
    np.ndarray inside so that functions are executed as fast as np.ndarray except for the 
    overhead before and after function calls.
    """
    from .func import complement_axes
    from .deco import make_history
    npfunc = getattr(numpy, key)
    def _func(img, *args, **kwargs):
        if not isinstance(img, ImgArray):
            raise TypeError("When numpy functions are called from impy, input must be ImgArray.")
        kw = kwargs.copy()
        if "axis" in kwargs:
            axis = kwargs["axis"]
            if isinstance(axis, str):
                kw["axis"] = tuple(img.axisof(a) for a in axis)
            elif isinstance(axis, tuple):
                axis = "".join(img.axes.axes[i] for i in kwargs["axis"])
            elif isinstance(axis, int):
                axis = img.axes.axes[axis]
        else:
            axis = ""
                
        out = npfunc(img.value, **kw).view(img.__class__)
        
        if isinstance(out, img.__class__):
            history = make_history(f"np.{npfunc.__name__}", args, kwargs)
            out._set_info(img, history, new_axes=complement_axes(axis, img.axes))
        return out
        
    return _func