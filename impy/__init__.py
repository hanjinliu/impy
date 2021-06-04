__version__ = "1.10.1"

# TODO
# - napari 0.4.8
# - nD Kalman filter
# - FSC, FRC
# - Colocalization ... https://note.com/sakulab/n/n0e2cf293cc1e#BGd2U
# - 3D Gabor filter
# - get line from shape layers

import warnings
from .core import (array, zeros, zeros_like, empty, empty_like, 
                   imread, imread_stack, imread_collection, read_meta, 
                   stack, set_cpu, set_verbose, sample_image, squeeze,
                   bind_method)
from .imgarray import ImgArray
from .specials import PropArray, MarkerFrame, TrackFrame
from .label import Label
from .phasearray import PhaseArray
from .viewer import window

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

def __getattr__(key):
    import numpy
    from .func import complement_axes
    from .deco import safe_str
    npfunc = getattr(numpy, key)
    def _func(img, **kwargs):
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
        _kwargs = [f"{safe_str(k)}={safe_str(v)}" for k, v in kwargs.items()]
        history = f"np.{npfunc.__name__}({','.join(_kwargs)})"
        out._set_info(img, history, new_axes=complement_axes(axis, img.axes))
        return out
        
    return _func