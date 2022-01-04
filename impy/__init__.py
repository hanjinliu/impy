__version__ = "1.24.4.dev0"

from __future__ import annotations
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
        def _func(*args, **kwargs) -> ImgArray:
            name = kwargs.pop("name", npfunc.__name__)
            axes = kwargs.pop("axes", None)
            out = npfunc(*args, **kwargs)
            return asarray(out, name=name, axes=axes)
        setattr(self, npfunc.__name__, _func)
        return _func
    
    @wraps(np.random.random)
    def random(self, 
               size, 
               *,
               name: str = None,
               axes: str = None) -> ImgArray:
        name = name or "random"
        return asarray(np.random.random(size), name=name, axes=axes)
    
    @wraps(np.random.normal)
    def normal(self, 
               loc=0.0, 
               scale=1.0,
               size=None, 
               *,
               name: str = None, 
               axes: str = None) -> ImgArray:
        return asarray(np.random.normal(loc, scale, size), name="normal")
    
    def random_uint8(self, 
                     size: int | tuple[int], 
                     *, 
                     name: str = None,
                     axes: str = None) -> ImgArray:
        """
        Return a random uint8 image, ranging 0-255.

        Parameters
        ----------
        size : int or tuple of int
            Image shape.
        name : str, optional
            Image name.
        axes : str, optional
            Image axes.
            
        Returns
        -------
        ImgArray
            Random Image in dtype ``np.uint8``.
        """
        arr = np.random.randint(0, 255, size, dtype=np.uint8)
        name = name or "random_uint8"
        return asarray(arr, name=name, axes=axes)
    
    def random_uint16(self, 
                      size,
                      *, 
                      name: str = None,
                      axes: str = None) -> ImgArray:
        """
        Return a random uint16 image, ranging 0-65535.

        Parameters
        ----------
        size : int or tuple of int
            Image shape.
        name : str, optional
            Image name.
        axes : str, optional
            Image axes.
            
        Returns
        -------
        ImgArray
            Random Image in dtype ``np.uint16``.
        """
        arr = np.random.randint(0, 65535, size, dtype=np.uint16)
        name = name or "random_uint16"
        return asarray(arr, name=name, axes=axes)


random = Random()

del logging, wraps