from __future__ import annotations
from functools import wraps
import numpy as np
from .arrays import ImgArray
from .array_api import xp
from .core import asarray

def __getattr__(name: str):
    xpfunc = getattr(xp.random, name)
    @wraps(xpfunc)
    def _func(*args, **kwargs) -> ImgArray:
        name = kwargs.pop("name", xpfunc.__name__)
        axes = kwargs.pop("axes", None)
        out = xp.asnumpy(xpfunc(*args, **kwargs))
        return asarray(out, name=name, axes=axes)
    return _func

@wraps(np.random.random)
def random(size, 
            *,
            name: str = None,
            axes: str = None) -> ImgArray:
    name = name or "random"
    return asarray(xp.asnumpy(xp.random.random(size)), name=name, axes=axes)

@wraps(np.random.normal)
def normal(loc=0.0, 
           scale=1.0,
           size=None, 
           *,
           name: str = None, 
           axes: str = None) -> ImgArray:
    name = name or "normal"
    return asarray(xp.asnumpy(xp.random.normal(loc, scale, size)), name=name, axes=axes)

def random_uint8(size: int | tuple[int], 
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
    arr = xp.random.randint(0, 255, size, dtype=np.uint8)
    name = name or "random_uint8"
    return asarray(xp.asnumpy(arr), name=name, axes=axes)

def random_uint16(size,
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
    arr = xp.random.randint(0, 65535, size, dtype=np.uint16)
    name = name or "random_uint16"
    return asarray(xp.asnumpy(arr), name=name, axes=axes)

