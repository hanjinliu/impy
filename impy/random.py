from __future__ import annotations
import numpy as np
from typing import Callable, Literal, TypeVar
import functools
from .arrays import ImgArray, LazyImgArray
from .arrays.bases import MetaArray
from .array_api import xp
from .core import asarray
from .axes import AxesLike

def wraps(npfunc):
    def _wraps(ipfunc):
        ipfunc.__doc__ = npfunc.__doc__
        return ipfunc
    return _wraps

def __getattr__(name: str):
    xpfunc = getattr(xp.random, name)
    @wraps(xpfunc)
    def _func(*args, **kwargs) -> ImgArray:
        name = kwargs.pop("name", xpfunc.__name__)
        axes = kwargs.pop("axes", None)
        out = xp.asnumpy(xpfunc(*args, **kwargs))
        return asarray(out, name=name, axes=axes)
    return _func


def _normalize_like(size, name, axes, like: MetaArray | LazyImgArray | None):
    if like is not None:
        name = name or like.name
        axes = axes or like.axes
        size = size or like.shape
    return size, name, axes
    

@wraps(np.random.random)
def random(
    size, 
    *,
    name: str = None,
    axes: str = None,
    like: MetaArray | LazyImgArray =None,
) -> ImgArray:
    size, name, like = _normalize_like(size, name, axes, like)
    name = name or "random"
    return asarray(xp.asnumpy(xp.random.random(size)), name=name, axes=axes)

@wraps(np.random.normal)
def normal(
    loc: float = 0.0, 
    scale: float = 1.0,
    size=None, 
    *,
    name: str = None, 
    axes: str = None,
    like: MetaArray | LazyImgArray | None = None,
) -> ImgArray:
    size, name, like = _normalize_like(size, name, axes, like)
    name = name or f"normal({loc}, {scale})"
    return asarray(xp.asnumpy(xp.random.normal(loc, scale, size)), name=name, axes=axes)

def random_uint8(
    size: int | tuple[int], 
    *, 
    name: str = None,
    axes: str = None,
    like: MetaArray | LazyImgArray =None,
) -> ImgArray:
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
    size, name, like = _normalize_like(size, name, axes, like)
    arr = xp.random.randint(0, 255, size, dtype=np.uint8)
    name = name or "random_uint8"
    return asarray(xp.asnumpy(arr), name=name, axes=axes)

def random_uint16(
    size,
    *, 
    name: str = None,
    axes: str = None,
    like: MetaArray | LazyImgArray =None,
) -> ImgArray:
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
    size, name, like = _normalize_like(size, name, axes, like)
    arr = xp.random.randint(0, 65535, size, dtype=np.uint16)
    name = name or "random_uint16"
    return asarray(xp.asnumpy(arr), name=name, axes=axes)


def default_rng(seed) -> ImageGenerator:
    """Get the default random number generator."""
    return ImageGenerator(xp.random.default_rng(seed))

class ImageGenerator:
    def __init__(self, rng: xp.random.Generator):
        self._rng = rng
    
    def standard_normal(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype = None,
        *,
        axes: AxesLike | None = None,
        name: str | None = None,
        like: MetaArray | LazyImgArray =None,
    ) -> ImgArray:
        size, name, like = _normalize_like(size, name, axes, like)
        arr = self._rng.standard_normal(size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return asarray(arr, axes=axes, name=name)
    
    def standard_exponential(
        self,
        size: int | tuple[int, ...] | None = None, 
        dtype = None,
        method: Literal["zig", "inv"] = None,
        *,
        axes: AxesLike | None = None,
        name: str | None = None,
        like: MetaArray | LazyImgArray =None,
    ) -> ImgArray:
        size, name, like = _normalize_like(size, name, axes, like)
        arr = self._rng.standard_exponential(size=size, dtype=dtype, method=method)
        if np.isscalar(arr):
            return arr
        return asarray(arr, axes=axes, name=name)
    
    def random(
        self,
        size: int | tuple[int, ...] | None = None, 
        dtype = None,
        *,
        axes: AxesLike | None = None,
        name: str | None = None,
        like: MetaArray | LazyImgArray =None,
    ) -> ImgArray:
        size, name, like = _normalize_like(size, name, axes, like)
        arr = self._rng.random(size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return asarray(arr, axes=axes, name=name)
    
    def normal(
        self,
        loc: float | np.ndarray = 0.,
        scale: float | np.ndarray = 1.,
        size: int | tuple[int, ...] | None = None,
        *,
        axes: AxesLike | None = None,
        name: str | None = None,
        like: MetaArray | LazyImgArray =None,
    ) -> ImgArray:
        size, name, like = _normalize_like(size, name, axes, like)
        arr = self._rng.normal(loc=loc, scale=scale, size=size)
        if np.isscalar(arr):
            return arr
        return asarray(arr, axes=axes, name=name)

    def poisson(
        self,
        lam: float,
        size: int | tuple[int, ...] | None = None,
        *,
        axes: AxesLike | None = None,
        name: str | None = None,
        like: MetaArray | LazyImgArray =None,
    ) -> ImgArray:
        size, name, like = _normalize_like(size, name, axes, like)
        arr = self._rng.poisson(lam=lam, size=size)
        if np.isscalar(arr):
            return arr
        return asarray(arr, axes=axes, name=name)
    
    def random_uint8(
        self,
        size: int | tuple[int], 
        *, 
        name: str = None,
        axes: str = None,
        like: MetaArray | LazyImgArray =None,
    ) -> ImgArray:
        size, name, like = _normalize_like(size, name, axes, like)
        arr = self._rng.integers(0, 255, size, dtype=np.uint8)
        name = name or "random_uint8"
        return asarray(xp.asnumpy(arr), name=name, axes=axes)

    def random_uint16(
        self,
        size,
        *, 
        name: str = None,
        axes: str = None,
        like: MetaArray | LazyImgArray =None,
    ) -> ImgArray:
        size, name, like = _normalize_like(size, name, axes, like)
        arr = self._rng.integers(0, 65535, size, dtype=np.uint16)
        name = name or "random_uint16"
        return asarray(xp.asnumpy(arr), name=name, axes=axes)


del wraps
