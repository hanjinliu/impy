from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
import weakref

import numpy as np
from impy.utils.axesop import switch_slice
from impy.utils.misc import check_nd
from impy.array_api import xp
from impy._types import nDFloat, Dims
from ._utils import _deconv, _filters
from ._utils._skimage import _get_ND_butterworth_filter

from dask import array as da

if TYPE_CHECKING:
    from .imgarray import ImgArray

class TiledAccessor:
    def __get__(self, instance: ImgArray, owner: type[ImgArray]) -> _PartialTiledImage:
        if instance is None:
            return self
        return _PartialTiledImage(instance)

class _PartialTiledImage:
    def __init__(self, img: ImgArray):
        self._img = weakref.ref(img)
    
    def __call__(
        self, 
        chunks: tuple[int, ...] | Any = "auto", 
        overlap: int | tuple[int, ...] = 32,
        boundary: str = "reflect",
        dims: Dims = None,
    ) -> TiledImage:
        img = self._img()
        if dims is None:
            dims = "".join(a for a in "zyx" if a in img.axes)
        depth = switch_slice(dims, img.axes, overlap, 0)
        return TiledImage(img, chunks, depth, boundary)

class TiledImage:
    def __init__(
        self,
        img: ImgArray,
        chunks: tuple[int, ...] | Any = "auto",
        depth: tuple[int, ...] | int = 32,
        boundary: str = "reflect",
    ):
        if img is None:
            raise RuntimeError("Image has been deleted")
        self._img = weakref.ref(img)
        self._chunks = chunks
        self._depth = depth
        self._boundary = boundary

    @property
    def chunks(self) -> tuple[int, ...]:
        """Chunksize of the tiled image."""
        return self._chunks

    @property
    def depth(self) -> tuple[int, ...]:
        return self._depth

    @property
    def boundary(self) -> str:
        return self._boundary

    def __repr__(self) -> str:
        img = self._img()
        if img is None:
            img_repr = "<deleted image>"
        else:
            img_repr = repr(img)
        return f"TiledImage<chunks={self.chunks}, depth={self.depth}, boundary={self.boundary}> of \n{img_repr}"
    
    def _deref_image(self) -> ImgArray:
        img = self._img()
        if img is None:
            raise RuntimeError("Image has been deleted")
        return img
    
    def _map_overlap(self, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> np.ndarray:
        img = self._deref_image()
        input = da.from_array(img.value, chunks=self._chunks)
        out: np.ndarray = xp.asnumpy(
            da.map_overlap(
                func, 
                input,
                *args,
                depth=self.depth,
                boundary=self.boundary,
                dtype=img.dtype,
                **kwargs,
            ).compute()
        )
        
        return out.view(img.__class__)._set_info(img, img.axes)

    def lowpass_filter(self, cutoff: float = 0.2, order: int = 2) -> ImgArray:
        return self._map_overlap(_lowpass, cutoff=cutoff, order=order)
    
    def lucy(
        self,
        psf: np.ndarray | Callable[[tuple[int, ...]], np.ndarray],
        niter: int = 50,
        eps: float = 1e-5,
    ) -> ImgArray:
        img = self._deref_image()
        scale = tuple(img.scale.values())
        
        def func(arr: np.ndarray):
            psf_ft, psf_ft_conj = _deconv.check_psf(arr.shape, scale, psf)
            return _deconv.richardson_lucy(arr, psf_ft, psf_ft_conj, niter, eps)
            
        return self._map_overlap(func)
    
    def gaussian_filter(self, sigma: float = 1.0, fourier: bool = False) -> ImgArray:
        filter_func = _filters.gaussian_filter_fourier if fourier else _filters.gaussian_filter
        return self._map_overlap(filter_func, sigma=sigma)
    
    def dog_filter(
        self,
        low_sigma: float = 1.0,
        high_sigma: float | None = None,
        fourier: bool = False,
    ) -> ImgArray:
        if high_sigma is None:
            high_sigma = low_sigma * 1.6
        filter_func = _filters.dog_filter_fourier if fourier else _filters.dog_filter
        return self._map_overlap(filter_func, low_sigma=low_sigma, high_sigma=high_sigma)

    def log_filter(self, sigma: float) -> ImgArray:
        return -self._map_overlap(_filters.gaussian_laplace, sigma=sigma)

def _lowpass(arr, cutoff, order=2):
    arr = xp.asarray(arr)
    shape = arr.shape
    _cutoff = check_nd(cutoff, len(shape))
    weight = _get_ND_butterworth_filter(shape, _cutoff, order, False, True)
    ft = xp.asarray(weight) * xp.fft.rfftn(arr)
    ift = xp.fft.irfftn(ft, s=shape)
    return xp.asnumpy(ift)
