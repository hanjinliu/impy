from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar
import weakref

import numpy as np
from impy.utils.axesop import switch_slice
from impy.utils.misc import check_nd
from impy.array_api import xp
from impy.arrays.axesmixin import AxesMixin
from impy._types import Dims
from ._utils import _deconv, _filters
from ._utils._skimage import _get_ND_butterworth_filter

from dask import array as da

_T = TypeVar("_T", bound=AxesMixin)
Boundary = Literal["reflect", "periodic", "nearest", "none"]

class TiledAccessor(Generic[_T]):
    def __get__(self, instance: _T, owner: type[_T]) -> _PartialTiledImage[_T]:
        if instance is None:
            return self
        return _PartialTiledImage(instance)


class _PartialTiledImage(Generic[_T]):
    def __init__(self, img: _T):
        self._img = weakref.ref(img)
    
    def __call__(
        self, 
        chunks: tuple[int, ...] | Literal["auto"] = "auto", 
        overlap: int | tuple[int, ...] = 32,
        boundary: Boundary | list[Boundary] = "reflect",
        dims: Dims = None,
    ) -> TiledImage:
        img = self._img()
        if img is None:
            raise RuntimeError("Image has been deleted")
        if dims is None:
            dims = "".join(a for a in "zyx" if a in img.axes)
        depth = switch_slice(dims, img.axes, overlap, 0)
        return TiledImage(img, chunks, depth, boundary)


class TiledImage(Generic[_T]):
    def __init__(
        self,
        img: _T,
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        depth: tuple[int, ...] | int = 32,
        boundary: Boundary | list[Boundary] = "reflect",
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
        """Depth of overlaps"""
        return self._depth

    @property
    def boundary(self) -> str:
        """How to handle the boundary of the image."""
        return self._boundary

    def __repr__(self) -> str:
        img = self._img()
        if img is None:
            img_repr = "<deleted image>"
        else:
            img_repr = repr(img)
        return f"TiledImage<chunks={self.chunks}, depth={self.depth}, boundary={self.boundary}> of \n{img_repr}"
    
    def _deref_image(self) -> _T:
        img = self._img()
        if img is None:
            raise RuntimeError("Image has been deleted")
        return img
    
    def _map_overlap(self, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> np.ndarray:
        from .imgarray import ImgArray
        from .lazy import LazyImgArray

        img = self._deref_image()
        if isinstance(img, ImgArray):
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
            out = out.view(img.__class__)._set_info(img, img.axes)
        elif isinstance(img, LazyImgArray):
            if self._chunks != "auto":
                img = img.rechunk(self._chunks)
            out = img._apply_map_overlap(
                func,
                c_axes="",
                depth=self.depth,
                boundary=self.boundary, 
                args=args,
                kwargs=kwargs,
            )
            out = LazyImgArray(out)._set_info(img)
        else:
            raise TypeError(f"Cannot tile {type(img)}")
        return out

    def lowpass_filter(self, cutoff: float = 0.2, order: int = 2) -> _T:
        """
        Tile-wise butterworth lowpass filter.

        Parameters
        ----------
        cutoff : float or array-like, default is 0.2
            Cutoff frequency.
        order : float, default is 2
            Steepness of cutoff.
        """
        return self._map_overlap(_lowpass, cutoff=cutoff, order=order)
    
    def lucy(
        self,
        psf: np.ndarray | Callable[[tuple[int, ...]], np.ndarray],
        niter: int = 50,
        eps: float = 1e-5,
    ) -> _T:
        """
        Deconvolution of N-dimensional image, using Richardson-Lucy's algorithm.
        
        Parameters
        ----------
        psf : ndarray or callable
            Point spread function. If a function is given, `psf(shape)` will be
            called to generate the PSF.
        niter : int, default is 50.
            Number of iterations.
        eps : float, default is 1e-5
            During deconvolution, division by small values in the convolve image 
            of estimation and PSF may cause divergence. Therefore, division by 
            values under `eps` is substituted to zero.
        """
        img = self._deref_image()
        scale = tuple(img.scale.values())
        
        def func(arr: np.ndarray):
            psf_ft, psf_ft_conj = _deconv.check_psf(arr.shape, scale, psf)
            return _deconv.richardson_lucy(arr, psf_ft, psf_ft_conj, niter, eps)
            
        return self._map_overlap(func)
    
    def gaussian_filter(self, sigma: float = 1.0, fourier: bool = False) -> _T:
        """
        Run Gaussian filter (Gaussian blur).
        
        Parameters
        ----------
        {sigma}{fourier}
        """
        filter_func = _filters.gaussian_filter_fourier if fourier else _filters.gaussian_filter
        return self._map_overlap(filter_func, sigma=sigma)
    
    def dog_filter(
        self,
        low_sigma: float = 1.0,
        high_sigma: float | None = None,
        fourier: bool = False,
    ) -> _T:
        """
        Run Difference of Gaussian filter. This function does not support `update`
        argument because intensity can be negative.
        
        Parameters
        ----------
        low_sigma : scalar or array of scalars, default is 1.
            lower standard deviation(s) of Gaussian.
        high_sigma : scalar or array of scalars, default is x1.6 of low_sigma.
            higher standard deviation(s) of Gaussian.
        {fourier}
        """        
        if high_sigma is None:
            high_sigma = low_sigma * 1.6
        filter_func = _filters.dog_filter_fourier if fourier else _filters.dog_filter
        return self._map_overlap(filter_func, low_sigma=low_sigma, high_sigma=high_sigma)

    def log_filter(self, sigma: float) -> _T:
        """
        Laplacian of Gaussian filter.

        Parameters
        ----------
        {sigma}
        """
        return -self._map_overlap(_filters.gaussian_laplace, sigma=sigma)


def _lowpass(arr, cutoff, order=2):
    arr = xp.asarray(arr)
    shape = arr.shape
    _cutoff = check_nd(cutoff, len(shape))
    weight = _get_ND_butterworth_filter(shape, _cutoff, order, False, True)
    ft = xp.asarray(weight) * xp.fft.rfftn(arr)
    ift = xp.fft.irfftn(ft, s=shape)
    return xp.asnumpy(ift)
