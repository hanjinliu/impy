from __future__ import annotations
from functools import wraps
import os
import itertools
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from warnings import warn
import tempfile

from .labeledarray import LabeledArray
from .imgarray import ImgArray
from .axesmixin import AxesMixin, get_axes_tuple
from ._utils._skimage import skres
from ._utils import _misc, _transform, _structures, _filters, _deconv, _corr, _docs

from ..utils.axesop import slice_axes, switch_slice, complement_axes, find_first_appeared
from ..utils.deco import check_input_and_output_lazy, dims_to_spatial_axes, same_dtype
from ..utils.misc import check_nd
from ..utils import slicer
from ..io import imsave
from ..collections import DataList

from .._types import nDFloat, Coords, Iterable, Dims, PaddingMode
from ..axes import ImageAxesError
from .._const import Const
from ..array_api import xp

if TYPE_CHECKING:
    from dask import array as da


class LazyImgArray(AxesMixin):
    additional_props = ["_source", "_metadata", "_name"]
    
    def __init__(
        self,
        obj: "da.core.Array",
        name: str = None,
        axes: str = None,
        source: str = None, 
        metadata: dict = None
    ):
        from dask import array as da
        if not isinstance(obj, da.core.Array):
            raise TypeError(f"The first input must be dask array, got {type(obj)}")
        self.value: "da.core.Array" = obj
        self.source = source
        self._name = name
        self.axes = axes
        self._metadata = metadata or {}
        
    @property
    def source(self):
        return self._source
    
    @source.setter
    def source(self, val):
        if val is None:
            self._source = None
        else:
            self._source = Path(val)
    
    @property
    def name(self) -> str:
        if self._name is None:
            source = self.source
            if source is None:
                return "No name"
            else:
                return source.name
        else:
            return self._name
    
    @name.setter
    def name(self, val):
        self._name = str(val)
    
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def ndim(self):
        return self.value.ndim
    
    @property
    def shape(self):
        try:
            tup = get_axes_tuple(self)
            return tup(*self.value.shape)
        except ImageAxesError:
            return self.value.shape
    
    @property
    def dtype(self):
        return self.value.dtype
    
    @property
    def size(self):
        return self.value.size
    
    @property
    def itemsize(self):
        return self.value.itemsize
    
    @property
    def chunksize(self):
        try:
            tup = get_axes_tuple(self)
            return tup(*self.value.chunksize)
        except ImageAxesError:
            return self.value.chunksize
        
    @property
    def GB(self) -> float:
        """Return the array size in GB."""
        return self.value.nbytes / 1e9
    
    gb = GB  # alias
    
    def __array__(self):
        # Should not be `self.compute` because in napari Viewer this function is called every time
        # sliders are moved.
        return xp.asnumpy(self.value.compute())
    
    def __getitem__(self, key):
        key = slicer.solve_slicer(key, self.axes)
            
        new_axes = slice_axes(self.axes, key)
        out = self.__class__(
            self.value[key], 
            name=self.name,
            source=self.source, 
            axes=new_axes, 
            metadata=self.metadata
        )
        
        out._getitem_additional_set_info(
            self, new_axes=new_axes, key=key
        )
        
        return out
    
    def __neg__(self) -> LazyImgArray:
        """Invert array."""
        out = self.__class__(-self.value)
        out._set_info(self)
        return out
    
    @same_dtype(asfloat=True)
    def __add__(self, other) -> LazyImgArray:
        if isinstance(other, self.__class__):
            out = self.value + other.value
        else:
            out = self.value + other
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    @same_dtype(asfloat=True)
    def __iadd__(self, other) -> LazyImgArray:
        if isinstance(other, self.__class__):
            self.value += other.value
        else:
            self.value += other
        return self
    
    @same_dtype(asfloat=True)
    def __sub__(self, other) -> LazyImgArray:
        if isinstance(other, self.__class__):
            out = self.value - other.value
        else:
            out = self.value - other
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    @same_dtype(asfloat=True)
    def __isub__(self, other) -> LazyImgArray:
        if isinstance(other, self.__class__):
            self.value -= other.value
        else:
            self.value -= other
        return self
    
    @same_dtype(asfloat=True)
    def __mul__(self, other) -> LazyImgArray:
        if isinstance(other, np.ndarray) and other.dtype.kind != "c":
            other = other.astype(np.float32)
            other = other
        elif isinstance(other, self.__class__) and other.dtype.kind != "c":
            other = other.as_float()
            other = other.value
        else:
            other = other
        out = self.value * other
        out = self.__class__(out)
        return out
    
    @same_dtype(asfloat=True)
    def __imul__(self, other) -> LazyImgArray:
        if isinstance(other, np.ndarray) and other.dtype.kind != "c":
            other = other.astype(np.float32)
            other = other
        elif isinstance(other, self.__class__) and other.dtype.kind != "c":
            other = other.as_float()
            other = other.value
        else:
            other = other
        self.value *= other
        return self
    
    def __truediv__(self, other) -> LazyImgArray:        
        self = self.as_float()
        if isinstance(other, np.ndarray) and other.dtype.kind != "c":
            other = other.astype(np.float32)
            other[other==0] = np.inf
            other = other
        elif isinstance(other, self.__class__) and other.dtype.kind != "c":
            other = other.as_float()
            other[other==0] = np.inf
            other = other.value
        elif np.isscalar(other) and other <= 0:
            raise ValueError("Cannot multiply negative value.")
        else:
            other = other
        out = self.value / other
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    def __itruediv__(self, other) -> LazyImgArray:
        if self.dtype.kind in "ui":
            raise ValueError("Cannot divide integer inplace.")
        if isinstance(other, np.ndarray) and other.dtype.kind != "c":
            other = other.astype(np.float32)
            other[other==0] = np.inf
            other = other
        elif isinstance(other, self.__class__) and other.dtype.kind != "c":
            other = other.as_float()
            other[other==0] = np.inf
            other = other.value
        elif np.isscalar(other) and other < 0:
            raise ValueError("Cannot multiply negative value.")
        else:
            other = other
        self.value /= other
        return self
    
    @property
    def chunk_info(self):
        chunk_info = ", ".join([f"{s}({o})" for s, o in zip(self.chunksize, self.axes)])
        return chunk_info
    
    def _repr_dict_(self):
        return {
            "name": self.name,
            "shape": self.shape_info,
            "chunk sizes": self.chunk_info,
            "dtype": self.dtype,
            "source": self.source,
            "scale": self.scale,
        }
    
    
    def compute(self, ignore_limit: bool = False) -> ImgArray:
        """
        Compute all the task and convert the result into ImgArray. If image size overwhelms MAX_GB
        then MemoryError is raised.
        """        
        if self.gb > Const["MAX_GB"] and not ignore_limit:
            raise MemoryError(f"Too large: {self.gb:.2f} GB")
        arr = self.value.compute()
        if arr.ndim > 0:
            img = xp.asnumpy(arr).view(ImgArray)
            for attr in ["_name", "_source", "axes", "_metadata"]:
                setattr(img, attr, getattr(self, attr, None))
        else:
            img = arr
        return img
    
    def release(self, update: bool = True) -> LazyImgArray:
        """
        Compute all the tasks and store the data in memory map, and read it as a dask array
        again.
        """
        from dask import array as da
        with tempfile.NamedTemporaryFile() as ntf:
            mmap = np.memmap(ntf, mode="w+", shape=self.shape, dtype=self.dtype)
            mmap[:] = self.value[:]
        
        img = da.from_array(mmap, chunks=self.chunksize).map_blocks(
            np.array, meta=np.array([], dtype=self.dtype)
            )
        if update:
            self.value = img
            out = self
        else:
            out = self.__class__(img)
            out._set_info(self)
                
        return out
    
    @_docs.copy_docs(LabeledArray.imsave)
    def imsave(self, save_path: str | Path, *, dtype = None, overwrite: bool = True):
        save_path = Path(save_path)
        if self.ndim < 2:
            raise ValueError("Cannot save <2D array as an image.")

        if save_path.suffix == "":
            if self.source is not None:
                ext = self.source.suffix
                if ext == "":
                    ext = ".tif"
            else:
                ext = ".tif"
            save_path = save_path.parent / (save_path.name + ext)
        
        if not Path(save_path).is_absolute():
            if self.source is None:
                raise ValueError(
                    "Image directory path is unknown. Set by \n"
                    " >>> img.source = \"...\"\n"
                    "or specify absolute path like\n"
                    " >>> img.imsave(\"/path/to/XXX.tif\")"
                    )
            save_path = self.source.parent / save_path
        
        if not overwrite and save_path.exists():
            raise FileExistsError(f"File {save_path!r} already exists.")
        if dtype is None:
            dtype = self.dtype
        
        self = self.as_img_type(dtype).sort_axes()
        imsave(save_path, self, lazy=True)
        
        return None
    
    def rechunk(
        self,
        chunks="auto",
        *,
        threshold=None,
        block_size_limit=None,
        balance=False, 
        update=False
    ) -> LazyImgArray:
        """
        Rechunk the bound dask array.

        Parameters
        ----------
        chunks, threshold, block_size_limit, balance
            Passed directly to dask.array's rechunk

        Returns
        -------
        LazyImgArray
            Rechunked dask array is bound.
        """        
        rechunked = self.value.rechunk(
            chunks=chunks, threshold=threshold, 
            block_size_limit=block_size_limit, balance=balance
        )
        if update:
            self.value = rechunked
            return self
        else:
            out = self.__class__(rechunked)
            out._set_info(self)
            return out
        
    def apply_dask_func(self, funcname: str, *args, **kwargs) -> LazyImgArray:
        """
        Apply dask array function to the connected dask array.

        Parameters
        ----------
        funcname : str
            Name of function to apply.
        args, kwargs :
            Parameters that will be passed to `funcname`.

        Returns
        -------
        LazyImgArray
            Updated one
        """        
        
        out = getattr(self.value, funcname)(*args, **kwargs)
        out = self.__class__(out)
        new_axes = self._INHERIT if out.shape == self.shape else None
        out._set_info(self, new_axes=new_axes)
        return out
    
    def _apply_function(
        self, 
        func: Callable,
        c_axes: str = None,
        drop_axis: Iterable[int] = [],
        new_axis: Iterable[int] = None, 
        dtype = np.float32,
        rechunk_to: tuple[int, ...] | str = "none",
        dask_wrap: bool = False,
        args: tuple = None, kwargs: dict[str] = None
    ) -> LazyImgArray:
        """
        Rechunk array in a correct shape and apply function using `map_blocks`. This function is similar
        to the `apply_dask` function in `MetaArray` while returns dask array bound LazyImgArray.

        Parameters
        ----------
        func : callable
            Function to apply for each chunk.
        c_axes : str, optional
            Axes to iterate.
        drop_axis : Iterable[int], optional
            Passed to map_blocks.
        new_axis : Iterable[int], optional
            Passed to map_blocks.
        dtype : any that can be converted to np.dtype object, default is np.float32
            Output data type.
        rechunk_to : tuple[int,...], optional
            In what size input array should be rechunked before `map_blocks` iteration. If str is given, 
            array will be rechunked in following rules:
            - "none": No rechunking
            - "default": Rechunked with "auto" method for each spatial dimension.
            - "max": Rechunked to the shape size for each spatial dimension.
        dask_wrap : bool, optional
            If True, for each chunk array will be converted to dask and rechunked with "auto" option
            before function call.
        args : tuple, optional
            Arguments that will passed to `func`.
        kwargs : dict
            Keyword arguments that will passed to `func`.

        Returns
        -------
        LazyImgArray
            Dask array after function is applied is bound to this newly generated object.
        """        
        slice_in = []
        slice_out = []
        for a in self.axes:
            if a in c_axes:
                slice_in.append(0)
                slice_out.append(np.newaxis)
            else:
                slice_in.append(slice(None))
                slice_out.append(slice(None))
                
        slice_in = tuple(slice_in)
        slice_out = tuple(slice_out)
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()
        
        if rechunk_to == "none":
            input_ = self.value
        else:
            if rechunk_to == "default":
                rechunk_to = switch_slice(c_axes, self.axes, ifin=1, ifnot="auto")
            
            elif rechunk_to == "max":
                rechunk_to = switch_slice(c_axes, self.axes, ifin=1, ifnot=self.shape)
                
            input_ = self.value.rechunk(rechunk_to)
        
        if dask_wrap:
            from dask import array as da
            @wraps(func)
            def _func(arr, *args, **kwargs):
                out = func(da.from_array(arr[slice_in]), *args, **kwargs)
                return out[slice_out].compute()
        else:
            @wraps(func)
            def _func(arr, *args, **kwargs):
                out = func(arr[slice_in], *args, **kwargs)
                return out[slice_out]
        
        out = input_.map_blocks(_func, *args, drop_axis=drop_axis, new_axis=new_axis, 
                                meta=xp.array([], dtype=dtype), **kwargs)
        return out
    
    def _apply_map_blocks(
        self,
        func: Callable,
        c_axes: str = None, 
        args: tuple = None,
        kwargs: dict[str, Any] = None
    ):
        from dask import array as da
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        all_axes = str(self.axes)
        def _func(input: ArrayLike, *args, **kwargs):
            out = xp.empty(input.shape, input.dtype)
            for sl in iter_slice(input.shape, c_axes, all_axes):
                out[sl] = func(input[sl], *args, **kwargs)
            return out
        
        return da.map_blocks(_func, self.value, dtype=self.dtype, *args, **kwargs)
    
    def _apply_map_overlap(
        self,
        func: Callable,
        c_axes: str = None,
        depth: int = 16,
        boundary="reflect",
        dtype: DTypeLike = None,
        args: tuple = None,
        kwargs: dict[str, Any] = None
    ):
        from dask import array as da
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if dtype is None:
            dtype = self.dtype
        all_axes = str(self.axes)
        def _func(input: ArrayLike, *args, **kwargs):
            out = xp.empty(input.shape, input.dtype)
            for sl in iter_slice(input.shape, c_axes, all_axes):
                out[sl] = func(input[sl], *args, **kwargs)
            return out
        depth = switch_slice(c_axes, self.axes, 0, depth)
        return da.map_overlap(
            _func, self.value, depth=depth, boundary=boundary, dtype=dtype,
            *args, **kwargs
        )
    
    def _apply_dask_filter(
        self,
        func: Callable,
        c_axes: str = None,
        args: tuple = None, 
        kwargs: dict[str, Any] = None
    ) -> LazyImgArray:
        # TODO: This is not efficient. Maybe using da.stack is better?
        from dask import array as da
        out = da.empty_like(self.value)
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        for sl, img in self.iter(c_axes, israw=False):
            out[sl] = func(img, *args, **kwargs)
        return out
    
    # TODO: This should wait for dask-image implement map_coordinates
    # @_docs.copy_docs(LabeledArray.rotated_crop)
    # @dims_to_spatial_axes
    # @record_lazy
    # def rotated_crop(self, origin, dst1, dst2, dims=2) -> LazyImgArray:
    #     origin = np.asarray(origin)
    #     dst1 = np.asarray(dst1)
    #     dst2 = np.asarray(dst2)
    #     ax0 = _misc.make_rotated_axis(origin, dst2)
    #     ax1 = _misc.make_rotated_axis(dst1, origin)
    #     all_coords = ax0[:, np.newaxis] + ax1[np.newaxis] - origin
    #     all_coords = np.moveaxis(all_coords, -1, 0)
        
        # Because output shape changes, we have to tell dask what chunk size it would be, otherwise output
        # shape is estimated in a wrong way. 
        # output_chunks = [1] * self.ndim
        # for i, a in enumerate(dims):
        #     it = self.axisof(a)
        #     output_chunks[it] = all_coords.shape[i+1]
        #
        # cropped_img = self._apply_function(xp.ndi.map_coordinates, 
        #                          c_axes=complement_axes(dims, self.axes), 
        #                          dtype=self.dtype,
        #                          rechunk_to="max",
        #                          args=(xp.asarray(all_coords),),
        #                          kwargs=dict(prefilter=False, order=1, chunks=output_chunks)
        #                          )
        
    
    @_docs.copy_docs(ImgArray.erosion)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def erosion(
        self,
        radius: float = 1,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        c_axes = complement_axes(dims, self.axes)
        filter_func = xp.ndi.grey_erosion if self.dtype != bool else xp.ndi.binary_erosion
        
        return self._apply_map_overlap(
            filter_func, 
            c_axes=c_axes,
            depth=_ceilint(radius),
            kwargs=dict(footprint=disk)
        )
        
    @_docs.copy_docs(ImgArray.dilation)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def dilation(
        self,
        radius: float = 1,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        c_axes = complement_axes(dims, self.axes)
        filter_func = xp.ndi.grey_dilation if self.dtype != bool else xp.ndi.binary_dilation
        
        return self._apply_map_overlap(
            filter_func, 
            c_axes=c_axes,
            depth=_ceilint(radius),
            kwargs=dict(footprint=disk)
        )
    
    @_docs.copy_docs(ImgArray.opening)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def opening(
        self,
        radius: float = 1,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        c_axes = complement_axes(dims, self.axes)
        filter_func = xp.ndi.grey_opening if self.dtype != bool else xp.ndi.binary_opening
        
        return self._apply_map_overlap(
            filter_func, 
            c_axes=c_axes,
            depth=_ceilint(radius)*2,
            kwargs=dict(footprint=disk)
        )
    
    @_docs.copy_docs(ImgArray.closing)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def closing(
        self,
        radius: float = 1,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        c_axes = complement_axes(dims, self.axes)
        filter_func = xp.ndi.grey_closing if self.dtype != bool else xp.ndi.binary_closing
        
        return self._apply_map_overlap(
            filter_func, 
            c_axes=c_axes,
            depth=_ceilint(radius)*2,
            kwargs=dict(footprint=disk)
        )
    
    @_docs.copy_docs(ImgArray.gaussian_filter)
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    def gaussian_filter(
        self,
        sigma: nDFloat = 1.0,
        *, 
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        c_axes = complement_axes(dims, self.axes)
        depth = _ceilint(sigma*4)
        return self._apply_map_overlap(
            xp.ndi.gaussian_filter, 
            c_axes=c_axes,
            depth=depth,
            kwargs=dict(sigma=sigma),
        )
    
    @_docs.copy_docs(ImgArray.spline_filter)
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    def spline_filter(
        self,
        order: int = 3,
        mode: PaddingMode = "mirror", 
        *,
        dims: Dims = None,
        update: bool = False,
    ):
        # TODO: need test
        depth = order
        return self._apply_map_overlap(
            _filters.spline_filter,
            c_axes=complement_axes(dims, self.axes), 
            depth=depth,
            args=(order, np.float32, mode),
        )
    
    
    @_docs.copy_docs(ImgArray.median_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def median_filter(
        self,
        radius: float = 1,
        *, 
        dims: Dims = None, 
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_map_overlap(
            xp.ndi.median_filter,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(footprint=disk)
            )
        
    @_docs.copy_docs(ImgArray.mean_filter)
    @same_dtype(asfloat=True)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def mean_filter(
        self,
        radius: float = 1,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        kernel = (disk/np.sum(disk)).astype(np.float32)
        return self._apply_map_overlap(
            xp.ndi.convolve,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(weights=kernel),
            )
    
    @_docs.copy_docs(ImgArray.convolve)
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    def convolve(
        self,
        kernel,
        *, 
        mode: str = "reflect", 
        cval: float = 0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        kernel = np.asarray(kernel)
        shape = np.array(kernel.shape)
        half_size = shape // 2
        depth = tuple(half_size)
        c_axes = complement_axes(dims, self.axes)
        return self._apply_map_overlap(
            xp.ndi.convolve, 
            c_axes=c_axes,
            depth=depth,
            kwargs=dict(weights=kernel, mode=mode, cval=cval),
        )
    
    @_docs.copy_docs(ImgArray.edge_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def edge_filter(
        self,
        method: str = "sobel", 
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        # BUG: returns zero array
        from ._utils._skimage import skfil
        method_dict = {
            "sobel": (skfil.sobel, 1),
            "farid": (skfil.farid, 2),
            "scharr": (skfil.scharr, 1),
            "prewitt": (skfil.prewitt, 1)
        }
        try:
            filter_func, depth = method_dict[method]
        except KeyError:
            raise ValueError("`method` must be 'sobel', 'farid' 'scharr', or 'prewitt'.")
        return self._apply_map_overlap(
            filter_func,
            depth=depth,
            c_axes=complement_axes(dims, self.axes)
        )
    
    @_docs.copy_docs(ImgArray.laplacian_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def laplacian_filter(
        self,
        radius: int = 1,
        *, 
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:  
        ndim = len(dims)
        _, laplace_op = skres.uft.laplacian(ndim, (2*radius+1,) * ndim)
        return self._apply_map_overlap(
            xp.ndi.convolve,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(weights=laplace_op),
        )
    
    @_docs.copy_docs(ImgArray.affine)
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    def affine(
        self, 
        matrix=None,
        scale=None,
        rotation=None,
        shear=None,
        translation=None,
        *,
        mode="constant", 
        cval=0,
        output_shape=None,
        order=1,
        dims=None
    ) -> LazyImgArray:
        if matrix is None:
            matrix = _transform.compose_affine_matrix(
                scale=scale, rotation=rotation, 
                shear=shear, translation=translation,
                ndim=len(dims)
            )
        from dask_image.ndinterp import affine_transform
        return self._apply_dask_filter(
            affine_transform,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(matrix=matrix, mode=mode, cval=cval, 
                        output_shape=output_shape, order=order)
        )
    
    @_docs.copy_docs(ImgArray.kalman_filter)
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    def kalman_filter(self, gain: float = 0.8, noise_var: float = 0.05, *, along: str = "t", 
                      dims: Dims = None, update: bool = False) -> LazyImgArray:
        if self.axisof(along) != 0:
            raise ValueError("Currently kalman_filter does not support t-axis != 0.")
        
        return self._apply_map_blocks(
            _filters.kalman_filter, 
            c_axes=complement_axes([along] + dims, self.axes), 
            args=(gain, noise_var)
        )
    
    @_docs.copy_docs(ImgArray.fft)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def fft(self, *, shape: int | Iterable[int] | str = "same", shift: bool = True, 
            dims: Dims = None) -> LazyImgArray:
        from dask import array as da
        axes = [self.axisof(a) for a in dims]
        if shape == "square":
            s = 2**int(np.ceil(np.max(self.sizesof(dims))))
            shape = (s,) * len(dims)
        elif shape == "same":
            shape = None
        else:
            shape = check_nd(shape, len(dims))
        freq = da.fft.fftn(self.value.astype(np.float32), s=shape, axes=axes).astype(np.complex64)
        if shift:
            freq[:] = da.fft.fftshift(freq, axes=axes)
        return freq

    @_docs.copy_docs(ImgArray.ifft)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def ifft(self, real:bool=True, *, shift:bool=True, dims=None) -> LazyImgArray:
        from dask import array as da
        axes = [self.axisof(a) for a in dims]
        if shift:
            freq = da.fft.ifftshift(self.value, axes=axes)
        else:
            freq = self.value
        out = da.fft.ifftn(freq, axes=axes).astype(np.complex64)
        
        if real:
            out = da.real(out)
        
        return out
    
    @_docs.copy_docs(ImgArray.power_spectra)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def power_spectra(self, shape = "same", norm: bool = False, zero_norm: bool = False, *,
                      dims: Dims = None) -> LazyImgArray:
        freq = self.fft(dims=dims, shape=shape)
        pw = freq.value.real**2 + freq.value.imag**2
        if norm:
            pw /= pw.max()
        if zero_norm:
            sl = switch_slice(dims, pw.axes, ifin=np.array(pw.shape)//2, ifnot=slice(None))
            pw[sl] = 0
        return pw
    
    def chunksizeof(self, axis:str):
        return self.value.chunksize[self.axes.find(axis)]
    
    def chunksizesof(self, axes:str):
        return tuple(self.chunksizeof(a) for a in axes)
        
    def transpose(self, axes):
        new_axes = [self.axes[i] for i in list(axes)]
        out = self.__class__(self.value.transpose(axes))
        out._set_info(self, new_axes=new_axes)
        return out
    
    def sort_axes(self):
        order = self.axes.argsort()
        return self.transpose(tuple(order))
    
    @_docs.copy_docs(LabeledArray.crop_center)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def crop_center(self, scale=0.5, *, dims=2) -> LazyImgArray:
        # check scale
        if hasattr(scale, "__iter__") and len(scale) == 3 and len(dims) == 2:
            dims = "zyx"
        scale = np.asarray(check_nd(scale, len(dims)))
        if np.any((scale <= 0) | (1 < scale)):
            raise ValueError(f"scale must be (0, 1], but got {scale}")
        
        # Make axis-targeted slicing string
        sizes = self.sizesof(dims)
        slices = []
        for a, size, sc in zip(dims, sizes, scale):
            x0 = int(size / 2 * (1 - sc))
            x1 = int(np.ceil(size / 2 * (1 + sc)))
            slices.append(f"{a}={x0}:{x1}")

        out = self[";".join(slices)]
        
        return out
    
    @_docs.copy_docs(ImgArray.tiled_lowpass_filter)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def tiled_lowpass_filter(self, cutoff: float = 0.2, order: int = 2, overlap: int = 16, *,
                             dims: Dims = None, update: bool = False) -> LazyImgArray:
        from ._utils._skimage import _get_ND_butterworth_filter
        self = self.as_float()
        c_axes = complement_axes(dims, self.axes)
        ndims = len(dims)
        cutoff = check_nd(cutoff, ndims)
        if all((c >= 0.5*np.sqrt(ndims) or c <= 0) for c in cutoff):
            return self
        
        depth = switch_slice(dims, self.axes, overlap, 0)
        
        def func(arr):
            arr = xp.asarray(arr)
            shape = arr.shape
            weight = _get_ND_butterworth_filter(shape, cutoff, order, False, True)
            ft = xp.asarray(weight) * xp.fft.rfftn(arr)
            ift = xp.fft.irfftn(ft, s=shape)
            return ift
        
        out = self._apply_map_overlap(func, c_axes=c_axes, depth=depth, boundary="reflect")
        return out
    
    @_docs.copy_docs(ImgArray.proj)
    def proj(self, axis: str = None, method: str = "mean") -> LazyImgArray:
        from dask import array as da
        if axis is None:
            axis = find_first_appeared("ztpi<c", include=self.axes, exclude="yx")
        elif not isinstance(axis, str):
            raise TypeError("`axis` must be str.")
        axisint = [self.axisof(a) for a in axis]
        
        if method == "mean":
            projection = getattr(da, method)(self.value, axis=tuple(axisint), dtype=np.float32)
        else:
            projection = getattr(da, method)(self.value, axis=tuple(axisint))
        
        out = self.__class__(projection)
        out._set_info(self, self.axes.drop(axisint))
        return out
    
    @_docs.copy_docs(ImgArray.binning)
    @dims_to_spatial_axes
    @same_dtype
    def binning(self, binsize: int = 2, method = "mean", *, check_edges: bool = True, dims: Dims = None) -> LazyImgArray:
        if binsize == 1:
            return self
        
        if isinstance(method, str):
            binfunc = getattr(xp, method)
        elif callable(method):
            binfunc = method
        else:
            raise TypeError("`method` must be a numpy function or callable object.")
        
        img_to_reshape, shape, scale_ = _misc.adjust_bin(self.value, binsize, check_edges, dims, self.axes)
        
        reshaped_img = img_to_reshape.reshape(shape)
        axes_to_reduce = tuple(i*2+1 for i in range(self.ndim))
        out = binfunc(reshaped_img, axis=axes_to_reduce)
        out = self.__class__(out)
        out._set_info(self)
        out.axes = str(self.axes) # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out
    
    @_docs.copy_docs(ImgArray.track_drift)
    def track_drift(self, along: str = None, upsample_factor: int = 10) -> "da.core.Array":
        if along is None:
            along = find_first_appeared("tpzc<i", include=self.axes)
        elif len(along) != 1:
            raise ValueError("`along` must be single character.")
                    
        dims = complement_axes(along, self.axes)
        chunks = switch_slice(dims, self.axes, ifin=self.shape, ifnot=1)
        img_fft = self.fft(shift=False, dims=dims).value.rechunk(chunks)
        ndim = len(dims)
        slice_out = (np.newaxis, slice(None)) + (np.newaxis,)*(ndim-1)
        each_shape = (1, ndim) + (1,)*(ndim-1)
        len_t = self.sizeof(along)
        
        def pcc(x):
            if x.shape[0] < 2:
                return np.array([0]*ndim, dtype=np.float32).reshape(*each_shape)
            x = xp.asarray(x)
            result = _corr.subpixel_pcc(x[0], x[1], upsample_factor=upsample_factor)[0]
            return xp.asnumpy(result[slice_out])
        
        from dask import array as da

        # I don't know the reason why but output dask array's chunk size along t-axis should be 
        # specified to be 1, and rechunk it map_overlap. 
        result = da.map_overlap(pcc, img_fft, 
                                depth={0: (1, 0)}, 
                                trim=False,
                                boundary="none",
                                chunks=(1, ndim) + (1,)*(ndim-1),
                                meta=np.array([], dtype=np.float32)
                                )
        
        # For cupy, we must call map_blocks (or from_delayed and delayed) here.
        result = da.map_blocks(np.cumsum, result[..., 0].rechunk((len_t, ndim)), 
                               axis=0, meta=np.array([], dtype=np.float32)
                               )

        return result
    
    @_docs.copy_docs(ImgArray.drift_correction)
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    @dims_to_spatial_axes
    def drift_correction(self, shift: Coords = None, ref: ImgArray = None, *, 
                         zero_ave: bool = True, along: str = None, dims: Dims = 2, 
                         update: bool = False, **affine_kwargs) -> LazyImgArray:
        
        if along is None:
            along = find_first_appeared("tpzcia", include=self.axes, exclude=dims)
        elif len(along) != 1:
            raise ValueError("`along` must be single character.")
        
        from ..frame import MarkerFrame
        
        if shift is None:
            # determine 'ref'
            if ref is None:
                ref = self
                _dims = complement_axes(along, self.axes)
                if dims != _dims:
                    warn(f"dims={dims} with along={along} and {self.axes}-image are not "
                         f"valid input. Changed to dims={_dims}",
                         UserWarning)
                    dims = _dims
            elif not isinstance(ref, self.__class__):
                raise TypeError(f"'ref' must be LazyImgArray object, but got {type(ref)}")
            elif ref.axes != along + dims:
                raise ValueError(f"Arguments `along`({along}) + `dims`({dims}) do not match "
                                 f"axes of `ref`({ref.axes})")

            shift = ref.track_drift(along=along)
        elif isinstance(shift, MarkerFrame):
            if len(shift) != self.sizeof(along):
                raise ValueError("Wrong shape of 'shift'.")
            shift = shift.values
        from dask import array as da
        if zero_ave:
            shift = shift - da.mean(shift, axis=0)
            
        t_index = self.axisof(along)
        slice_in = switch_slice(dims, self.axes, ifin=slice(None), ifnot=0)
        slice_out = switch_slice(dims, self.axes, ifin=slice(None), ifnot=np.newaxis)
        ndim = len(dims)

        # Here shift must be a local variable for the function. Otherwise, it takes dask very long time 
        # for graph construction.
        def warp(arr, shift, block_info=None):
            arr = xp.asarray(arr)
            mx = xp.eye(ndim+1, dtype=np.float32)
            loc = block_info[None]["array-location"][0]
            mx[:-1, -1] = -xp.asarray(shift[loc[t_index]])
            return xp.asnumpy(
                _transform.warp(arr[slice_in], mx, **affine_kwargs)[slice_out]
                )
        
        chunks = switch_slice(dims, self.axes, ifin=self.shape, ifnot=1)
        
        out = da.map_blocks(warp, self.value.rechunk(chunks), shift, meta=np.array([], dtype=self.dtype))
        
        return out
    
    @_docs.copy_docs(ImgArray.pad)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def pad(self, pad_width, mode: str = "constant", *, dims: Dims = None, **kwargs) -> LazyImgArray:
        pad_width = _misc.make_pad(pad_width, dims, self.axes, **kwargs)
        padimg = np.pad(self.value, pad_width, mode, **kwargs)
        return padimg
    
    # @_docs.copy_docs(ImgArray.wiener)
    # @dims_to_spatial_axes
    # @same_dtype(asfloat=True)
    # @record_lazy
    # def wiener(self, psf: np.ndarray, lmd: float = 0.1, *, depth="auto", dims: Dims = None, update: bool = False) -> LazyImgArray:
    #     if lmd <= 0:
    #         raise ValueError(f"lmd must be positive, but got: {lmd}")
        
    #     if depth == "auto":
    #         depth = 32  # TODO: any better way?
        
    #     psf_ft, psf_ft_conj = _deconv.check_psf(self, psf, dims)
        
    #     return self._apply_map_overlap
    #     return self._apply_function(_deconv.wiener, 
    #                       c_axes=complement_axes(dims, self.axes),
    #                       rechunk_to="max",
    #                       args=(psf_ft, psf_ft_conj, lmd)
    #                       )
    
    # @_docs.copy_docs(ImgArray.lucy)
    # @dims_to_spatial_axes
    # @same_dtype(asfloat=True)
    # @record_lazy
    # def lucy(self, psf: np.ndarray, niter: int = 50, eps: float = 1e-5, depth: int = 32, *, dims: Dims = None, 
    #          update: bool = False) -> LazyImgArray:
    #     psf_ft, psf_ft_conj = _deconv.check_psf(self, psf, dims)

    #     return self._apply_map_overlap(_deconv.richardson_lucy, 
    #                       c_axes=complement_axes(dims, self.axes),
    #                       depth=depth,
    #                       boundary="nearest",
    #                       args=(psf_ft, psf_ft_conj, niter, eps)
    #                       )
    
    def __array_function__(self, func, types, args, kwargs):
        """
        Every time a numpy function (np.mean...) is called, this function will be called. Essentially numpy
        function can be overloaded with this method.
        """
        from dask import array as da
        args, kwargs = _replace_inputs(self, args, kwargs)
        
        _types = []
        for t in types:
            if t is self.__class__:
                _types.append(da.core.Array)
            else:
                _types.append(t)
        
        result = self.value.__array_function__(func, _types, args, kwargs)
        
        if result is NotImplemented:
            return NotImplemented
                
        if isinstance(result, (tuple, list)):
            out = []
            for r in result:
                if isinstance(r, da.core.Array):
                    out.append(
                        self.__class__(r)._process_output(self, args, kwargs)
                    )
                else:
                    out.append(r)
            out = DataList(out)
            
        elif isinstance(result, da.core.Array):
            out = self.__class__(result)
            out._process_output(self, args, kwargs)
            
        return out
    
    def _process_output(self, input: LazyImgArray, args: tuple, kwargs: dict):
        if "axis" in kwargs.keys():
            new_axes = input.axes.drop(kwargs["axis"])
        else:
            new_axes = self._INHERIT
        self._set_info(input, new_axes=new_axes)
        return None
        
        
    def as_uint8(self) -> LazyImgArray:
        img = self.value
        if img.dtype == np.uint8:
            return img
        
        if img.dtype == np.uint16:
            out = img / 256
        elif img.dtype.kind == "f":
            out = img + 0.5
            out = np.clip(out, 0, 255)
        else:
            raise TypeError(f"invalid data type: {img.dtype}")
        out = out.astype(np.uint8)
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    def as_uint16(self) -> LazyImgArray:
        img = self.value
        if img.dtype == np.uint16:
            return img
        if img.dtype == np.uint8:
            out = img * 256
        elif img.dtype == bool:
            out = img
        elif img.dtype.kind == "f":
            out = img + 0.5
            out = np.clip(out, 0, 65535)
        else:
            raise TypeError(f"invalid data type: {img.dtype}")
        out = out.astype(np.uint16)
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    def as_float(self) -> LazyImgArray:
        if self.dtype == np.float32:
            return self
        out = self.value.astype(np.float32)
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    def as_img_type(self, dtype=np.uint16) -> LazyImgArray:
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        elif dtype == "uint16":
            return self.as_uint16()
        elif dtype == "uint8":
            return self.as_uint8()
        elif dtype == "float32":
            return self.as_float()
        elif dtype == "float64":
            warn("Data type float64 is not valid for images. It was converted to float32 instead",
                 UserWarning)
            return self.as_float()
        elif dtype == "complex64":
            out = self.value.astype(np.complex64)
            out = self.__class__(out)
            out._set_info(self)
            return out
        elif dtype == "complex128":
            warn("Data type complex128 is not valid for images. It was converted to complex64 instead",
                 UserWarning)
            out = self.value.astype(np.complex64)
            out = self.__class__(out)
            out._set_info(self)
            return out
        elif dtype == "int8":
            out = self.value.astype(np.int8)
            out = self.__class__(out)
            out._set_info(self)
            return out
        else:
            raise ValueError(f"dtype: {dtype}")
    
    
    def _set_additional_props(self, other):
        # set additional properties
        # If `other` does not have it and `self` has, then the property will be inherited.
        for p in self.__class__.additional_props:
            setattr(self, p, getattr(other, p, 
                                     getattr(self, p, 
                                             None)))
    
    
    def _getitem_additional_set_info(self, other, **kwargs):
        self._set_info(other, kwargs["new_axes"])
        return None
    
    def _set_info(self, other: LazyImgArray, new_axes: Any = AxesMixin._INHERIT):
        self._set_additional_props(other)
        # set axes
        try:
            if new_axes is not self._INHERIT:
                self.axes = new_axes
            else:
                self.axes = other.axes.copy()
        except ImageAxesError:
            self.axes = None
        
        return None
    

def _replace_inputs(img: LazyImgArray, args, kwargs):
    _as_dask_array = lambda a: a.value if isinstance(a, LazyImgArray) else a
    # convert arguments
    args = tuple(_as_dask_array(a) for a in args)
    if "axis" in kwargs:
        axis = kwargs["axis"]
        if isinstance(axis, str):
            _axis = tuple(map(img.axisof, axis))
            if len(_axis) == 1:
                _axis = _axis[0]
            kwargs["axis"] = _axis
    
    if "out" in kwargs:
        kwargs["out"] = tuple(_as_dask_array(a) for a in kwargs["out"])
    
    return args, kwargs

def _ceilint(a: float):
    return int(np.ceil(a))

def iter_slice(shape, iteraxes: str, all_axes: str, exclude: str = ""):
    ndim = len(all_axes)
    iterlist = switch_slice(axes=iteraxes, 
                            all_axes=all_axes,
                            ifin=[range(s) for s in shape], 
                            ifnot=[(slice(None),)]*ndim)
    
    it = itertools.product(*iterlist)
    c = 0 # counter
    for sl in it:
        if len(exclude) == 0:
            outsl = sl
        else:
            outsl = tuple(s for i, s in enumerate(sl) 
                            if all_axes[i] not in exclude)
        yield outsl
        c += 1
        
    # if iterlist = []
    if c == 0:
        outsl = (slice(None),) * (len(all_axes) - len(exclude))
        yield outsl