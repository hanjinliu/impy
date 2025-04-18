from __future__ import annotations
from functools import lru_cache, wraps
import itertools
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING, Literal, Sequence
import numpy as np
from numpy.typing import DTypeLike
from warnings import warn
import tempfile
import operator

from .labeledarray import LabeledArray
from .imgarray import ImgArray
from .axesmixin import AxesMixin
from .tiled import TiledAccessor
from ._utils import _misc, _transform, _structures, _filters, _deconv, _corr, _docs

from impy.utils.axesop import slice_axes, switch_slice, complement_axes, find_first_appeared
from impy.utils.deco import check_input_and_output_lazy, dims_to_spatial_axes, same_dtype
from impy.utils.misc import check_nd
from impy.utils import slicer
from impy.io import imsave
from impy.collections import DataList

from impy.axes import ImageAxesError, AxisLike
from impy.array_api import xp
from impy._types import nDFloat, Coords, Iterable, Dims, PaddingMode
from impy._const import Const

if TYPE_CHECKING:
    from dask.array.core import Array as DaskArray
    from impy.axes import AxesTuple

class LazyImgArray(AxesMixin):
    additional_props = ["_source", "_metadata", "_name"]
    tiled: TiledAccessor[LazyImgArray] = TiledAccessor()

    def __init__(
        self,
        obj: DaskArray,
        name: str = None,
        axes: str = None,
        source: str = None,
        metadata: dict = None
    ):
        from dask.array.core import Array as DaskArray
        if not isinstance(obj, DaskArray):
            raise TypeError(f"The first input must be dask array, got {type(obj)}")
        self.value: DaskArray = obj
        self.source = source
        self._name = name
        self.axes = axes
        self._metadata = metadata or {}

    @property
    def source(self):
        """The source file path."""
        return self._source

    @source.setter
    def source(self, val):
        if val is None:
            self._source = None
        else:
            self._source = Path(val)

    @property
    def name(self) -> str:
        """Name of the array."""
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
        """Metadata dictionary of the array."""
        return self._metadata

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        return self.value.ndim

    @property
    def shape(self) -> AxesTuple[int]:
        """Shape of the array."""
        return self.axes.tuple(self.value.shape)

    @property
    def dtype(self):
        """Data type of the array"""
        return self.value.dtype

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        return self.value.size

    @property
    def itemsize(self):
        """Size of each element in bytes."""
        return self.value.itemsize

    @property
    def chunksize(self) -> AxesTuple[int]:
        """Chunk size of the array."""
        return self.axes.tuple(self.value.chunksize)

    @property
    def GB(self) -> float:
        """Return the array size in GB."""
        return self.value.nbytes / 2**30

    gb = GB  # alias

    def __array__(self, dtype=None, copy=True):
        # Should not be `self.compute` because in napari Viewer this function is called
        # every time sliders are moved.
        return xp.asnumpy(self.value.compute()).astype(dtype, copy=False)

    def __getitem__(self, key):
        key = slicer.solve_slicer(key, self.axes, self.shape)

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

    def _apply_operator(self, op: Callable[[Any, Any], Any], other: Any) -> LazyImgArray:
        if isinstance(other, self.__class__):
            out = op(self.value, other.value)
        else:
            out = op(self.value, other)
        out = self.__class__(out)
        out._set_info(self)
        return out

    @same_dtype(asfloat=True)
    def __add__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.add, other)

    @same_dtype(asfloat=True)
    def __iadd__(self, other) -> LazyImgArray:
        if isinstance(other, self.__class__):
            self.value += other.value
        else:
            self.value += other
        return self

    @same_dtype(asfloat=True)
    def __sub__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.sub, other)

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

    def __gt__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.gt, other)

    def __ge__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.ge, other)

    def __lt__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.lt, other)

    def __le__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.le, other)

    def __eq__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.eq, other)

    def __ne__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.ne, other)

    def __mod__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.mod, other)

    def __floordiv__(self, other) -> LazyImgArray:
        return self._apply_operator(operator.floordiv, other)

    def __pow__(self, power: float) -> LazyImgArray:
        return self._apply_operator(operator.pow, power)

    @property
    def chunk_info(self):
        chunk_info = ", ".join([f"{s}({o})" for s, o in zip(self.chunksize, self.axes)])
        return chunk_info

    def astype(self, dtype: DTypeLike, copy: bool = True) -> LazyImgArray:
        """Convert the array to the specified data type."""
        out = self.__class__(self.value.astype(dtype, copy=copy))
        out._set_info(self)
        return out

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
        Compute all the task and convert the result into ImgArray. If image size
        overwhelms MAX_GB then MemoryError is raised.
        """
        if self.gb > Const["MAX_GB"] and not ignore_limit:
            raise MemoryError(f"Too large: {self.gb:.2f} GB")
        arr: np.ndarray = self.value.compute()
        if arr.ndim > 0:
            img = xp.asnumpy(arr).view(ImgArray)
            for attr in ["_name", "_source", "axes", "_metadata"]:
                setattr(img, attr, getattr(self, attr, None))
        else:
            img = arr.item()
        return img

    def release(self, update: bool = True) -> LazyImgArray:
        """
        Compute all the tasks and store the data in memory map, and read it as a dask
        array again.
        """
        from dask import array as da

        with tempfile.NamedTemporaryFile() as ntf:
            mmap = np.memmap(ntf, mode="w+", shape=self.shape, dtype=self.dtype)
            da.store(self.value.map_blocks(xp.asnumpy), mmap, compute=True)
            mmap.flush()

        img = da.from_array(mmap, chunks=self.chunksize).map_blocks(
            xp.array, meta=xp.array([], dtype=self.dtype)
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
    ) -> DaskArray:
        from dask import array as da

        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        all_axes = str(self.axes)
        _func = _make_map_blocks_func(func, self.dtype, c_axes, all_axes)
        return da.map_blocks(_func, self.value, dtype=self.dtype, *args, **kwargs)

    def _apply_map_overlap(
        self,
        func: Callable,
        c_axes: str | None = None,
        depth: int = 16,
        boundary="none",
        dtype: DTypeLike = None,
        args: tuple | None = None,
        kwargs: dict[str, Any] | None = None
    ) -> DaskArray:
        from dask import array as da

        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if dtype is None:
            dtype = self.dtype
        all_axes = str(self.axes)
        _func = _make_map_overlap_func(func, dtype, c_axes, all_axes)
        depths = switch_slice(c_axes, self.axes, 0, depth)
        return da.map_overlap(_func, self.value, *args, depth=depths, dtype=dtype, **kwargs, boundary=boundary, align_arrays=False)

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

    @_docs.copy_docs(ImgArray.shift)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def shift(
        self,
        translation: Coords,
        *,
        order: int = 3,
        mode: PaddingMode = "constant",
        cval: float = 0,
        prefilter: bool | None = None,
        dims: Dims = 2,
        update: bool = False
    ) -> LazyImgArray:

        prefilter = prefilter or order > 1

        return self._apply_map_blocks(
            _transform.shift,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(shift=translation, order=order, mode=mode, cval=cval,
                        prefilter=prefilter)
        )

    @_docs.copy_docs(ImgArray.rotate)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def rotate(
        self,
        degree: float,
        center: Sequence[float] | Literal["center"] = "center",
        *,
        order: int = 3,
        mode: PaddingMode = "constant",
        cval: float = 0,
        dims: Dims = 2,
        update: bool = False
    ) -> LazyImgArray:
        if center == "center":
            center = np.array(self.sizesof(dims))/2. - 0.5
        else:
            center = np.asarray(center)

        translation_0 = _transform.compose_affine_matrix(translation=center)
        rotation = _transform.compose_affine_matrix(rotation=np.deg2rad(degree))
        translation_1 = _transform.compose_affine_matrix(translation=-center)

        mx = translation_0 @ rotation @ translation_1
        mx[-1, :] = [0] * len(dims) + [1]
        return self._apply_map_blocks(
            _transform.warp,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(matrix=mx, order=order, mode=mode, cval=cval),
        )

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


    @_docs.copy_docs(ImgArray.tophat)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def tophat(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_map_overlap(
            _filters.white_tophat,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            kwargs=dict(footprint=disk, mode=mode, cval=cval)
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

    @_docs.copy_docs(ImgArray.dog_filter)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def dog_filter(
        self,
        low_sigma: nDFloat = 1.0,
        high_sigma: nDFloat = 2.0,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        c_axes = complement_axes(dims, self.axes)
        depth = _ceilint(high_sigma*4)
        return self._apply_map_overlap(
            _filters.dog_filter,
            c_axes=c_axes,
            depth=depth,
            kwargs=dict(low_sigma=low_sigma, high_sigma=high_sigma),
        )

    @_docs.copy_docs(ImgArray.doh_filter)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def doh_filter(
        self,
        sigma: nDFloat = 1.0,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        c_axes = complement_axes(dims, self.axes)
        depth = _ceilint(sigma*4)
        return self._apply_map_overlap(
            _filters.doh_filter,
            c_axes=c_axes,
            depth=depth,
            kwargs=dict(sigma=sigma),
        )

    @_docs.copy_docs(ImgArray.log_filter)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def log_filter(
        self,
        sigma: nDFloat = 1.0,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        c_axes = complement_axes(dims, self.axes)
        depth = _ceilint(sigma*4)
        return -self._apply_map_overlap(
            _filters.gaussian_laplace,
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
    ) -> LazyImgArray:
        depth = order
        from functools import partial
        func = partial(
            _filters.spline_filter, order=order, output=np.float32, mode=mode
        )
        return self._apply_map_overlap(
            func,
            c_axes=complement_axes(dims, self.axes),
            depth=depth,
        )


    @_docs.copy_docs(ImgArray.median_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def median_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_map_overlap(
            xp.ndi.median_filter,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(footprint=disk, mode=mode, cval=cval),
        )

    @_docs.copy_docs(ImgArray.min_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def min_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_map_overlap(
            xp.ndi.minimum_filter,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(footprint=disk, mode=mode, cval=cval),
        )

    @_docs.copy_docs(ImgArray.max_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def max_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_map_overlap(
            xp.ndi.maximum_filter,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(footprint=disk, mode=mode, cval=cval),
        )

    @_docs.copy_docs(ImgArray.std_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def std_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_map_overlap(
            _filters.std_filter,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(selem=disk, mode=mode, cval=cval),
        )

    @_docs.copy_docs(ImgArray.coef_filter)
    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output_lazy
    def coef_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_map_overlap(
            _filters.coef_filter,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(selem=disk, mode=mode, cval=cval),
        )

    @_docs.copy_docs(ImgArray.mean_filter)
    @same_dtype(asfloat=True)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def mean_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        kernel = (disk / np.sum(disk)).astype(np.float32)
        return self._apply_map_overlap(
            xp.ndi.convolve,
            depth=_ceilint(radius),
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(weights=kernel, mode=mode, cval=cval),
        )

    @_docs.copy_docs(ImgArray.convolve)
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    def convolve(
        self,
        kernel,
        *,
        mode: PaddingMode = "reflect",
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
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    def edge_filter(
        self,
        method: str = "sobel",
        *,
        dims: Dims = None,
        update: bool = False
    ) -> LazyImgArray:
        from skimage.filters.edges import sobel, farid, scharr, prewitt

        method_dict = {
            "sobel": (sobel, 1), "farid": (farid, 2), "scharr": (scharr, 1),
            "prewitt": (prewitt, 1)
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
        from skimage.restoration import uft
        ndim = len(dims)
        _, laplace_op = uft.laplacian(ndim, (2*radius+1,) * ndim)
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
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
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

    @_docs.copy_docs(ImgArray.threshold)
    @check_input_and_output_lazy
    def threshold(
        self,
        thr: float | str,
        *,
        along: AxisLike | None = None,
    ) -> LazyImgArray:
        if self.dtype == bool:
            return self

        from dask import array as da

        if along is None:
            along = "c" if "c" in self.axes else ""

        if isinstance(thr, str) and thr.endswith("%"):
            p = float(thr[:-1].strip())
            out = da.zeros(self.shape, dtype=bool)
            for sl, img in self.iter(along):
                thr = da.percentile(img, p)
                out[sl] = img >= thr

        elif np.isscalar(thr) or (hasattr(thr, "compute") and thr.ndim == 0):
            out = self >= thr
        else:
            raise TypeError(
                "'thr' must be a scalar or string in 'X%' format."
            )
        return out

    @_docs.copy_docs(ImgArray.fft)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def fft(self, *, shape: int | Iterable[int] | str = "same", shift: bool = True,
            double_precision: bool = False, dims: Dims = None) -> LazyImgArray:
        from dask import array as da

        axes = [self.axisof(a) for a in dims]
        if shape == "square":
            s = 2**int(np.ceil(np.max(self.sizesof(dims))))
            shape = (s,) * len(dims)
        elif shape == "same":
            shape = None
        else:
            shape = check_nd(shape, len(dims))
        fft = _get_fft_func(Const["RESOURCE"])
        dtype = np.float64 if double_precision else np.float32
        freq = fft(self.value.astype(dtype), s=shape, axes=axes)
        if shift:
            freq[:] = da.fft.fftshift(freq, axes=axes)
        return freq

    @_docs.copy_docs(ImgArray.ifft)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def ifft(
        self,
        real: bool = True,
        *,
        shift: bool = True,
        double_precision: bool = False,
        dims: Dims = None,
    ) -> LazyImgArray:
        from dask import array as da

        dtype = np.complex128 if double_precision else np.complex64
        axes = [self.axisof(a) for a in dims]
        if shift:
            freq = da.fft.ifftshift(self.value, axes=axes)
        else:
            freq = self.value
        ifft = _get_ifft_func(Const["RESOURCE"])
        out = ifft(freq.astype(dtype), axes=axes)

        if real:
            out = da.real(out)

        return out

    @_docs.copy_docs(ImgArray.power_spectra)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def power_spectra(
        self,
        shape = "same",
        norm: bool = False,
        zero_norm: bool = False,
        *,
        double_precision: bool = False,
        dims: Dims = None,
    ) -> LazyImgArray:
        freq = self.fft(dims=dims, shape=shape, double_precision=double_precision)
        pw = freq.value.real ** 2 + freq.value.imag ** 2
        if norm:
            pw /= pw.max()
        if zero_norm:
            sl = switch_slice(dims, self.axes, ifin=np.array(self.shape)//2, ifnot=slice(None))
            pw[sl] = 0
        return pw

    def chunksizeof(self, axis:str):
        """Get the chunk size of the given axis"""
        return self.value.chunksize[self.axes.find(axis)]

    def chunksizesof(self, axes:str):
        """Get the chunk sizes of the given axes"""
        return tuple(self.chunksizeof(a) for a in axes)

    @_docs.copy_docs(ImgArray.transpose)
    def transpose(self, axes=None):
        if axes is None:
            _axes = None
            new_axes = self.axes[::-1]
        else:
            _axes = [self.axisof(a) for a in axes]
            new_axes = [self.axes[i] for i in list(axes)]
        out = self.__class__(self.value.transpose(_axes))
        out._set_info(self, new_axes=new_axes)
        return out

    @_docs.copy_docs(ImgArray.sort_axes)
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

    @_docs.copy_docs(ImgArray.proj)
    @same_dtype
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
    def binning(self, binsize: int = 2, method = "mean", *, check_edges: bool = True,
                dims: Dims = None) -> LazyImgArray:
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
        out.axes = self.axes.copy()  # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out

    @_docs.copy_docs(ImgArray.track_drift)
    def track_drift(self, along: str = None, upsample_factor: int = 10) -> DaskArray:
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
        result = da.map_overlap(
            pcc,
            img_fft,
            depth={0: (1, 0)},
            trim=False,
            boundary="none",
            chunks=(1, ndim) + (1,)*(ndim-1),
            meta=np.array([], dtype=np.float32)
        )

        # For cupy, we must call map_blocks (or from_delayed and delayed) here.
        result = da.map_blocks(
            np.cumsum,
            result[..., 0].rechunk((len_t, ndim)),
            axis=0,
            meta=np.array([], dtype=np.float32)
        )

        return result

    @_docs.copy_docs(ImgArray.drift_correction)
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    @dims_to_spatial_axes
    def drift_correction(
        self, shift: Coords | None = None, ref: ImgArray | Any | None = None, *,
        zero_ave: bool = True, along: str = None, dims: Dims = 2,
        update: bool = False, **affine_kwargs,
    ) -> LazyImgArray:
        if along is None:
            along = find_first_appeared("tpzcia", include=self.axes, exclude=dims)
        elif len(along) != 1:
            raise ValueError("`along` must be single character.")

        from impy.frame import MarkerFrame

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
                ref = self[ref]
            if ref.axes != along + dims:
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


    @_docs.copy_docs(ImgArray.radon)
    @dims_to_spatial_axes
    def radon(
        self,
        degrees: float | Iterable[float],
        *,
        central_axis: AxisLike | Sequence[float] | None = None,
        order: int = 3,
        dims: Dims = None,
    ) -> LazyImgArray:
        from dask import array as da
        from dask_image.ndinterp import affine_transform

        params, output_shape, squeeze = _transform.normalize_radon_input(
            self, dims, central_axis, degrees
        )

        # apply spline filter in advance.
        input = self.as_float().spline_filter(order=order)
        tasks: list[DaskArray] = [
            affine_transform(
                input.value, p, order=order, output_shape=output_shape, prefilter=False
            ).sum(axis=0)
            for p in params
        ]
        out = LazyImgArray(da.stack(tasks, axis=0))
        out._set_info(self, self.axes.drop(0).insert(0, "degree"))
        out.axes[0].labels = list(degrees)
        if squeeze:
            out = out[0]
        return out

    @_docs.copy_docs(ImgArray.iradon)
    def iradon(
        self,
        degrees: Sequence[float],
        *,
        central_axis: AxisLike = "y",
        degree_axis: AxisLike | None = None,
        height_axis: AxisLike | None = None,
        height: int | None = None,
        window: str = "hamming",
        order: int = 3,
    ) -> LazyImgArray:
        from dask import delayed, array as da

        interp = {0: "nearest", 1: "linear", 3: "cubic"}[order]
        central_axis, degree_axis, output_shape, new_axes = _transform.normalize_iradon_input(
            self, central_axis, height_axis, degree_axis, height
        )
        input = np.moveaxis(self.value, self.axisof(degree_axis), -1)
        filter_func = _transform.get_fourier_filter(input.shape[-2], window)

        func = delayed(_transform.iradon)
        if self.ndim == 3:
            arrays = [
                da.from_delayed(
                    func(
                        image_slice,
                        degrees=degrees,
                        interpolation=interp,
                        filter_func=filter_func,
                        output_shape=output_shape,
                    ),
                    shape=self.shape[1:],
                    dtype=self.dtype,
                )
                for image_slice in input
            ]

            out = da.stack(arrays, axis=0)
            out = np.moveaxis(out, 0, 1)
        else:
            out = da.from_delayed(
                func(
                    input,
                    degrees=degrees,
                    interpolation=interp,
                    filter_func=filter_func,
                    output_shape=output_shape,
                ),
                shape=output_shape,
                dtype=self.dtype,
            )
        out = out[::-1]
        out = self.__class__(out, axes=new_axes, name=self.name, source=self.source)
        out._set_info(self, new_axes=new_axes)
        return out

    @_docs.copy_docs(ImgArray.pad)
    @dims_to_spatial_axes
    @check_input_and_output_lazy
    def pad(
        self,
        pad_width: int | tuple[int, int] | Sequence[tuple[int, int]],
        mode: str = "constant",
        *,
        dims: Dims = None,
        **kwargs,
    ) -> LazyImgArray:
        import dask.array as da
        pad_width = _misc.make_pad(pad_width, dims, self.axes, **kwargs)
        padimg = da.pad(self.value, pad_width, mode, **kwargs)
        return padimg

    @_docs.copy_docs(ImgArray.wiener)
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    @dims_to_spatial_axes
    def wiener(
        self,
        psf: np.ndarray,
        lmd: float = 0.1,
        *,
        dims: Dims = None,
        update: bool = False,
    ) -> LazyImgArray:
        if lmd <= 0:
            raise ValueError(f"lmd must be positive, but got: {lmd}")

        psf_ft, psf_ft_conj = _deconv.check_psf(self, psf, dims)

        return self._apply_function(
            _deconv.wiener,
            c_axes=complement_axes(dims, self.axes),
            args=(psf_ft, psf_ft_conj, lmd)
        )

    @_docs.copy_docs(ImgArray.lucy)
    @same_dtype(asfloat=True)
    @check_input_and_output_lazy
    @dims_to_spatial_axes
    def lucy(
        self,
        psf: np.ndarray,
        niter: int = 50,
        eps: float = 1e-5,
        *,
        dims: Dims = None,
        update: bool = False,
    ) -> LazyImgArray:
        psf_ft, psf_ft_conj = _deconv.check_psf(self, psf, dims)

        return self._apply_function(
            _deconv.richardson_lucy,
            c_axes=complement_axes(dims, self.axes),
            args=(psf_ft, psf_ft_conj, niter, eps)
        )

    def __array_function__(self, func, types, args, kwargs):
        """
        Every time a numpy function (np.mean...) is called, this function will be called. Essentially numpy
        function can be overloaded with this method.
        """
        from dask.array.core import Array as DaskArray
        args, kwargs = _replace_inputs(self, args, kwargs)

        _types = []
        for t in types:
            if t is self.__class__:
                _types.append(DaskArray)
            else:
                _types.append(t)

        result = self.value.__array_function__(func, _types, args, kwargs)

        if result is NotImplemented:
            return NotImplemented

        if isinstance(result, (tuple, list)):
            out = []
            for r in result:
                if isinstance(r, DaskArray):
                    out.append(
                        self.__class__(r)._process_output(self, args, kwargs)
                    )
                else:
                    out.append(r)
            out = DataList(out)

        elif isinstance(result, DaskArray):
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

    def as_float(self,  *, depth: int = 32) -> LazyImgArray:
        if depth == 16:
            dtype = np.float16
        elif depth == 32:
            dtype = np.float32
        elif depth == 64:
            dtype = np.float64
        else:
            raise ValueError(f"depth must be 16, 32, or 64, but got {depth}")
        if self.dtype == dtype:
            return self
        out = self.value.astype(dtype)
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
        elif dtype == "float16":
            return self.as_float(depth=16)
        elif dtype == "float32":
            return self.as_float(depth=32)
        elif dtype == "float64":
            return self.as_float(depth=64)
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
        elif dtype in ("int8", "int16", "int32", "uint32"):
            out = self.value.astype(dtype)
            out = self.__class__(out)
            out._set_info(self)
            return out
        else:
            raise ValueError(f"dtype: {dtype}")

    @_docs.copy_docs(ImgArray.mean)
    def mean(self, axis=None, keepdims=False, **kwargs):
        return np.mean(self, axis=axis, keepdims=keepdims, **kwargs)

    @_docs.copy_docs(ImgArray.std)
    def std(self, axis=None, keepdims=False, **kwargs):
        return np.std(self, axis=axis, keepdims=keepdims, **kwargs)

    @_docs.copy_docs(ImgArray.sum)
    def sum(self, axis=None, keepdims=False, **kwargs):
        return np.sum(self, axis=axis, keepdims=keepdims, **kwargs)

    @_docs.copy_docs(ImgArray.max)
    def max(self, axis=None, keepdims=False, **kwargs):
        return np.max(self, axis=axis, keepdims=keepdims, **kwargs)

    @_docs.copy_docs(ImgArray.min)
    def min(self, axis=None, keepdims=False, **kwargs):
        return np.min(self, axis=axis, keepdims=keepdims, **kwargs)

    def _set_additional_props(self, other):
        # set additional properties
        # If `other` does not have it and `self` has, then the property will be inherited.
        for p in self.__class__.additional_props:
            setattr(self, p, getattr(other, p, getattr(self, p, None)))

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

        return self


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

# NOTE: following FFT functions needs the "resource" argument to make it work with both
# numpy and cupy.
@lru_cache
def _get_fft_func(resource) -> Callable[[DaskArray], DaskArray]:
    """Get the scipy FFT function for dask."""
    from impy.array_api import scipy_fft
    from dask import array as da

    return da.fft.fft_wrap(scipy_fft.fftn)

@lru_cache
def _get_ifft_func(resource) -> Callable[[DaskArray], DaskArray]:
    """Get the scipy IFFT function for dask."""
    from impy.array_api import scipy_fft
    from dask import array as da

    return da.fft.fft_wrap(scipy_fft.ifftn)

def _make_map_blocks_func(func, dtype, c_axes, all_axes):
    def _func(input: np.ndarray, *args, **kwargs):
        out = xp.empty(input.shape, dtype)
        for sl in iter_slice(input.shape, c_axes, all_axes):
            out[sl] = func(input[sl], *args, **kwargs)
        return out
    return _func

def _make_map_overlap_func(func, dtype, c_axes, all_axes):
    def _func(input: np.ndarray, *args, **kwargs):
        out = xp.empty(input.shape, dtype=dtype)
        for sl in iter_slice(input.shape, c_axes, all_axes):
            out[sl] = func(input[sl], *args, **kwargs).astype(dtype, copy=False)
        return out
    return _func
