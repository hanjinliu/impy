from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Hashable, Union, SupportsInt, Mapping, Any
from pathlib import Path
import numpy as np
from numpy.typing import DTypeLike
from impy._types import Callable, Slices
from impy.axes import ImageAxesError, AxesLike, Axes, AxisLike
from impy.array_api import xp
from impy.utils import axesop, slicer
from impy.collections import DataList
from impy.arrays.axesmixin import AxesMixin

if TYPE_CHECKING:
    from typing_extensions import Self

try:
    _NoValue = np._NoValue
except AttributeError:
    _NoValue = None  # simply avoid errors.

SupportOneSlicing = Union[SupportsInt, slice]
SupportSlicing = Union[
    SupportsInt,
    str,
    slice,
    tuple[SupportOneSlicing, ...], 
    Mapping[str, SupportOneSlicing],
]

class MetaArray(AxesMixin, np.ndarray):
    additional_props = ["_source", "_metadata", "_name"]
    NP_DISPATCH = {}
    _name: str
    _source: Path | None
    _metadata: dict[str, Any]
    
    def __new__(
        cls: type[MetaArray], 
        obj,
        name: str | None = None,
        axes: Iterable[Hashable] | None = None,
        source: str | Path | None = None, 
        metadata: dict[str, Any] | None = None,
        dtype: DTypeLike = None,
    ) -> Self:
        if isinstance(obj, cls):
            return obj
        
        self = np.asarray(obj, dtype=dtype).view(cls)
        self.source = source
        self._name = name
        self.axes = axes
        self._metadata = metadata or {}
        return self
    
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

    @metadata.setter
    def metadata(self, value):
        if not isinstance(value, dict):
            raise TypeError(f"Cannot set {type(value)} as a metadata.")
        self._metadata = value
    
    @property
    def value(self) -> np.ndarray:
        """Numpy view of the array."""
        return np.asarray(self)
    
    def __repr__(self) -> str:
        if self.ndim > 0:
            return super().__repr__()
        return self.value.item()

    def _repr_dict_(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape_info,
            "dtype": self.dtype,
            "source": self.source,
            "scale": self.scale,
        }
    
    def __str__(self):
        return f"{self.__class__.__name__}<{self.name!r}>"
    
    @property
    def shape(self):
        return self.axes.tuple(super().shape)
    
    def __getitem__(self, key: SupportSlicing) -> Self:
        key = slicer.solve_slicer(key, self.axes)

        if isinstance(key, np.ndarray):
            key = self._broadcast(key)
        
        out = super().__getitem__(key)  # get item as np.ndarray
        
        if isinstance(out, self.__class__):  # cannot set attribution to such as numpy.int32 
            new_axes = axesop.slice_axes(self.axes, key)
            out._getitem_additional_set_info(self, new_axes=new_axes, key=key)

        return out
    
    def __setitem__(self, key: SupportSlicing, value):
        key = slicer.solve_slicer(key, self.axes)
        
        if isinstance(key, MetaArray) and key.dtype == bool:
            key = axesop.add_axes(self.axes, self.shape, key, key.axes)
            
        elif isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
            # img[arr] ... where arr is 2-D boolean array
            key = axesop.add_axes(self.axes, self.shape, key)

        super().__setitem__(key, value)
    
    def sel(self, indexer=None, /, **kwargs: dict[str, Any]) -> Self:
        """
        A label based indexing method, mimicking ``xarray.sel``.
        
        Example
        -------
        >>> img.sel(c="Red")
        >>> img.sel(t=slice("frame 3", "frame, 5"))
        """
        if indexer is not None:
            kwargs.update(indexer)
        axes = self.axes
        slices = [slice(None)] * self.ndim
        for k, v in kwargs.items():
            idx = axes.find(k)
            axis = axes[idx]
            if lbl := axis.labels:
                if isinstance(v, list):
                    slices[idx] = [lbl.index(each) for each in v]
                else:
                    slices[idx] = lbl.get_slice(v)
            else:
                raise ValueError(f"Cannot select {k} because it has no labels.")
        return self[tuple(slices)]
    
    def isel(self, indexer=None, /, **kwargs: dict[str, Any]) -> Self:
        """
        A index based indexing method, mimicking ``xarray.isel``.
        
        Example
        -------
        >>> img.isel(c=3)
        >>> img.isel(t=slice(4, 7))
        """
        if indexer is not None:
            kwargs.update(indexer)

        key = slicer.solve_slicer(kwargs, self.axes)
        out = super().__getitem__(key)  # get item as np.ndarray
        
        if isinstance(out, self.__class__):  # cannot set attribution to such as numpy.int32 
            new_axes = axesop.slice_axes(self.axes, key)
            out._getitem_additional_set_info(self, new_axes=new_axes, key=key)
        return out
    
    def __array_finalize__(self, obj):
        """
        Every time an np.ndarray object is made by numpy functions inherited to ImgArray,
        this function will be called to set essential attributes. Therefore, you can use
        such as img.copy() and img.astype("int") without problems (maybe...).
        """
        if obj is None: return None
        self._set_additional_props(obj)

        try:
            self.axes = getattr(obj, "axes", None)
        except Exception:
            self.axes = None
        else:
            if len(self.axes) != self.ndim:
                self.axes = None
        
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Every time a numpy universal function (add, subtract, ...) is called,
        this function will be called to set/update essential attributes.
        """
        args_, _ = _replace_inputs(self, args, kwargs)

        result = getattr(ufunc, method)(*args_, **kwargs)

        if result is NotImplemented:
            return NotImplemented
        
        result = result.view(self.__class__)
        
        # in the case result is such as np.float64
        if not isinstance(result, self.__class__):
            return result
        
        result._process_output(ufunc, args, kwargs)
        return result
    
    def __array_function__(self, func, types, args, kwargs):
        """
        Every time a numpy function (np.mean...) is called, this function will be called. Essentially numpy
        function can be overloaded with this method.
        """
        if (func in self.__class__.NP_DISPATCH and 
            all(issubclass(t, MetaArray) for t in types)):
            return self.__class__.NP_DISPATCH[func](*args, **kwargs)
        
        args_, _ = _replace_inputs(self, args, kwargs)

        result = func(*args_, **kwargs)

        if result is NotImplemented:
            return NotImplemented
        
        if isinstance(result, (tuple, list)):
            _as_meta_array = lambda a: a.view(self.__class__)._process_output(func, args, kwargs) \
                if type(a) is np.ndarray else a
            result = DataList(_as_meta_array(r) for r in result)
            
        else:
            if isinstance(result, np.ndarray):
                result = result.view(self.__class__)
            # in the case result is such as np.float64
            if isinstance(result, self.__class__):
                result._process_output(func, args, kwargs)
        
        return result

    @classmethod
    def implements(cls, numpy_function):
        """
        Add functions to NP_DISPATCH so that numpy functions can be overloaded.
        """        
        def decorator(func):
            cls.NP_DISPATCH[numpy_function] = func
            func.__name__ = numpy_function.__name__
            return func
        return decorator
    
    def sort_axes(self) -> Self:
        """
        Sort image dimensions to ptzcyx-order

        Returns
        -------
        MetaArray
            Sorted image
        """
        order = self.axes.argsort()
        return self.transpose(order)

    def argmax_nd(self) -> tuple[int, ...]:
        """
        N-dimensional version of argmax.
        
        For instance, if yx-array takes its maximum at (5, 8), this function returns
        ``AxesShape(y=5, x=8)``.

        Returns
        -------
        AxesShape
            Argmax of the array.
        """
        argmax = np.unravel_index(np.argmax(self), self.shape)
        return self.axes.tuple(argmax)
    
    def split(self, axis=None) -> DataList[Self]:
        """
        Split n-dimensional image into (n-1)-dimensional images.

        Parameters
        ----------
        axis : str or int, optional
            Along which axis the original image will be split, by default "c"

        Returns
        -------
        list of arrays
            Separate images
        """
        # determine axis in int.
        if axis is None:
            axis = axesop.find_first_appeared(self.axes, include="cztp")
        axisint = self.axisof(axis)
            
        imgs: DataList[MetaArray] = DataList(np.moveaxis(self, axisint, 0))
        for img in imgs:
            img.axes = self.axes.drop(axisint)
            img.set_scale(self)
            
        return imgs
    
    def _apply_dask(
        self, 
        func: Callable,
        c_axes: str | None = None,
        drop_axis: Iterable[int] = [], 
        new_axis: Iterable[int] = None, 
        dtype = np.float32, 
        out_chunks: tuple[int, ...] = None,
        args: tuple[Any] = None,
        kwargs: dict[str, Any] = None
    ) -> Self:
        """
        Convert array into dask array and run a batch process in parallel. In many cases batch process 
        in this way is faster than `multiprocess` module.

        Parameters
        ----------
        func : callable
            Function to apply.
        c_axes : str, optional
            Axes to iterate.
        drop_axis : Iterable[int], optional
            Passed to map_blocks.
        new_axis : Iterable[int], optional
            Passed to map_blocks.
        dtype : any that can be converted to np.dtype object, default is np.float32
            Output data type.
        out_chunks : tuple of int, optional
            Output chunks. This argument is important when the output shape will change.
        args : tuple, optional
            Arguments that will passed to `func`.
        kwargs : dict
            Keyword arguments that will passed to `func`.

        Returns
        -------
        MetaArray
            Processed array.
        """        
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()
        
        if len(c_axes) == 0:
            # Do not construct dask tasks if it is not needed.
            out = xp.asnumpy(func(self.value, *args, **kwargs), dtype=dtype)
        else:
            from dask import array as da
            new_axis = _list_of_axes(self, new_axis)
            drop_axis = _list_of_axes(self, drop_axis)
                
            # determine chunk size and slices
            chunks = axesop.switch_slice(c_axes, self.axes, ifin=1, ifnot=self.shape)
            slice_in = []
            slice_out = []
            for i, a in enumerate(self.axes):
                if a in c_axes:
                    slice_in.append(0)
                    slice_out.append(np.newaxis)
                else:
                    slice_in.append(slice(None))
                    slice_out.append(slice(None))
                
                if i in drop_axis:
                    slice_out.pop(-1)
                if i in new_axis:
                    slice_in.append(np.newaxis)
                    
            slice_in = tuple(slice_in)
            slice_out = tuple(slice_out)

            all_args = (self.value,) + args
            img_idx = []
            _args = []
            for i, arg in enumerate(all_args):
                if isinstance(arg, (np.ndarray, xp.ndarray)) and arg.shape == self.shape:
                    _args.append(da.from_array(arg, chunks=chunks))
                    img_idx.append(i)
                else:
                    _args.append(arg)
                    
            def _func(*args, **kwargs):
                args = list(args)
                for i in img_idx:
                    if args[i].ndim < len(slice_in):
                        continue
                    args[i] = args[i][slice_in]
                out = func(*args, **kwargs)
                return xp.asnumpy(out[slice_out])
            
            out = da.map_blocks(
                _func, 
                *_args, 
                drop_axis=drop_axis,
                new_axis=new_axis, 
                meta=xp.array([], dtype=dtype), 
                chunks=out_chunks,
                **kwargs
            )
            
            out = xp.asnumpy(out.compute())
            
        out = out.view(self.__class__)
        
        return out
    
    def transpose(self, axes=None) -> Self:
        """
        change the order of image dimensions.
        'axes' will also be arranged.
        """
        if axes is None:
            _axes = None
            new_axes = self.axes[::-1]
        else:
            _axes = [self.axisof(a) for a in axes]
            new_axes = [self.axes[i] for i in list(axes)]
        out: np.ndarray = np.transpose(self.value, _axes)
        out = out.view(self.__class__)
        out._set_info(self, new_axes=new_axes)
        return out
    
    def reshape(self, *shape, order="C", axes: AxesLike | None = None) -> Self:
        out: MetaArray = super().reshape(*shape, order=order)
        if axes:
            out.axes = axes
        return out
    
    @property
    def T(self) -> Self:
        out = super().T
        out.axes = out.axes[::-1]
        return out
    
    def _broadcast(self, value: Any):
        """Broadcasting method used in most of the mathematical operations."""
        if not isinstance(value, MetaArray):
            return value
        current_axes = self.axes
        if (current_axes == value.axes 
            or current_axes.has_undef() or
            value.axes.has_undef()):
            # In most cases arrays don't need broadcasting. Check axes first to
            # avoid spending time on broadcasting.
            return value
        value = value.broadcast_to(self.shape, current_axes)
        return value
    
    def broadcast_to(
        self, 
        shape: tuple[int, ...], 
        axes: AxesLike | None = None,
    ) -> Self:
        """
        Broadcast array to specified shape and axes.

        Parameters
        ----------
        shape : shape-like
            Shape of output array.
        axes : AxesLike, optional
            Axes of output array. If given, it must match the dimensionality of
            input shape.

        Returns
        -------
        MetaArray
            Broadcasted array.
        """
        if axes is None:
            return np.broadcast_to(self, shape)
        elif len(shape) != len(axes):
            raise ValueError(f"Dimensionality mismatch: {shape=} and {axes=}")
        current_axes = self.axes
        if self.shape == shape and current_axes == axes:
            return self
        if any(a not in axes for a in current_axes):
            ax0 = [str(a) for a in current_axes]
            ax1 = [str(a) for a in axes]
            raise ImageAxesError(
                f"Cannot broadcast array with axes {ax0} to {ax1}."
            )

        out = self.value
        for i, axis in enumerate(axes):
            if axis not in current_axes:
                out = np.stack([out] * shape[i], axis=i)

        out = out.view(self.__class__)

        if out.shape != shape:
            raise ValueError(
                f"Shape {shape} required but returned {out.shape}."
            )
        
        if not isinstance(axes, Axes):
            new_axes = Axes(axes)
            for a in self.axes:
                # update axis metadata such as scale
                new_axes.replace(str(a), a)
        else:
            new_axes = axes
        out._set_info(self, new_axes=new_axes)
        return out
    
    def min(
        self,
        axis=None,
        out: None = None,
        keepdims: bool = _NoValue,
        *,
        where: np.ndarray = _NoValue,
    ):
        """Minimum value of the array along a given axis."""
        return np.min(self, axis=axis, out=out, keepdims=keepdims, where=where)
    
    def max(
        self,
        axis=None,
        out: None = None,
        keepdims: bool = _NoValue,
        *,
        where: np.ndarray = _NoValue,
    ):
        """Maximum value of the array along a given axis."""
        return np.max(self, axis=axis, out=out, keepdims=keepdims, where=where)
    
    def mean(
        self,
        axis=None,
        dtype: DTypeLike = None,
        out: None = None,
        keepdims: bool = _NoValue,
        *,
        where: np.ndarray = _NoValue,
    ):
        """Mean value of the array along a given axis."""
        return np.mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
    
    def sum(
        self,
        axis=None,
        dtype: DTypeLike = None,
        out: None = None,
        keepdims: bool = _NoValue,
        *,
        where: np.ndarray = _NoValue,
    ):
        """Sum value of the array along a given axis."""
        return np.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
    
    def std(
        self,
        axis=None,
        dtype: DTypeLike = None,
        out: None = None,
        ddof: int = 0,
        keepdims: bool = _NoValue,
        *,
        where: np.ndarray = _NoValue,
    ):
        """Standard deviation of the array along a given axis."""
        return np.std(self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)

    def _dimension_matches(self, array: MetaArray):
        """Check if dimension satisfies ``self <: array``."""
        img_shape = array.shape
        label_shape = self.shape
        return all(
            [getattr(img_shape, str(a), _NOTME) == getattr(label_shape, str(a), _NOTME)
            for a in self.axes]
        )
        
    def as_rgba(
        self, 
        cmap: str | Callable[[np.ndarray], np.ndarray],
        *,
        axis: AxisLike = "c",
        clim: tuple[float, float] | None = None,
        alpha: np.ndarray | None = None,
    ) -> Self:
        """
        Convert array to an RGBA image with given colormap.

        Parameters
        ----------
        cmap : str or callable
            Colormap. Can be a string name of a colormap registered in vispy.
        axis : AxisLike, default is "c"
            The axis name used for the color axis.
        clim : (float, float), optional
            Contrast limits. If not given, the minimum and maximum values of the
            array will be used.

        Returns
        -------
        MetaArray
            Colored image.
        """
        new_axes = self.axes + axis
        if isinstance(cmap, str):
            from vispy.color import get_colormap
            
            vispy_cmap = get_colormap(cmap)
            
            def cmap(arr):
                out = vispy_cmap.map(arr)
                return out.reshape(self.shape + (4,))
            
        if clim is None:
            clim = self.min(), self.max()
            
        low, high = clim
        input = (np.clip(self.value, low, high) - low) / (high - low)
        from impy import asarray
        output = asarray(cmap(input), axes=new_axes, like=self)
        
        if alpha is not None:
            output["c=3"] = np.clip(alpha / alpha.max(), 0, 1)
        return output
        
    def __add__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__add__(value)
    
    def __sub__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__sub__(value)
    
    def __mul__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__mul__(value)
    
    def __truediv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__truediv__(value)
    
    def __mod__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__mod__(value)
    
    def __floordiv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__floordiv__(value)
    
    def __gt__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__gt__(value)
    
    def __ge__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ge__(value)
    
    def __lt__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__lt__(value)
    
    def __le__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__le__(value)
    
    def __eq__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__eq__(value)
    
    def __ne__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ne__(value)
    
    def __and__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__and__(value)
    
    def __or__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__or__(value)
    
    def __ne__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ne__(value)
    
    def __iadd__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__iadd__(value)
    
    def __isub__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__isub__(value)
    
    def __imul__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__imul__(value)
    
    def __itruediv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__itruediv__(value)
    
    def __imod__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__imod__(value)
    
    def __ifloordiv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ifloordiv__(value)
    
    
    def _set_additional_props(self, other):
        # set additional properties
        # If `other` does not have it and `self` has, then the property will be inherited.
        for p in self.__class__.additional_props:
            setattr(self, p, getattr(other, p, 
                                     getattr(self, p, 
                                             None)))
    
    def _set_info(self, other: Self, new_axes: Any= AxesMixin._INHERIT):
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
    
    def _process_output(self, func, args, kwargs):
        # find the largest MetaArray. Largest because of broadcasting.
        arr = None
        for arg in args:
            if isinstance(arg, self.__class__):
                if arr is None or arr.ndim < arg.ndim:
                    arr = arg
                    
        if isinstance(arr, self.__class__):
            self._inherit_meta(arr, func, **kwargs)
        
        return self
    
    def _getitem_additional_set_info(self, other: Self, key: Slices, new_axes):
        return self._set_info(other, new_axes=new_axes)
    
    def _inherit_meta(self, obj: AxesMixin, ufunc, **kwargs):
        """
        Copy axis etc. from obj.
        This is called in __array_ufunc__(). Unlike _set_info(), keyword `axis` must be
        considered because it changes `ndim`.
        """
        if "axis" in kwargs and "keepdims" not in kwargs:
            new_axes = obj.axes.drop(kwargs["axis"])
        else:
            new_axes = self._INHERIT
        self._set_info(obj, new_axes=new_axes)
        return self

    if TYPE_CHECKING:
        def astype(self, dtype) -> Self: ...
        def flatten(self, order="C") -> Self: ...
        def ravel(self, order="C") -> Self: ...

def _list_of_axes(img: MetaArray, axis):
    if axis is None:
        axis = []
    elif hasattr(axis, "__iter__"):
        axis = [img.axisof(a) for a in axis]
    elif np.isscalar(axis):
        axis = [axis]
    return axis
        
def _replace_inputs(img: MetaArray, args: tuple[Any], kwargs: dict[str, Any]):
    _as_np_ndarray = lambda a: a.value if isinstance(a, MetaArray) else a
    # convert arguments
    args = tuple(_as_np_ndarray(a) for a in args)
    if kwargs.get("axis", None) is not None:
        axis = kwargs["axis"]
        if not hasattr(axis, "__iter__"):
            axis = [axis]
        kwargs["axis"] = tuple(map(img.axisof, axis))
    
    if kwargs.get("axes", None) is not None:
        # used in such as np.rot90
        axes = kwargs["axes"]
        kwargs["axes"] = tuple(map(img.axisof, axes))
                
    if kwargs.get("out", None) is not None:
        kwargs["out"] = tuple(_as_np_ndarray(a) for a in kwargs["out"])
    
    return args, kwargs


class NotMe:
    def __eq__(self, other):
        return False

_NOTME = NotMe()
