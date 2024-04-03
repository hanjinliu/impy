from __future__ import annotations

import os
import re
import glob
from typing import TYPE_CHECKING, Any, Callable, Sequence
from typing_extensions import ParamSpec
import numpy as np
from functools import wraps

from impy import io
from impy.axes import AxesLike, AxesTuple
from impy.arrays.bases import MetaArray
from impy.arrays import LazyImgArray

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike
    # NOTE: "_ShapeLike" is not a public type in numpy.typing.
    from typing import SupportsIndex, Union
    ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

__all__ = [
    "array",
    "asarray",
    "zeros",
    "empty",
    "ones",
    "full",
    "arange",
    "indices",
    "imread",
]

_P = ParamSpec("_P")

shared_docs = \
    """
    dtype : data type, optional
        Image data type.
    name : str, optional
        Name of image.
    axes : str, optional
        Image axes.
    """

def _write_docs(func):
    """Add doc for numpy function."""
    func.__doc__ = re.sub(r"{}", shared_docs, func.__doc__)
    return func

def _inject_numpy_function(func: Callable[_P, Any | None]) -> Callable[_P, LazyImgArray]:

    numpy_func = getattr(np, func.__name__)
    @wraps(func)
    def _func(*args, **kwargs):
        from dask import array as da

        like = kwargs.pop("like", None)
        axes = kwargs.pop("axes", None)
        name = kwargs.pop("name", None)
        dafunc: Callable = getattr(da, func.__name__)
        return asarray(dafunc(*args, **kwargs), name=name, axes=axes, like=like)

    _func.__doc__ = (
        f"""
        impy.lazy version of numpy.{func.__name__}. This function has additional parameters
        ``axes`` and ``name``. Original docstring follows.

        Additional Parameters
        ---------------------
        axes : str, optional
            Image axes. Must be same length as image dimension.
        name: str, optional
            Image name.
        like: MetaArray, optional
            Reference array from which name and axes will be copied.

        {numpy_func.__doc__}
        """
        )
    _func.__annotations__.update({"return": LazyImgArray})
    return _func

@_inject_numpy_function
def zeros(shape: ShapeLike, dtype: DTypeLike = np.uint16, *, name: str | None = None, axes: AxesLike | None = None, like: MetaArray | None = None): ...

@_inject_numpy_function
def empty(shape: ShapeLike, dtype: DTypeLike = np.uint16, *, name: str | None = None, axes: AxesLike | None = None, like: MetaArray | None = None): ...

@_inject_numpy_function
def ones(shape: ShapeLike, dtype: DTypeLike = np.uint16, *, name: str | None = None, axes: AxesLike | None = None, like: MetaArray | None = None): ...

@_inject_numpy_function
def full(shape: ShapeLike, fill_value: Any, dtype: DTypeLike = np.uint16, *, name: str | None = None, axes: AxesLike | None = None, like: MetaArray | None = None): ...

@_inject_numpy_function
def arange(stop: int, dtype: DTypeLike = None): ...

@_write_docs
def array(
    arr: ArrayLike,
    /,
    dtype: DTypeLike = None,
    *,
    copy: bool = True,
    chunks="auto",
    name: str = None,
    axes: str = None,
    like: MetaArray | None = None,
) -> LazyImgArray:
    """
    make an LazyImgArray object, like ``np.array(x)``

    Parameters
    ----------
    arr : array-like
        Base array.
    copy : bool, default is True
        If True, a copy of the original array is made.
    {}

    Returns
    -------
    LazyImgArray
    """
    from dask import array as da
    from dask.array.core import Array as DaskArray

    if isinstance(arr, MetaArray):
        if axes is None:
            axes = arr.axes
        if name is None:
            name = arr.name
        arr = da.from_array(arr.value, chunks=chunks)
    elif isinstance(arr, (np.ndarray, np.memmap)):
        arr = da.from_array(arr, chunks=chunks)
    elif isinstance(arr, LazyImgArray):
        if axes is None:
            axes = arr.axes
        if name is None:
            name = arr.name
        arr = arr.value
    elif not isinstance(arr, DaskArray):
        arr = da.asarray(arr)

    if isinstance(arr, np.ndarray) and dtype is None:
        if arr.dtype in (np.uint8, np.uint16, np.float32):
            dtype = arr.dtype
        elif arr.dtype.kind == "f":
            dtype = np.float32
        else:
            dtype = arr.dtype

    axes, name = _normalize_params(axes, name, like)

    # Automatically determine axes
    if axes is None:
        axes = ["x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][arr.ndim-1]
    if copy:
        arr = arr.copy()
    self = LazyImgArray(arr, name=name, axes=axes)

    return self

@_write_docs
def asarray(
    arr: ArrayLike,
    dtype: DTypeLike | None = None,
    *,
    name: str | None = None,
    axes: str | None = None,
    like: MetaArray | None = None,
    chunks="auto",
) -> LazyImgArray:
    """
    make an LazyImgArray object, like ``np.asarray(x)``

    Parameters
    ----------
    arr : array-like
        Base array.
    {}
    copy : bool, default is True
        If True, a copy of the original array is made.

    Returns
    -------
    ImgArray

    """
    return array(arr, dtype=dtype, name=name, axes=axes, copy=False, chunks=chunks, like=like)

def indices(
    dimensions: ShapeLike,
    dtype: DTypeLike = np.uint16,
    *,
    name: str | None = None,
    axes: AxesLike | None = None,
    like: MetaArray | None = None,
) -> AxesTuple[LazyImgArray]:
    """
    Copy of ``numpy.indices``.

    Parameters
    ----------
    dimensions : shape-like
        The shape of the grid.
    dtype : dtype, optional
        Data type of the result.
    name : str, optional
        Name of the result arrays.
    axes : AxesLike, optional
        Axes of the result arrays.
    like : MetaArray, optional
        Reference array from which name and axes will be copied.

    Returns
    -------
    tuple of LazyImgArray
    """
    from dask import array as da
    out = tuple(
        asarray(ind, name=name, axes=axes, like=like) for ind in da.indices(dimensions, dtype=dtype)
    )
    return out[0].axes.tuple(out)

def imread(
    path: str,
    chunks="auto",
    *,
    name: str | None = None,
    squeeze: bool = False,
) -> LazyImgArray:
    """
    Read an image lazily. Image file is first opened as an memory map, and subsequently converted
    to `numpy.ndarray` or `cupy.ndarray` chunkwise by `dask.array.map_blocks`.

    Parameters
    ----------
    path : str
        Path to the file.
    chunks : optional
        Specify chunk sizes. By default, yx-axes are assigned to the same chunk for every slice of
        image, whild chunk sizes of the rest of axes are automatically set with "auto" option.
    name : str, optional
        Name of array.
    squeeze : bool, default is False
        If True and there is one-sized axis, then call `np.squeeze`.

    Returns
    -------
    LazyImgArray
    """
    path = str(path)
    if "*" in path:
        return _imread_glob(path, chunks=chunks, squeeze=squeeze)
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    # read as a dask array
    image_data = io.imread_dask(path, chunks)
    img = image_data.image
    axes = image_data.axes
    scale = image_data.scale
    spatial_scale_unit = image_data.unit
    metadata = image_data.metadata
    labels = image_data.labels

    if squeeze:
        axes = "".join(a for i, a in enumerate(axes) if img.shape[i] > 1)
        img = np.squeeze(img)

    self = LazyImgArray(img, name=name, axes=axes, source=path, metadata=metadata)

    # read scale if possible
    self.set_scale(**scale)

    if "c" in self.axes and labels is not None:
        self.set_axis_label(c=labels)

    for k, v in scale.items():
        if k in self.axes:
            self.set_scale({k: v})
            if k in "zyx":
                self.axes[k].unit = spatial_scale_unit.get(k)

    return self.sort_axes()


def _imread_glob(path: str, squeeze: bool = False, **kwargs) -> LazyImgArray:
    """
    Read images recursively from a directory, and stack them into one LazyImgArray.

    Parameters
    ----------
    path : str
        Path with wildcard.

    """
    path = str(path)
    paths = glob.glob(path, recursive=True)

    imgs: list[LazyImgArray] = []
    for path in paths:
        imgl = imread(path, **kwargs)
        imgs.append(imgl)

    if len(imgs) == 0:
        raise RuntimeError("Could not read any images.")

    from dask import array as da
    out = da.stack([i.value for i in imgs], axis=0)
    out = LazyImgArray(out)
    out._set_info(imgs[0], new_axes="p"+str(imgs[0].axes))

    if squeeze:
        axes = "".join(a for i, a in enumerate(out.axes) if out.shape[i] > 1)
        img = da.squeeze(out.value)
        out = LazyImgArray(img)
        out._set_info(imgs[0], new_axes=axes)
    try:
        out.source = os.path.split(path.split("*")[0])[0]
    except Exception:
        pass

    return out

def _normalize_params(
    axes: AxesLike | None = None,
    name: str | None = None,
    like: MetaArray | None = None
) -> tuple[AxesLike | None, str | None]:
    """Normalize input parameters for MetaArray construction."""
    if like is not None:
        if not isinstance(like, (MetaArray, LazyImgArray)):
            raise TypeError(
                f"'like' must be a MetaArray or a LazyImgArray, not {type(like)}"
            )
        name = name or like.name
        axes = axes or like.axes
    return axes, name
