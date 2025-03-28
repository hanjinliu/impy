from __future__ import annotations
import os
from pathlib import Path
import re
import glob
import itertools
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence
from typing_extensions import Literal, ParamSpec
import warnings
import numpy as np
from functools import wraps

from impy import io
from impy.utils import gauss
from impy.utils.slicer import solve_slicer
from impy._types import AxesTargetedSlicer, nDFloat
from impy.axes import ImageAxesError, broadcast, Axes, AxesLike, AxesTuple
from impy.collections import DataList
from impy.arrays.bases import MetaArray
from impy.arrays import ImgArray, LazyImgArray, Label, BigImgArray
from impy._const import Const

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike
    from impy.roi import RoiList
    from impy.io._utils import ImageMetadata
    # NOTE: "_ShapeLike" is not a public type in numpy.typing.
    from typing import SupportsIndex, Union
    ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

__all__ = [
    "array",
    "asarray",
    "aslabel",
    "aslazy",
    "asbigarray",
    "zeros",
    "empty",
    "ones",
    "full",
    "arange",
    "indices",
    "gaussian_kernel",
    "circular_mask",
    "imread",
    "imread_collection",
    "big_imread",
    "read_meta",
    "read_header",
    "roiread",
    "sample_image",
    "broadcast_arrays",
    "stack_relaxed",
]

# TODO:
# - ip.imread("...\$i$j.tif", key="i=2:"), ip.imread("...\*.tif", key="p=0") will raise error.


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


@_write_docs
def array(
    arr: ArrayLike,
    /,
    dtype: DTypeLike = None,
    *,
    copy: bool = True,
    name: str = None,
    axes: str = None,
    like: MetaArray | None = None,
) -> ImgArray:
    """Make an ImgArray object, like ``np.array(x)``

    Parameters
    ----------
    arr : array-like
        Base array.
    copy : bool, default is True
        If True, a copy of the original array is made.
    {}

    Returns
    -------
    ImgArray

    """
    if isinstance(arr, np.ndarray) and dtype is None:
        dtype = arr.dtype

    axes, name = _normalize_params(axes, name, like)

    if copy:
        _arr = np.array(arr, dtype=dtype)
    else:
        _arr = np.asarray(arr, dtype=dtype)

    # Automatically determine axes
    if axes is None:
        if isinstance(arr, (MetaArray, LazyImgArray)):
            axes = arr.axes
        else:
            axes = ["", "x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][_arr.ndim]

    self = ImgArray(_arr, name=name, axes=axes)

    return self

@_write_docs
def asarray(
    arr: ArrayLike,
    dtype: DTypeLike | None = None,
    *,
    name: str | None = None,
    axes: str | None = None,
    like: MetaArray | None = None,
) -> ImgArray:
    """Make an ImgArray object, like ``np.asarray(x)``

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
    return array(arr, dtype=dtype, name=name, axes=axes, copy=False, like=like)

@_write_docs
def aslabel(
    arr: ArrayLike,
    dtype: DTypeLike = None,
    *,
    name: str | None = None,
    axes: str | None = None,
    like: MetaArray | None = None,
) -> ImgArray:
    """
    Make an Label object.

    This function helps to create a Label object from an array. Dtype check is performed
    on array creation.

    Parameters
    ----------
    arr : array-like
        Base array.
    {}

    Returns
    -------
    Label

    """
    if isinstance(arr, np.ndarray) and dtype is None:
        if arr.dtype.kind == "u":
            dtype = arr.dtype
        elif arr.dtype.kind == "i":
            if arr.dtype in ("int8", "int16"):
                dtype = np.uint8
            elif arr.dtype == "int32":
                dtype = np.uint16
            elif arr.dtype == "int64":
                dtype = np.uint32
            else:
                dtype = np.uint64
        else:
            raise ValueError(f"Dtype {arr.dtype} is not supported for a Label.")

    axes, name = _normalize_params(axes, name, like)

    _arr = np.asarray(arr, dtype=dtype)

    # Automatically determine axes
    if axes is None:
        if isinstance(arr, (MetaArray, LazyImgArray)):
            axes = arr.axes
        else:
            axes = ["", "x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][_arr.ndim]

    self = Label(_arr, name=name, axes=axes)
    return self

def aslazy(*args, **kwargs) -> LazyImgArray:
    from impy.lazy import asarray as lazy_asarray

    warnings.warn(
        "impy.aslazy is deprecated. Please use impy.lazy.asarray instead.",
        DeprecationWarning,
    )
    return lazy_asarray(*args, **kwargs)

def asbigarray(
    arr: ArrayLike,
    dtype: DTypeLike = None,
    *,
    name: str | None = None,
    axes: str | None = None,
    like: MetaArray | None = None,
) -> BigImgArray:
    """
    Make an BigImgArray object from other types of array.

    Parameters
    ----------
    arr : array-like
        Base array.
    {}

    Returns
    -------
    BigImgArray

    """
    from dask import array as da
    from dask.array.core import Array as DaskArray

    if isinstance(arr, MetaArray):
        arr = da.from_array(arr.value)
    elif isinstance(arr, (np.ndarray, np.memmap)):
        arr = da.from_array(arr)
    elif isinstance(arr, BigImgArray):
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

    self = BigImgArray(arr, name=name, axes=axes)

    return self

_P = ParamSpec("_P")

def _inject_numpy_function(func: Callable[_P, Any | None]) -> Callable[_P, ImgArray]:
    npfunc: Callable = getattr(np, func.__name__)
    @wraps(func)
    def _func(*args, **kwargs):
        like = kwargs.pop("like", None)
        axes = kwargs.pop("axes", None)
        name = kwargs.pop("name", None)
        return asarray(npfunc(*args, **kwargs), name=name, axes=axes, like=like)

    _func.__doc__ = (
        f"""impy version of numpy.{func.__name__}. This function has additional
        parameters ``axes`` and ``name``. Original docstring follows.

        Additional Parameters
        ---------------------
        axes : str, optional
            Image axes. Must be same length as image dimension.
        name: str, optional
            Image name.
        like: MetaArray, optional
            Reference array from which name and axes will be copied.

        {npfunc.__doc__}
        """
        )
    _func.__annotations__.update({"return": ImgArray})
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

@_inject_numpy_function
def fromiter(iterable: Iterable, dtype: DTypeLike, count: int = -1): ...


def indices(
    dimensions: ShapeLike,
    dtype: DTypeLike = np.uint16,
    *,
    name: str | None = None,
    axes: AxesLike | None = None,
    like: MetaArray | None = None,
) -> AxesTuple[ImgArray]:
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
    tuple of ImgArray

    """
    out = tuple(
        asarray(ind, name=name, axes=axes, like=like) for ind in np.indices(dimensions, dtype=dtype)
    )
    return out[0].axes.tuple(out)


def gaussian_kernel(
    shape: ShapeLike,
    sigma: nDFloat = 1.0,
    peak: float = 1.0,
    *,
    name: str | None = None,
    axes: AxesLike | None = None,
) -> ImgArray:
    """
    Make an Gaussian kernel or Gaussian image.

    Parameters
    ----------
    shape : tuple of int
        Shape of image.
    sigma : float or array-like, default is 1.0
        Standard deviation of Gaussian.
    peak : float, default is 1.0
        Peak intensity.

    Returns
    -------
    ImgArray
        Gaussian image
    """
    if np.isscalar(sigma):
        sigma = (sigma,)*len(shape)
    g = gauss.GaussianParticle([(np.array(shape) - 1) / 2, sigma, peak, 0])
    ker = g.generate(shape)
    ker = array(ker, name=name or "Gaussian-Kernel", axes=axes)
    if axes is None and ker.ndim == 3:
        ker.axes = "zyx"
    return ker


def circular_mask(
    radius: nDFloat,
    shape: ShapeLike,
    center: Literal["center"] | tuple[float, ...] = "center",
    soft: bool = False,
    out_value: bool = True,
) -> ImgArray:
    """
    Make a circular or ellipsoid shaped mask. Region close to center will be filled with ``False``.

    Parameters
    ----------
    radius : float or array-like
        Radius of non-mask region
    shape : tuple
        Shape of mask.
    center : tuple or "center"
        Center of circle. By default circle locates at the center.

    Returns
    -------
    ImgArray
        Boolean image with zyx or yx axes
    """
    if center == "center":
        center = np.array(shape)/2. - 0.5
    elif len(shape) != len(center):
        raise ValueError("Length of `shape` and `center` must be same.")

    if np.isscalar(radius):
        radius = [radius] * len(shape)
    else:
        if len(radius) != len(shape):
            raise ValueError("Length of `shape` and `radius` must be same.")
        if soft:
            raise NotImplementedError("Soft mask is not implemented for ellipsoid mask.")

    x = np.indices(shape)
    s: np.ndarray = sum(((x0 - c0) / r0) ** 2 for x0, c0, r0 in zip(x, center, radius))
    axes = "zyx" if len(shape) == 3 else None  # change the default axes in `array`
    val = s > 1.0
    if soft:
        pix = 1 / radius[0]
        val = val.astype(np.float32)
        sl = (1.0 < s) & (s <= 1.0 + pix)
        val[sl] = (np.sqrt(s[sl]) - 1.0) / pix
        out = array(val, dtype=np.float32, axes=axes)
    else:
        out = array(val, dtype=bool, axes=axes)
    if not out_value:
        if soft:
            out[:] = 1.0 - out
        else:
            out[:] = ~out
    return out


def sample_image(name: str) -> ImgArray:
    """
    Get sample images from ``skimage`` and convert it into ImgArray.

    Parameters
    ----------
    name : str
        Name of sample image, such as "camera".

    Returns
    -------
    ImgArray
        Sample image.
    """
    from skimage import data as skdata
    img = getattr(skdata, name)()
    out = array(img, name=name)
    if out.shape[-1] == 3:
        out.axes = ["x", "yx", "zyx"][out.ndim-2] + "c"
        out = out.sort_axes()
    return out

def broadcast_arrays(*arrays: MetaArray) -> list[MetaArray]:
    """Broadcast input arrays to the same shape and axes"""
    axes_list: list[Axes] = []
    shapes: dict[str, int] = {}
    for arr in arrays:
        axes_list.append(arr.axes)
        for a, s in zip(arr.axes, arr.shape):
            shapes[a] = s
    axes = broadcast(*axes_list)
    shape = tuple(shapes[a] for a in axes)

    out = [a.broadcast_to(shape, axes) for a in arrays]
    return out

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Imread functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def imread(
    path: str,
    dtype: DTypeLike = None,
    key: AxesTargetedSlicer | None = None,
    *,
    name: str | None = None,
    squeeze: bool = False,
) -> ImgArray:
    """
    Load image(s) from a path. You can read list of images from directories with
    wildcards or ``"$"`` in ``path``.

    Parameters
    ----------
    path : str
        Path to the image or directory.
    dtype : Any type that np.dtype accepts
        Data type of images.
    key : AxesTargetedSlicer, optional
        If not None, image is read in a memory-mapped array first, and only
        ``img[key]`` is returned. Only axis-targeted slicing is supported. This
        argument is important when reading a large file.
    name : str, optional
        Name of array.
    squeeze : bool, default is False
        If True, redundant dimensions will be squeezed.

    Returns
    -------
    ImgArray
        Image data read from the file.

    Examples
    --------
    Read a part of an image

        >>> path = "path/to/image.mrc"
        >>> %time ip.imread(path)["x=:10;y=:10"]
        Wall time: 136 ms
        >>> %time ip.imread(path, key="x=:10;y=:10")
        Wall time: 3.01 ms

    """
    path = str(path)
    is_memmap = (key is not None)

    if "$" in path:
        return _imread_stack(path, dtype=dtype, key=key, squeeze=squeeze)
    elif "*" in path:
        return _imread_glob(path, dtype=dtype, key=key, squeeze=squeeze)
    elif not Path(path).exists():
        raise FileNotFoundError(f"No such file or directory: {path}")

    # read tif metadata
    if not is_memmap:
        size = Path(path).stat().st_size / 2**30
        if size > Const["MAX_GB"]:
            raise MemoryError(f"Too large {size:.2f} GB")

    image_data = io.imread(path, memmap=is_memmap)

    img = image_data.image
    axes = image_data.axes
    scale = image_data.scale
    scale_unit = image_data.unit
    metadata = image_data.metadata
    labels = image_data.labels

    if not isinstance(scale_unit, dict):
        scale_unit = {a: scale_unit for a in "zyx"}

    if is_memmap and axes is not None:
        sl = solve_slicer(key, Axes(axes), img.shape)
        axes = "".join(a for a, k in zip(axes, sl) if not isinstance(k, int))
        img = np.asarray(img[sl], dtype=dtype)

    self = ImgArray(img, name=name, axes=axes, source=path, metadata=metadata)

    if "c" in self.axes:
        if self.shape.c > self.shape.x:
            # In case the image is in yxc-order. This sometimes happens.
            self: ImgArray = np.moveaxis(self, -1, -3)
        if labels is not None:
            self.set_axis_label(c=labels)

    if dtype is None:
        dtype = self.dtype

    if squeeze:
        self = np.squeeze(self)

    # if key="y=0", ImageAxisError happens here because image loses y-axis. We have to set scale
    # one by one.
    if scale is not None:
        for k, v in scale.items():
            if k in self.axes:
                self.set_scale({k: v})
                if k in "zyx":
                    self.axes[k].unit = scale_unit[k]

    return self.sort_axes().as_img_type(dtype) # arrange in tzcyx-order

def _imread_glob(path: str, squeeze: bool = False, **kwargs) -> ImgArray:
    """
    Read images recursively from a directory, and stack them into one ImgArray.

    Parameters
    ----------
    path : str
        Path with wildcard.
    axis : str, default is "p"
        To specify which axis will be the new one.

    """
    path = str(path)
    paths = glob.glob(path, recursive=True)

    imgs = []
    for path in paths:
        img = imread(path, **kwargs)
        imgs.append(img)

    if len(imgs) == 0:
        raise RuntimeError("Could not read any images.")

    out: ImgArray = np.stack(imgs, axis="p")
    if squeeze:
        out = np.squeeze(out)
    try:
        base = os.path.split(path.split("*")[0])[0]
        out.source = base
    except Exception:
        pass

    return out

def _imread_stack(
    path: str,
    dtype: DTypeLike = None,
    key: AxesTargetedSlicer | None = None,
    squeeze: bool = False
):
    r"""
    Read separate image files using formated string. This function is useful when files/folders
    are named in a certain rule, such as ".../pos_0/img_0.tif", ".../pos_0/img_1.tif".

    Parameters
    ----------
    path : str
        Formated path string.
    dtype : Any type that np.dtype accepts
        dtype of the images.

    Returns
    -------
    ImgArray
        Image stack

    Example
    -------
    (1) For following file structure, read pos0, pos1, ... as p-stack.
        Base
        |- pos0.tif
        |- pos1.tif
        |- pos2.tif
        :
        >>> img = ip.imread_stack(r"C:\...\Base\pos$p.tif")

    (2) For following file structure, read xxx0, xxx1, ... as z-stack, and read yyy0, yyy1, ...
    as t-stack.
        Base
        |- xxx0
        |   |- yyy0.tif
        |   |- yyy1.tif
        |       :
        |- xxx1
        |   |- yyy0.tif
        |   |- yyy1.tif
        |       :
        :
        >>> img = ip.imread_stack(r"C:\...\Base\xxx$z\yyy$t.tif")
    """
    path = str(path)
    if "$" not in path:
        raise ValueError("`path` must contain '$' to specify variables in the string.")

    FORMAT = r"\$[a-z]"
    new_axes = list(map(lambda x: x[1:], re.findall(r"\$[a-z]", path)))

    # To convert input path string into that for glob.glob, replace $X with wildcard
    # e.g.) ~\XX$t_YY$z -> ~\XX*_YY*
    finder_path = re.sub(FORMAT, "*", path)

    # To convert input path string into file-finding regex pattern.
    # e.g.) ~\XX$t_YY$z\*.tif -> ~\\XX(\d)_YY(\d)\\.*\.tif
    path_ = repr(path)[1:-1]
    pattern = re.sub(r"\.", r"\.", path_)        # dots to non-escape
    pattern = re.sub(r"\*", ".*", pattern)       # asters to non-escape
    pattern = re.sub(FORMAT, r"(\\d+)", pattern) # make number finders
    pattern = re.compile(pattern)

    # To convert input path string into format string to execute imread.
    # e.g.) ~\XX$t_YY$z -> ~\XX{}_YY{}
    fpath = re.sub(FORMAT, "{}", path_)

    paths = glob.glob(finder_path)
    indices = [pattern.findall(p) for p in paths]
    ranges = [list(np.unique(ind)) for ind in np.array(indices).T]
    ranges_sorted = [[r[i] for i in np.argsort(list(map(int, r)))] for r in ranges]

    # read all the images
    img0 = None
    imgs = []
    for i in itertools.product(*ranges_sorted):
        # check if the image to read is unique
        found_paths = glob.glob(fpath.format(*i))
        n_found = len(found_paths)
        if n_found > 1:
            raise ValueError(f"{n_found} paths found at {fpath.format(*i)}.")
        elif n_found == 0:
            raise FileNotFoundError(f"No path found at {fpath.format(*i)}.")

        img = imread(found_paths[0], dtype=dtype, key=key)

        # To speed up error handling, check shape and axes here.
        if img0 is None:
            img0 = img
            for a in new_axes:
                if a in img0.axes:
                    raise ImageAxesError(f"{a} appeared twice.")
        else:
            if img.shape != img0.shape:
                raise ValueError(f"Shape mismatch at {fpath.format(*i)}. Make sure all "
                                  "the input images have exactly the same shapes.")

        imgs.append(img)

    # reshape image and set metadata
    new_shape = tuple(len(r) for r in ranges) + imgs[0].shape
    self = np.array(imgs, dtype=dtype).reshape(*new_shape).view(ImgArray)
    self._set_info(imgs[0])
    self.axes = "".join(new_axes) + str(img.axes)
    self.set_scale(imgs[0])
    # determine source
    name_list = []
    for p in path.split(os.sep):
        if "$" in p or "*" in p:
            break
        else:
            name_list.append(p)
    if len(name_list) > 0:
        self.source = os.path.join(*name_list)
    else:
        self.source = None

    if squeeze:
        self = np.squeeze(self)
    return self.sort_axes()


def imread_collection(
    path: str | list[str],
    filt: Callable[[np.ndarray], bool] | None = None,
) -> DataList[ImgArray]:
    """
    Open images as ImgArray and store them in DataList.

    Parameters
    ----------
    path : str or list of str
        Path than can be passed to ``glob.glob``. If a list of path is given, all the
        matched images will be read and concatenated into a DataList.
    filt : callable, optional
        If specified, only images that satisfies filt(img)==True will be stored in the
        returned DataList.

    Returns
    -------
    DataList
    """
    if isinstance(path, list):
        arrlist = DataList()
        for p in path:
            arrlist += imread_collection(p, filt=filt)
        return arrlist

    path = str(path)
    if os.path.isdir(path):
        path = os.path.join(path, "*.tif")
    paths = glob.glob(path, recursive=True)
    if filt is None:
        filt = lambda x: True
    arrlist = DataList()
    for path in paths:
        img = imread(path)
        if filt(img):
            arrlist.append(img)
    return arrlist

def big_imread(
    path: str,
    chunks="auto",
    *,
    name: str | None = None,
    squeeze: bool = False,
) -> BigImgArray:
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
    BigImgArray
    """
    from impy.lazy import imread as lazy_imread_

    out = lazy_imread_(path, chunks=chunks, name=name, squeeze=squeeze)
    return BigImgArray(out.value, out.name, out.axes, out.source, out.metadata)

def read_meta(path: str) -> dict[str, Any]:
    """
    Read the metadata of an image file.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    dict
        Dictionary of keys {"axes", "scale", "metadata"}
    """
    meta = read_header(path)
    return {
        "axes": meta.axes,
        "scale": meta.scale,
        "metadata": meta.metadata
    }

def read_header(path: str | Path) -> ImageMetadata:
    """
    Read the header of an image file.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    ImageMetadata
        Tuple of the metadata.
    """
    return io.read_header(path)

def roiread(path: str) -> RoiList:
    """Read a Roi.zip file as a RoiList object."""
    from .roi import RoiList

    return RoiList.fromfile(path)

def stack_relaxed(imgs: Sequence[ImgArray], axis=0) -> ImgArray:
    if len({img.ndim for img in imgs}) != 1:
        raise ValueError("All the arrays should have the same number of dimensions.")
    shapes = np.array([img.shape for img in imgs], dtype=int)
    shape_max = shapes.max(axis=0)
    arrays = []
    for img, shape in zip(imgs, shapes):
        arr = zeros(shape_max, dtype=img.dtype)
        sl = tuple(slice(0, s) for s in shape)
        arr[sl] = img
        arrays.append(arr)
    out = np.stack(arrays, axis=axis)
    return out
