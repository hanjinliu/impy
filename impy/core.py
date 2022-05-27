from __future__ import annotations
import os
import sys
import re
import glob
import itertools
if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, _ShapeLike
from functools import wraps

from . import io
from .utils import gauss
from .utils.slicer import *
from ._types import *

from .axes import ImageAxesError, broadcast, Axes
from .collections import DataList
from .arrays.bases import MetaArray
from .arrays import ImgArray, LazyImgArray
from ._const import Const

__all__ = [
    "array", 
    "asarray", 
    "aslazy", 
    "zeros", 
    "empty", 
    "ones", 
    "full",
    "gaussian_kernel", 
    "circular_mask", 
    "imread", 
    "imread_collection", 
    "lazy_imread", 
    "read_meta", 
    "sample_image",
    "broadcast_arrays"
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
    
def write_docs(func):
    func.__doc__ = re.sub(r"{}", shared_docs, func.__doc__)
    return func

@write_docs
def array(
    arr: ArrayLike,
    dtype: DTypeLike = None, 
    *,
    name: str = None,
    axes: str = None,
    copy: bool = True
) -> ImgArray:
    """
    make an ImgArray object, like ``np.array(x)``
    
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
    if isinstance(arr, np.ndarray) and dtype is None:
        if arr.dtype in (np.uint8, np.uint16, np.float32):
            dtype = arr.dtype
        elif arr.dtype.kind == "f":
            dtype = np.float32
        else:
            dtype = arr.dtype
    
    arr = np.array(arr, dtype=dtype, copy=copy)
        
    # Automatically determine axes
    if axes is None:
        axes = ["x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][arr.ndim-1]
            
    self = ImgArray(arr, name=name, axes=axes)
    
    return self

@write_docs
def asarray(
    arr: ArrayLike,
    dtype: DTypeLike = None,
    *, 
    name: str = None,
    axes: str = None
) -> ImgArray:
    """
    make an ImgArray object, like ``np.asarray(x)``
    
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
    return array(arr, dtype=dtype, name=name, axes=axes, copy=False)

@write_docs
def aslazy(
    arr: ArrayLike, 
    dtype: DTypeLike = None,
    *, 
    name: str = None,
    axes: str = None,
    chunks="auto"
) -> LazyImgArray:
    """
    Make an LazyImgArray object from other types of array.
    
    Parameters
    ----------
    arr : array-like
        Base array.
    {}
    chunks : int, tuple
        How to chunk the array. For details see ``dask.array.from_array``.
        
    Returns
    -------
    LazyImgArray
    
    """
    from dask import array as da
    if isinstance(arr, MetaArray):
        arr = da.from_array(arr.value, chunks=chunks)
    if isinstance(arr, (np.ndarray, np.memmap)):
        arr = da.from_array(arr, chunks=chunks)
    elif not isinstance(arr, da.core.Array):
        arr = da.asarray(arr)
        
    if isinstance(arr, np.ndarray) and dtype is None:
        if arr.dtype in (np.uint8, np.uint16, np.float32):
            dtype = arr.dtype
        elif arr.dtype.kind == "f":
            dtype = np.float32
        else:
            dtype = arr.dtype
        
    # Automatically determine axes
    if axes is None:
        axes = ["x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][arr.ndim-1]
            
    self = LazyImgArray(arr, name=name, axes=axes)
    
    return self

_P = ParamSpec("_P")

def _inject_numpy_function(func: Callable[_P, np.ndarray]) -> Callable[_P, ImgArray]:
    npfunc: Callable = getattr(np, func.__name__)
    @wraps(func)
    def _func(*args, **kwargs):
        axes = kwargs.pop("axes", None)
        name = kwargs.pop("name", None)
        return asarray(npfunc(*args, **kwargs), name=name, axes=axes)
    _func.__doc__ = (
        f"""
        impy version of numpy.{func.__name__}. This function has additional parameters ``axes``
        and ``name``. Original docstring follows.
        
        Additional Parameters
        ---------------------
        axes : str, optional
            Image axes. Must be same length as image dimension.
        name: str, optional
            Image name.
        
        {npfunc.__doc__}
        """
        )
    _func.__annotations__.update({"return": ImgArray})
    return _func

@_inject_numpy_function
def zeros(shape: _ShapeLike, dtype: DTypeLike = np.uint16, *, name: str = None, axes: str = None): ...
    
@_inject_numpy_function
def empty(shape: _ShapeLike, dtype: DTypeLike = np.uint16, *, name: str = None, axes: str = None): ...

@_inject_numpy_function
def ones(shape: _ShapeLike, dtype: DTypeLike = np.uint16, *, name: str = None, axes: str = None): ...

@_inject_numpy_function
def full(shape: _ShapeLike, fill_value: Any, dtype: DTypeLike = np.uint16, *, name: str = None, axes: str = None): ...


def gaussian_kernel(
    shape: _ShapeLike, 
    sigma: nDFloat = 1.0,
    peak: float = 1.0,
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
    g = gauss.GaussianParticle([(np.array(shape)-1)/2, sigma, peak, 0])
    ker = g.generate(shape)
    ker = array(ker, name="Gaussian-Kernel")
    if ker.ndim == 3:
        ker.axes = "zyx"
    return ker


def circular_mask(
    radius: nDFloat, 
    shape: _ShapeLike,
    center: str | tuple[float, ...] = "center"
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
        radius = [radius]*len(shape)
    elif len(radius) != len(shape):
        raise ValueError("Length of `shape` and `radius` must be same.")

    x = np.indices(shape)
    s = sum(((x0 - c0)/r0)**2 for x0, c0, r0 in zip(x, center, radius))
    axes = "zyx" if len(shape) == 3 else None # change the default axes in `array`
    
    return array(s > 1.0, dtype=bool, axes=axes)


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
    key: str = None,
    *, 
    name: str | None = None,
    squeeze: bool = False,
) -> ImgArray:
    r"""
    Load image(s) from a path. You can read list of images from directories with wildcards or ``"$"``
    in ``path``.

    Parameters
    ----------
    path : str
        Path to the image or directory.
    dtype : Any type that np.dtype accepts
        Data type of images.
    key : str, optional
        If not None, image is read in a memory-mapped array first, and only ``img[key]`` is returned.
        Only axis-targeted slicing is supported. This argument is important when reading a large
        file.
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
    
        >>> path = r"C:\...\Image.mrc"
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
    elif not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: {path}")
    
    # read tif metadata
    if not is_memmap:
        size = os.path.getsize(path) / 1e9
        if size > Const["MAX_GB"]:
            raise MemoryError(f"Too large {size:.2f} GB")

    image_data = io.imread(path, memmap=is_memmap)

    img = image_data.image
    axes = image_data.axes
    scale = image_data.scale
    metadata = image_data.metadata
    spatial_scale_unit = metadata.get("unit", "px")
        
    if is_memmap and axes is not None:
        sl = axis_targeted_slicing(tuple(axes), key)
        axes = "".join(a for a, k in zip(axes, sl) if not isinstance(k, int))
        img = np.asarray(img[sl], dtype=dtype)
    
    self = ImgArray(img, name=name, axes=axes, source=path, metadata=metadata)
        
    # In case the image is in yxc-order. This sometimes happens.
    if "c" in self.axes and self.shape.c > self.shape.x:
        self: ImgArray = np.moveaxis(self, -1, -3)
        _axes = self.axes._axis_list
        _axes = _axes[:-3] + "cyx"
        self.axes = _axes
    
    if dtype is None:
        dtype = self.dtype
    
    if squeeze:
        self = np.squeeze(self)
        
    # if key="y=0", ImageAxisError happens here because image loses y-axis. We have to set scale
    # one by one.
    for k, v in scale.items():
        if k in self.axes:
            self.set_scale({k: v})
            if k in "zyx":
                self.axes[k].unit = spatial_scale_unit
    
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
    key: str = None,
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
) -> DataList:
    """
    Open images as ImgArray and store them in DataList.

    Parameters
    ----------
    path : str or list of str
        Path than can be passed to ``glob.glob``. If a list of path is given, all the matched 
        images will be read and concatenated into a DataList.
    filt : callable, optional
        If specified, only images that satisfies filt(img)==True will be stored in the returned 
        DataList.

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


def lazy_imread(
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
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}.")
    
    if "*" in path:
        return _lazy_imread_glob(path, chunks=chunks, squeeze=squeeze)
    
    # read as a dask array
    image_data = io.imread_dask(path, chunks)
    img = image_data.image
    axes = image_data.axes
    scale = image_data.scale
    metadata = image_data.metadata
    spatial_scale_unit = metadata.get("unit", "px")
    
    if squeeze:
        axes = "".join(a for i, a in enumerate(axes) if img.shape[i] > 1)
        img = np.squeeze(img)
    
    self = LazyImgArray(img, name=name, axes=axes, source=path, metadata=metadata)

    # read lateral scale if possible
    self.set_scale(**scale)
    
    for k, v in scale.items():
        if k in self.axes:
            self.set_scale({k: v})
            if k in "zyx":
                self.axes[k].unit = spatial_scale_unit
    
    return self.sort_axes()


def _lazy_imread_glob(path: str, squeeze: bool = False, **kwargs) -> LazyImgArray:
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
        imgl = lazy_imread(path, **kwargs)
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


def read_meta(path: str) -> dict[str]:
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
    path = str(path)
    image_data = io.imread_dask(path, chunks="default")
    return {
        "axes": image_data.axes,
        "scale": image_data.scale,
        "metadata": image_data.metadata
    }
