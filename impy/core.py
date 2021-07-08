from __future__ import annotations
from .datalist import DataList
from .arrays import ImgArray, LazyImgArray
import numpy as np
import os
import re
import glob
import itertools
from .func import *
from .axes import ImageAxesError
from .utilcls import Progress
from skimage import data as skdata

__all__ = ["array", "zeros", "empty", "gaussian_kernel", "imread", "imread_collection", "lazy_imread",
           "read_meta", "set_cpu", "set_verbose", "sample_image"]

# TODO: 
# - delayed imread
# - e.g. ip.imread("...\$i$j.tif", key="i=2:") will raise error.

def array(arr, dtype=None, *, name=None, axes=None) -> ImgArray:
    """
    make an ImgArray object, just like np.array(x)
    """
    if isinstance(arr, str):
        raise TypeError(f"String is invalid input. Do you mean imread(path)?")
    if isinstance(arr, np.ndarray) and dtype is None:
        if arr.dtype in (np.uint8, np.uint16, np.float32):
            dtype = arr.dtype
        elif arr.dtype.kind == "f":
            dtype = np.float32
        else:
            dtype = arr.dtype
    
    arr = np.asarray(arr, dtype=dtype)
        
    # Automatically determine axes
    if axes is None:
        axes = ["x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][arr.ndim-1]
            
    self = ImgArray(arr, name=name, axes=axes)
    
    return self

def zeros(shape, dtype=np.uint16, *, name=None, axes=None) -> ImgArray:
    return array(np.zeros(shape, dtype=dtype), dtype=dtype, name=name, axes=axes)

def empty(shape, dtype=np.uint16, *, name=None, axes=None) -> ImgArray:
    return array(np.empty(shape, dtype=dtype), dtype=dtype, name=name, axes=axes)

def gaussian_kernel(shape:tuple[int], sigma=1.0, peak=1.0):
    """
    Make an Gaussian kernel or Gaussian image.

    Parameters
    ----------
    shape : tuple[int]
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

def imread(path:str, dtype:str=None, key:str=None, *, axes=None) -> ImgArray:
    """
    Load image(s) from a path. You can read list of images from directories with wildcards or "$"
    in `path`.

    Parameters
    ----------
    path : str
        Path to the image or directory.
    dtype : Any type that np.dtype accepts
        Data type of images.
    key : str, optional
        If not None, image is read in a memory-mapped array first, and only img[key] is returned.
        Only axis-targeted slicing is supported. This argument is important when reading a large
        file.
        >>> path = r"C:\...\Image.mrc"
        >>> %time ip.imread(path)["x=:10;y=:10"]
            Wall time: 136 ms
        >>> %time ip.imread(path, key="x=:10;y=:10")
            Wall time: 3.01 ms
            
    axes : str or None, optional
        If the image does not have axes metadata, this value will be used.

    Returns
    -------
    ImgArray
    """    
    path = str(path)
    is_memmap = (key is not None)
    
    if "$" in path:
        return _imread_stack(path, dtype=dtype, key=key)
    elif "*" in path:
        return _imread_glob(path, dtype=dtype, key=key)
    elif not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: {path}")
    
    fname, fext = os.path.splitext(os.path.basename(path))
    dirpath = os.path.dirname(path)
    
    # read tif metadata
    meta, img = open_img(path, memmap=is_memmap)
    axes = meta["axes"]
    metadata = meta["ijmeta"]
    if meta["history"]:
        name = meta["history"].pop(0)
        history = meta["history"]
    else:
        name = fname
        history = []
        
    if is_memmap:
        sl = axis_targeted_slicing(img, axes, key)
        img = np.asarray(img[sl])
    
    self = ImgArray(img, name=name, axes=axes, dirpath=dirpath, 
                    history=history, metadata=metadata)
        
    # In case the image is in yxc-order. This sometimes happens.
    if "c" in self.axes and self.sizeof("c") > self.sizeof("x"):
        self = np.moveaxis(self, -1, -3)
        _axes = self.axes.axes
        _axes = _axes[:-3] + "cyx"
        self.axes = _axes
    
    if dtype is None:
        dtype = self.dtype
        
    if self.axes.is_none():
        return self
    else:
        # read lateral scale if possible
        scale = get_scale_from_meta(meta)
        self.set_scale(**scale)
        return self.sort_axes().as_img_type(dtype) # arrange in tzcyx-order

def _imread_glob(path:str, axis:str="p", dtype=None) -> ImgArray:
    """
    Read images recursively from a directory, and stack them into one ImgArray.

    Parameters
    ----------
    dirname : str
        Path to the directory
    axis : str, default is "p"
        To specify which axis will be the new one.
    filname : str, default is "*.tif"
        File name that satisfies this string will be read. This variable will be passed to `glob.glob`.
    template : dict or MetaArray, optional
        Images that matches the template will added to image stack.
    ignore_exception : bool, default is False
        If true, arrays with wrong shape will be ignored.
    dtype : Any type that np.dtype accepts
        dtype of the images.
    """    
    path = str(path)
    paths = glob.glob(path, recursive=True)
        
    imgs = []
    for path in paths:
        img = imread(path, dtype=dtype)
        imgs.append(img)
    
    if len(imgs) == 0:
        raise RuntimeError("Could not read any images.")
    
    out = np.stack(imgs, axis=axis)
    try:
        base = os.path.split(path.split("*")[0])[0]
        out.dirpath, out.name = os.path.split(base)
    except Exception:
        pass
    out.temp = paths
    return out

def _imread_stack(path:str, dtype=None, key:str=None):
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
    pattern = re.sub(r"\.", r"\.",path_)          # dots to non-escape
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
    # determine dirpath and name
    name_list = []
    for p in path.split(os.sep):
        if "$" in p or "*" in p:
            break
        else:
            name_list.append(p)
    if len(name_list) > 0:
        self.dirpath = os.path.join(*name_list[:-1])
        self.name = name_list[-1]
    else:
        self.dirpath = None
        self.name = None
    return self.sort_axes()

def imread_collection(path:str, filt=None) -> DataList:
    """
    Open images as ImgArray and store them in ArrayList.

    Parameters
    ----------
    path : str
        Path than can be passed to `glob.glob`.
    filt : callable, optional
        If specified, only images that satisfies filt(img)==True will be stored in the returned 
        ArrayList.

    Returns
    -------
    ArrayList
    """    
    paths = glob.glob(str(path), recursive=True)
    if filt is None:
        filt = lambda x: True
    arrlist = DataList()
    for path in paths:
        img = imread(path)
        if filt(img):
            arrlist._append(img)
    return arrlist

def read_meta(path:str) -> dict[str]:
    """
    Read the metadata of a tiff file. 

    Parameters
    ----------
    path : str
        Path to the tiff file.

    Returns
    -------
    dict
        Dictionary of metadata with following keys.
        "axes": axes information
        "ijmeta": ImageJ metadata
        "history": impy history
        "tags": tiff tags
    """    
    fname, fext = os.path.splitext(os.path.basename(path))
    
    if fext in (".tif", ".tiff"):
        meta = open_tif(path)
    elif fext in (".mrc", ".rec"):
        meta = open_mrc(path)
    else:
        raise ValueError("Unsupported file extension.")
    
    return meta

def lazy_imread(path):
    path = str(path)
    fname, fext = os.path.splitext(os.path.basename(path))
    dirpath = os.path.dirname(path)
    
    # read tif metadata
    meta, img = open_as_dask(path)
    axes = meta["axes"]
    metadata = meta["ijmeta"]
    if meta["history"]:
        name = meta["history"].pop(0)
        history = meta["history"]
    else:
        name = fname
        history = []
        
    self = LazyImgArray(img, name=name, axes=axes, dirpath=dirpath, 
                        history=history, metadata=metadata)
        
        
    return self

def set_cpu(n_cpu:int) -> None:
    ImgArray.n_cpu = n_cpu
    return None

def set_verbose(b:bool) -> None:
    Progress.show_progress = b
    return None

def sample_image(name:str) -> ImgArray:
    """
    Get sample images from `skimage` and convert it into ImgArray.

    Parameters
    ----------
    name : str
        Name of sample image, such as "camera".

    Returns
    -------
    ImgArray
        Sample image.
    """    
    img = getattr(skdata, name)()
    out = array(img, name=name)
    if out.shape[-1] == 3:
        out.axes = "yxc"
        out = out.sort_axes()
    return out

