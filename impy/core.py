from __future__ import annotations
from impy.label import Label
from .specials import PropArray
from .deco import dims_to_spatial_axes, safe_str, record
import numpy as np
import os
import glob
import itertools
import collections
from skimage import io
from .imgarray import ImgArray
from .func import *
from .bases import MetaArray, HistoryArray
from .axes import Axes, ImageAxesError
from .utilcls import Progress
from skimage import data as skdata
from typing import Callable

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

def zeros_like(img:ImgArray, name:str=None) -> ImgArray:
    if not isinstance(img, ImgArray):
        raise TypeError("'zeros_like' in impy can only take ImgArray as an input")
    
    return zeros(img.shape, dtype=img.dtype, name=name, axes=img.axes)

def empty(shape, dtype=np.uint16, *, name=None, axes=None) -> ImgArray:
    return array(np.empty(shape, dtype=dtype), dtype=dtype, name=name, axes=axes)

def empty_like(img:ImgArray, name:str=None) -> ImgArray:
    if not isinstance(img, ImgArray):
        raise TypeError("'empty_like' in impy can only take ImgArray as an input")
    
    return empty(img.shape, dtype=img.dtype, name=name, axes=img.axes)

def imread(path:str, dtype:str=None, *, axes=None) -> ImgArray:
    """
    Load image from path.

    Parameters
    ----------
    path : str
        Path to the image.
    dtype : Any type that np.dtype accepts
        dtype of the images.
    axes : str or None, optional
        If the image does not have axes metadata, this value will be used.

    Returns
    -------
    ImgArray
    """    
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: {path}")
    
    fname, fext = os.path.splitext(os.path.basename(path))
    img = io.imread(path)
    dirpath = os.path.dirname(path)
    
    # read tif metadata
    if fext == ".tif":
        meta = get_meta(path)
    elif fext in (".png", ".jpg") and img.ndim == 3 and img.shape[-1] <= 4:
        meta = {"axes":"yxc", "ijmeta":{}, "history":[]}
    else:
        meta = {"axes":axes, "ijmeta":{}, "history":[]}
    
    axes = meta["axes"]
    metadata = meta["ijmeta"]
    if meta["history"]:
        name = meta["history"].pop(0)
        history = meta["history"]
    else:
        name = fname
        history = []
    
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
        spacing = meta["ijmeta"].get("spacing", 1.0)
        try:
            tags = meta["tags"]
            xres = tags["XResolution"]
            yres = tags["YResolution"]
            dx = xres[1]/xres[0]
            dy = yres[1]/yres[0]
        except KeyError:
            dx = dy = spacing
        
        self.set_scale(x=dx, y=dy)
        
        # read z scale if needed
        if "z" in self.axes:
            dz = spacing
            self.set_scale(z=dz)
        return self.sort_axes().as_img_type(dtype) # arrange in tzcyx-order

def imread_collection(dirname:str, axis:str="p", *, filename:str="*.tif", template:dict|MetaArray=None,
                      ignore_exception:bool=False, dtype=None) -> ImgArray:
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
    
    Example
    -------
    (1) Read Tiff images that start with "100nM-":
    >>> img = ip.imread_collection(r"C:\...", filename="100nM-*.tif")
    
    (2) Read Tiff images that have tyx-axes:
    >>> img = ip.imread_collection(r"C:\...", template={"axes: "tyx"})
    
    (3) Read Tiff images that have strictly same features as a reference image `ref`:
    >>> img = ip.imread_collection(r"C:\...", template=ref)
    """    
    paths = glob.glob(os.path.join(dirname, "**", filename), recursive=True)
    
    # determine template
    template_keys = {"shape", "axes", "scale"}
    if template is None:
        template = {}
    elif isinstance(template, dict):
        if not set(template.keys()) <= template_keys:
            raise ValueError(f"template only takes {template_keys} as keys.")
    elif isinstance(template, MetaArray):
        template = {k: getattr(template, k) for k in template_keys}
    else:
        raise TypeError(f"template must be dict or MetaArray, but got {type(template)}.")
    
    imgs = []
    shapes = []
    for path in paths:
        img = imread(path, dtype=dtype)
        for k, v in template.items():
            if getattr(img, k) != v:
                continue
        imgs.append(img)
        shapes.append(img.shape)
    
    # check shape compatibility
    list_of_shape = list(set(shapes))
    if len(list_of_shape) > 1:
        if ignore_exception:
            ctr = collections.Counter(shapes)
            common_shape = ctr.most_common()[0][0]
            imgs = [img for img in imgs if img.shape == common_shape]
        else:
            raise ValueError("Input directory has images with different shapes: "
                            f"{', '.join(map(str, list_of_shape))}")
    
    if len(imgs) == 0:
        raise RuntimeError("Could not read any images.")
    
    out = stack(imgs, axis=axis)
    out.dirpath, out.name = os.path.split(dirname)
    out.history[-1] = "imread_collection"
    out.temp = paths
    return out

def imread_stack(path:str, dtype=None):
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
    FORMAT = r"\$[a-z]"
    new_axes = list(map(lambda x: x[1:], re.findall(r"\$[a-z]", path)))
    
    # To convert input path string into that for glob.glob, replace $X with wildcard
    # e.g.) ~\XX$t_YY$z -> ~\XX*_YY*
    finder_path = re.sub(FORMAT, "*", path)
    
    # To convert input path string into file-finding regex pattern.
    # e.g.) ~\XX$t_YY$z\*.tif -> ~\\XX(\d)_YY(\d)\\.*\.tif
    path = repr(path)[1:-1]
    pattern = re.sub(r"\.", r"\.",path)          # dots to non-escape
    pattern = re.sub(r"\*", ".*", pattern)       # asters to non-escape
    pattern = re.sub(FORMAT, r"(\\d+)", pattern) # make number finders
    pattern = re.compile(pattern)
    
    # To convert input path string into format string to execute imread.
    # e.g.) ~\XX$t_YY$z -> ~\XX{}_YY{}
    fpath = re.sub(FORMAT, "{}", path)
    
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
        
        img = imread(found_paths[0], dtype=dtype)
        
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
    return self.sort_axes()
    

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
    if not path.endswith(".tif"):
        raise ValueError("Cannot read metadata from file extension other than tif.")
    meta = get_meta(path)
    return meta

def stack(imgs, axis="c", dtype=None):
    """
    Create stack image from list of images.

    Parameters
    ----------
    imgs : iterable object of images.
        Images to stack. These images must have exactly the same shapes.
    axis : str, default is "c"
        Which axis will be the new one.
    dtype : str, optional
        Output dtype.

    Returns
    -------
    ImgArray
        Image stack
    """    
    
    if isinstance(imgs, np.ndarray):
        raise TypeError("cannot stack single array.")
    
    # find where to add new axis
    if imgs[0].axes:
        new_axes = Axes(axis + str(imgs[0].axes))
        new_axes.sort()
        _axis = new_axes.find(axis)
    else:
        new_axes = None
        _axis = 0

    if dtype is None:
        dtype = imgs[0].dtype

    arrs = [img.as_img_type(dtype).value for img in imgs]

    out = np.stack(arrs, axis=0)
    out = np.moveaxis(out, 0, _axis)
    out = out.view(ImgArray)
    out._set_info(imgs[0], f"Make-Stack(axis={axis})", new_axes)
    
    return out

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

def squeeze(img:MetaArray):
    out = np.squeeze(img)
    new_axes = "".join(a for a in img.axes if img.sizeof(a) > 1)
    if isinstance(img, HistoryArray):
        out._set_info(img, "squeeze", new_axes=new_axes)
    else:
        try:
            out._set_info(img, new_axes=new_axes)
        except Exception:
            pass
    return out
    
def bind_method(func:Callable, funcname:str=None, *, indtype=None, outdtype=None, kind="image",
                mapping:dict[str, tuple[str, Callable]]=None) -> None:
    """
    Dynamically define ImgArray function that can iterate over axes. You can integrate your own
    function, or useful functions from `skimage` or `opencv`. History of functiom call will be 
    similarly recorded in `self.history`.

    Parameters
    ----------
    func : callable
        Function to wrapped and bound to ImgArray.
    funcname : str, optional
        Method name after set to ImgArray. The name of `func` itself will be set by default.
    indtype : dtype, optional
        If given, input data type will be converted by `as_img_type` method before passed to `func`.
    outdtype : dtype, optional
        If given, output data array will be defined in this type.
    kind : str, default is "image"
        What kind of function will be defined.
    mapping : dict of str -> (str, callable), optional
        If given, keyword arguments are converted using this mapping before passed to `func`. This
        keyword is used for modifing original function without wrapping it. For more detail see
        Example (2).

    Example
    -------
    (1) Bind "normalize" method that 
    >>> def normalize(img):
    >>>    min_, max_ = img.min(), img.max()
    >>>    return (img - min_)/(max_ - min_)
    >>> ip.bind_method(normalize, indtype=np.float32, outdtype=np.float32)
    >>> img = ip.imread(...)
    >>> img.max_filter(radius=3)
    
    (2) Bind `skimage.filters.rank.maximum` for filtering, but make it take "radius" rather than
    "selem" as a keyword argument.
    >>> from impy.func import ball_like
    >>> from skimage.filters.rank import maximum
    >>> ip.bind_method(maximum, "max_filter", mapping={"radius":("selem": ball_like)})
    >>> img = ip.imread(...)
    >>> img.max_filter(radius=3)
    
    (3) Bind a method `calc_mean` that calculate mean value around spatial dimensions. For one yx-
    or zyx-image, a scalar value is returned, so that `calc_mean` should return `PropArray`.
    >>> ip.bind_method(np.mean, "calc_mean", outdtype=np.float32, kind="property")
    >>> img = ip.imread(...)
    >>> img.calc_mean()
    """    
    # check function's name
    if funcname is None:
        fn = func.__name__
    elif isinstance(funcname, str):
        fn = funcname
    else:
        raise TypeError("`funcname` must be str if given.")
    
    if mapping is None:
        mapping = {}
    
    # Dynamically define functions used inside the plugin method, depending on `kind` option.
    # _prepare_output_array : returns a subclass of ndarray for output.
    # _iter : returns an iterator around spatial dimensions.
    # _exit : overwrites output attributes.
    if kind == "image":
        def _prepare_output_array(self, dims):
            dtype = outdtype if outdtype is not None else self.dtype
            return np.empty(self.shape, dtype=dtype)
        
        def _iter(self, dims):
            return self.iter(complement_axes(dims, self.axes))
        
        def _exit(out, self, func, *args, **kwargs):
            out = out.view(ImgArray)
            _args = list(map(safe_str, args))
            _kwargs = [f"{safe_str(k)}={safe_str(v)}" for k, v in kwargs.items()]
            history = f"{func.__name__}({','.join(_args + _kwargs)})"
            out._set_info(self, history)
            return out
        
    elif kind == "property":
        def _prepare_output_array(self, dims):
            dtype = outdtype if outdtype is not None else object
            c_axes = complement_axes(dims, self.axes)
            shape = self.sizesof(c_axes)
            return PropArray(np.empty(shape, dtype=dtype), name=self.name, dirpath=self.dirpath, 
                             axes=c_axes, dtype=dtype)
        
        def _iter(self, dims):
            return self.iter(complement_axes(dims, self.axes), exclude=dims)
        
        def _exit(out, self, func, *args, **kwargs):
            out.propname = fn
            return out
            
    elif kind == "label":
        def _prepare_output_array(self, dims):
            return largest_zeros(self.shape)
        
        def _iter(self, dims):
            return self.iter(complement_axes(dims, self.axes))
        
        def _exit(out, self, func, *args, **kwargs):
            self.labels = Label(out, name=self.name, axes=self.axes, dirpath=self.dirpath).optimize()
            self.labels.history.append(fn)
            self.labels.set_scale(self)
            return out    
        
    else:
        raise NotImplementedError(kind)
    
    # Define method
    @dims_to_spatial_axes
    def _func(self, *args, dims=None, **kwargs):
        if indtype is not None:
            self = self.as_img_type(indtype)
        
        # mapping keyword arguments if necessary
        kw = dict()
        for k, v in kwargs.items():
            m = mapping.get(k, None)
            if m is None:
                kw[k] = v
            else:
                newkey, val = m
                try:
                    kw[newkey] = val(v, len(dims))
                except TypeError:
                    kw[newkey] = val(v)
            
        out = _prepare_output_array(self, dims)
            
        with Progress(fn):
            for sl, img in _iter(self, dims):
                out[sl] = func(img, *args, **kw)
            out = _exit(out, self, func, *args, **kwargs)
        return out
    
    if hasattr(ImgArray, fn):
        print(f"ImgArray already has attribute '{fn}'. It is overwritten.")
    return setattr(ImgArray, fn, _func)
