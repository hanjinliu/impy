from __future__ import annotations
import numpy as np
import os
import glob
import collections
from skimage import io
from .imgarray import ImgArray
from .func import *
from .bases.metaarray import MetaArray
from .axes import Axes
from .utilcls import Progress
from skimage import data as skdata

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
    dtype : str, optional
        dtype of the image.
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
    dtype : str, optional
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
    img = getattr(skdata, name)()
    out = array(img, name=name)
    if out.shape[-1] == 3:
        out.axes = "yxc"
        out = out.sort_axes()
    return out

def squeeze(img:MetaArray):
    out = np.squeeze(img)
    out.axes = "".join(a for a in img.axes if img.sizeof(a) > 1)
    return out
    
    