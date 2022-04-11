from __future__ import annotations
import numpy as np
from typing import Callable
from functools import wraps
from .arrays import *
from .utils.axesop import *
from .utils.utilcls import Progress
from .utils.deco import *

# Extend ImgArray with custom functions.
# TODO: use annotation to determine "kind"
class bind:
    """
    Dynamically define ImgArray function that can iterate over axes. You can integrate your own
    function, or useful functions from `skimage` or `opencv`.
    This class is designed as a kind of decorator class so that it can be used as decorator of
    any function or directly takes a function as the first argument.

    Parameters
    ----------
    func : callable
        Function to wrapped and bound to ImgArray.
    funcname : str, optional
        Method name after set to ImgArray. The name of `func` itself will be set by default.
    indtype : dtype, optional
        If given, input data type will be converted by `as_img_type` method before passed to `func`.
    outdtype : dtype, optional
        If given, output data array will be defined in this type if needed.
    kind : str, {"image", "property", "label", "label_binary"}, default is "image"
        What kind of function will be bound.
        
        - "image" ... Given an image, calculate a new image that has the exactily same shape.
          Bound method will return `ImgArray` that has the same shape and axes as the input image.
        - "property" ... Given an image, calculate a scalar value or any other object such as
        tuple, and store them in a `PropArray`. Axes of returned `PropArray` is (the axes of input
        image) - (axes of spatial dimensions specified by `dims` argument of bound method).
        - "label" ... Given an image, calculate a label image with value 0 being background and set
        it to `labels` attribute. The label image must have the exactly same shape as input image.
        - "label_binary" ... Given an image, calculate a binary image. Label image is generated from
        the binary image with `label` method in `LabeledArray`. The connectivity is None. The binary
        image must have the exactly same shape as input image.
        
    ndim : {None, 2, 3}, default is None
        Dimension of image that the original function supports. If None, then it is assumed to
        support both 2 and 3 dimensional images and automatically determined by the universal
        `dims_to_spatial_axes` method.

    Examples
    --------
    1. Bind "normalize" method that will normalize images separately.
    
        >>> def normalize(img):
        >>>    min_, max_ = img.min(), img.max()
        >>>    return (img - min_)/(max_ - min_)
        >>> ip.bind(normalize, indtype=np.float32, outdtype=np.float32)
        >>> img = ip.imread(...)
        >>> img.normalize()
    
    2. Bind a method `calc_mean` that calculate mean value around spatial dimensions. For one yx-
    or zyx-image, a scalar value is returned, so that `calc_mean` should return `PropArray`.
    
        >>> ip.bind(np.mean, "calc_mean", outdtype=np.float32, kind="property")
        >>> img = ip.imread(...)
        >>> img.calc_mean()
    
    3. Wrap the normalize function in (1) in a decorator method.
    
        >>> @ip.bind(indtype=np.float32, outdtype=np.float32)
        >>> def normalize(img):
        >>>    min_, max_ = img.min(), img.max()
        >>>    return (img - min_)/(max_ - min_)
        >>> img = ip.imread(...)
        >>> img.normalize()
    
    or if you think `indtype` and `outdtype` are unnecessary:
    
        >>> @ip.bind
        >>> def normalize(img):
        >>>     ...
    
    4. Bind custom percentile labeling function (although `label_threshold` method can do the 
    exactly same thing).
    
        >>> @ip.bind(kind="label_binary")
        >>> def mylabel(img, p=90):
        >>>     per = np.percentile(img, p)
        >>>     thr = img > per
        >>>     return thr
        >>> img = ip.imread(...)
        >>> img.mylabel(95)   # img.labels is added here
        
    """    
    bound = set()
    last_added = None
    def __init__(self, func:Callable=None, funcname:str=None, *, indtype=None, outdtype=None, 
                 kind:str="image", ndim:int|None=None):
        """
        Method binding is done inside this when bind object is used as function like:
            >>> ip.bind(func, "funcname", ...)
        """        
        if callable(func):
            self._bind_method(func, 
                              funcname=funcname, 
                              indtype=indtype, 
                              outdtype=outdtype, 
                              kind=kind, 
                              ndim=ndim)
        else:
            self.funcname = func
            self.indtype = indtype
            self.outdtype = outdtype
            self.kind = kind
            self.ndim = ndim
    
    def __call__(self, func:Callable):
        """
        Method binding is done inside this when bind object is used as decorator like:
            >>> @ip.bind(...)
            >>> def ...
        """
        if callable(func):
            self._bind_method(func, 
                              funcname=self.funcname, 
                              indtype=self.indtype, 
                              outdtype=self.outdtype, 
                              kind=self.kind, 
                              ndim=self.ndim, 
                              )
        return func
    
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self._unbind_method(self.__class__.last_added)
        
    def _bind_method(self, func:Callable, funcname:str=None, *, indtype=None, outdtype=None,
                     kind="image", ndim=None):
        # check function's name
        if funcname is None:
            fn = func.__name__
        elif isinstance(funcname, str):
            fn = funcname
        else:
            raise TypeError("`funcname` must be str if given.")
        if hasattr(ImgArray, fn) and fn not in self.bound:
            raise AttributeError(f"ImgArray already has attribute '{fn}'. Consider other names.")
        
        # check ndim and define default value of dims
        if ndim is None:
            default_dims = None
        elif ndim == 2:
            default_dims = "yx"
        elif ndim == 3:
            default_dims = "zyx"
        else:
            raise ValueError(f"`ndim` must be None, 2 or 3, but got {ndim}.")
        
        
        if outdtype == "float64" or outdtype is None:
            outdtype = np.float32
        
        # Dynamically define functions used inside the plugin method, depending on `kind` option.
        # _exit : overwrites output attributes.
        
        if kind == "image":
            _drop_axis = lambda dims: None
            def _exit(out, img, func, *args, **kwargs):
                out = out.view(ImgArray).as_img_type(outdtype)
                out._set_info(img)
                return out
            
        elif kind == "property":
            _drop_axis = lambda dims: dims
            def _exit(out, img, func, *args, dims=None, **kwargs):
                out = PropArray(out, name=img.name, axes=complement_axes(dims, img.axes), 
                                propname=fn, dtype=outdtype)
                return out
                
        elif kind == "label":
            _drop_axis = lambda dims: None
            def _exit(out, img, func, *args, dims=None, **kwargs):
                img.labels = Label(out, name=img.name, axes=img.axes, dirpath=img.dirpath).optimize()
                img.labels.set_scale(img)
                return img.labels
            
        elif kind == "label_binary":
            _drop_axis = lambda dims: None
            def _exit(out, img, func, *args, dims=None, **kwargs):
                arr = LabeledArray(out)
                arr._set_info(img)
                lbl = arr.label(dims=dims)
                img.labels = lbl
                img.labels.set_scale(img)
                return img.labels
                        
        else:
            raise NotImplementedError(kind)
        
        # Define method and bind it to ImgArray
        @wraps(func)
        @dims_to_spatial_axes
        def _func(img, *args, dims=default_dims, **kwargs):
            if indtype is not None:
                img = img.as_img_type(indtype)
                
            with Progress(fn):
                out = img.apply_dask(func,
                                     c_axes=complement_axes(dims, img.axes),
                                     drop_axis=_drop_axis(dims),
                                     args=args,
                                     kwargs=kwargs
                                     )
                out = _exit(out, img, func, *args, dims=dims, **kwargs)
                
            return out
        
        self.__class__.bound.add(fn)
        self.__class__.last_added = fn
        return setattr(ImgArray, fn, _func)
    
    def _unbind_method(self, funcname:str):
        self.__class__.bound.remove(funcname)
        return delattr(ImgArray, funcname)