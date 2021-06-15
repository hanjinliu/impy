from __future__ import annotations
import numpy as np
from typing import Callable
from .arrays import *
from .func import *
from .utilcls import Progress
from .deco import *


# Extend ImgArray with custom functions.

class bind:
    """
    Dynamically define ImgArray function that can iterate over axes. You can integrate your own
    function, or useful functions from `skimage` or `opencv`. History of function call will be 
    similarly recorded in `self.history`.
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
    mapping : dict of str -> (str, callable), optional
        If given, keyword arguments are converted using this mapping before passed to `func`. This
        keyword is used for modifing original function without wrapping it. For more detail see
        Example (2).

    Examples
    --------
    (1) Bind "normalize" method that will normalize images separately.
    >>> def normalize(img):
    >>>    min_, max_ = img.min(), img.max()
    >>>    return (img - min_)/(max_ - min_)
    >>> ip.bind(normalize, indtype=np.float32, outdtype=np.float32)
    >>> img = ip.imread(...)
    >>> img.normalize()
    
    (2) Bind `skimage.filters.rank.maximum` for filtering, but make it take "radius" rather than
    "selem" as a keyword argument.
    >>> from impy.func import ball_like
    >>> from skimage.filters.rank import maximum
    >>> ip.bind(maximum, "max_filter", mapping={"radius":("selem", ball_like)})
    >>> img = ip.imread(...)
    >>> img.max_filter(radius=3)
    
    (3) Bind a method `calc_mean` that calculate mean value around spatial dimensions. For one yx-
    or zyx-image, a scalar value is returned, so that `calc_mean` should return `PropArray`.
    >>> ip.bind(np.mean, "calc_mean", outdtype=np.float32, kind="property")
    >>> img = ip.imread(...)
    >>> img.calc_mean()
    
    (4) Wrap the normalize function in (1) in a decorator method.
    >>> @ip.bind(indtype=np.float32, outdtype=np.float32)
    >>> def normalize(img):
    >>>    min_, max_ = img.min(), img.max()
    >>>    return (img - min_)/(max_ - min_)
    >>> img = ip.imread(...)
    >>> img.normalize()
    or if you thick `indtype` and `outdtype` are unnecessary:
    >>> @ip.bind
    >>> def normalize(img):
    >>>     ...
    
    (5) Bind custom percentile labeling function (although `label_threshold` method can do the 
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
    def __init__(self, func:Callable=None, funcname:str=None, *, indtype=None, outdtype=None, 
                 kind:str="image", ndim:int|None=None, mapping:dict[str, tuple[str, Callable]]=None):
        """
        Method binding is done inside this when bind object is used as function like:
        >>> ip.bind(func, "funcname", ...)
        """        
        if callable(func):
            self._bind_method(func, funcname=funcname, indtype=indtype, outdtype=outdtype, 
                              kind=kind, ndim=ndim, mapping=mapping)
        else:
            self.funcname = func
            self.indtype = indtype
            self.outdtype = outdtype
            self.kind = kind
            self.ndim = ndim
            self.mapping = mapping
    
    def __call__(self, func:Callable):
        """
        Method binding is done inside this when bind object is used as decorator like:
        >>> @ip.bind(...)
        >>> def ...
        """
        if callable(func):
            self._bind_method(func, funcname=self.funcname, indtype=self.indtype, 
                              outdtype=self.outdtype, kind=self.kind, ndim=self.ndim, mapping=self.mapping)
        return func
    
    def _bind_method(self, func:Callable, funcname:str=None, *, indtype=None, outdtype=None,
                     kind="image", ndim=None, mapping:dict[str, tuple[str, Callable]]=None):
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
        
        # check mapping
        if mapping is None:
            mapping = {}
        elif not isinstance(mapping, dict):
            raise TypeError(f"`mapping` must be dict, but got {type(mapping)}")
        
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
                history = make_history(func.__name__, args, kwargs)
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
                return self.labels
            
        elif kind == "label_binary":
            def _prepare_output_array(self, dims):
                return largest_zeros(self.shape)
            
            def _iter(self, dims):
                return self.iter(complement_axes(dims, self.axes))
            
            def _exit(out, self, func, *args, **kwargs):
                self.labels = LabeledArray(out, axes=self.axes).label().labels
                self.labels.history.append(fn)
                self.labels.set_scale(self)
                return self.labels
                        
        else:
            raise NotImplementedError(kind)
        
        # Define method and bind it to ImgArray
        @wraps(func)
        @dims_to_spatial_axes
        def _func(self, *args, dims=default_dims, **kwargs):
            if indtype is not None:
                self = self.as_img_type(indtype)
            
            # map keyword arguments if necessary
            if mapping:
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
                kwargs = kw
                
            out = _prepare_output_array(self, dims)
                
            with Progress(fn):
                for sl, img in _iter(self, dims):
                    out[sl] = func(img, *args, **kwargs)
                out = _exit(out, self, func, *args, **kwargs)
            return out
        
        self.__class__.bound.add(fn)
        return setattr(ImgArray, fn, _func)