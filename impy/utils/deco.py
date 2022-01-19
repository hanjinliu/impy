from functools import wraps
import numpy as np
import inspect
import re
from dask import array as da
from .utilcls import Progress
from .._cupy import xp_ndarray, asnumpy

__all__ = ["record",
           "record_lazy",
           "same_dtype",
           "dims_to_spatial_axes",
           "make_history",
           ]
            
    
def record(func=None, *, append_history=True, record_label=False, only_binary=False, need_labels=False):
    def f(func):
        @wraps(func)
        def _record(self, *args, **kwargs):
            # check requirements of the ongoing function.
            if only_binary and self.dtype != bool:
                raise TypeError(f"Cannot run {func.__name__} with non-binary image.")
            if need_labels and not hasattr(self, "labels"):
                raise AttributeError(f"Function {func.__name__} needs labels."
                                    " Add labels to the image first.")
            
            # show the dimensionality of for-loop
            if "dims" in kwargs.keys():
                ndim = len(kwargs["dims"])
                suffix = f" ({ndim}D)"
            else:
                suffix = ""

            # start process
            with Progress(func.__name__ + suffix):
                out = func(self, *args, **kwargs)
            
            temp = getattr(out, "temp", None)
            
            if type(out) in (np.ndarray, xp_ndarray):
                out = asnumpy(out).view(self.__class__)
            
            # record history and update if needed
            ifupdate = kwargs.pop("update", False)
            
            if append_history:
                history = make_history(func.__name__, args, kwargs)
                if record_label:
                    out.labels._set_info(self.labels, history)
                else:
                    try:
                        out._set_info(self, history)
                    except AttributeError:
                        pass
                    
            ifupdate and self._update(out)
            
            # if temporary item exists
            if temp is not None:
                out.temp = temp
            return out
        return _record
    return f if func is None else f(func)

def record_lazy(func=None, *, append_history=True, only_binary=False):
    def f(func):
        @wraps(func)
        def _record(self, *args, **kwargs):
            if only_binary and self.dtype != bool:
                raise TypeError(f"Cannot run {func.__name__} with non-binary image.")
            
            out = func(self, *args, **kwargs)
            
            if isinstance(out, da.core.Array):
                out = self.__class__(out)
            
            # record history and update if needed
            ifupdate = kwargs.pop("update", False)
            
            if append_history:
                history = make_history(func.__name__, args, kwargs)
                try:
                    out._set_info(self, history)
                except AttributeError:
                    pass
                    
            if ifupdate:
                self.value = out.value
                self.history = out.history
            
            return out
        return _record
    return f if func is None else f(func)

def same_dtype(func=None, asfloat=False):
    """
    Decorator to assure output image has the same dtype as the input image. 
    This decorator is compatible with both ImgArray and LazyImgArray.

    Parameters
    ----------
    asfloat : bool, optional
        If input image should be converted to float first, by default False
    """    
    def f(func):
        @wraps(func)
        def _same_dtype(self, *args, **kwargs):
            dtype = self.dtype
            if asfloat and self.dtype.kind in "ui":
                self = self.as_float()
            out = func(self, *args, **kwargs)
            out = out.as_img_type(dtype)
            return out
        return _same_dtype
    return f if func is None else f(func)


def dims_to_spatial_axes(func):
    """
    Decorator to convert input `dims` to correct spatial axes. Compatible with ImgArray and
    LazyImgArray
    e.g.)
    dims=None (default) -> "yx" or "zyx" depend on the input image
    dims=2 -> "yx"
    dims=3 -> "zyx"
    dims="ty" -> "ty"
    """    
    @wraps(func)
    def _dims_to_spatial_axes(self, *args, **kwargs):
        dims = kwargs.get("dims", 
                          inspect.signature(func).parameters["dims"].default)
        if dims is None or dims == "":
            dims = len([a for a in "zyx" if a in self._axes])
            if dims not in (2, 3):
                raise ValueError(
                    f"Image spatial dimension must be 2 or 3, but {dims} was detected. If "
                    "image axes is not a standard one, such as 'tx' in kymograph, specify "
                    "the spatial axes by dims='tx' or dims='x'."
                    )
            
        if isinstance(dims, int):
            s_axes = "".join([a for a in "zyx" if a in self._axes])[-dims:]
        else:
            s_axes = str(dims)
        
        kwargs["dims"] = s_axes # update input
        return func(self, *args, **kwargs)
    
    return _dims_to_spatial_axes

def _safe_str(obj):
    try:
        if isinstance(obj, float):
            s = f"{obj:.3g}"
        else:
            s = str(obj)
        s = re.sub("\n", ";", s)
        if len(s) > 20:
            return str(type(obj))
        else:
            return s
    except Exception:
        return str(type(obj))

def make_history(funcname, args, kwargs):
    _args = list(map(_safe_str, args))
    _kwargs = [f"{_safe_str(k)}={_safe_str(v)}" for k, v in kwargs.items()]
    history = f"{funcname}({','.join(_args + _kwargs)})"
    return history