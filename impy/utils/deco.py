from functools import wraps
import numpy as np
from .utilcls import Progress
import re

    
def record(append_history=True, record_label=False, only_binary=False, need_labels=False):
    """
    Record the name of ongoing function.
    """
    def _record(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if only_binary and self.dtype != bool:
                raise TypeError(f"Cannot run {func.__name__} with non-binary image.")
            if need_labels and not hasattr(self, "labels"):
                raise AttributeError(f"Function {func.__name__} needs labels."
                                    " Add labels to the image first.")
            
            with Progress(func.__name__):
                out = func(self, *args, **kwargs)
            
            temp = getattr(out, "temp", None)
                            
            if type(out) is np.ndarray and type(self) is not np.ndarray:
                out = out.view(self.__class__)
            
            # record history and update if needed
            ifupdate = kwargs.pop("update", False)
            
            if append_history:
                history = make_history(func.__name__, args, kwargs)
                if record_label:
                    out.labels._set_info(self.labels, history)
                else:
                    out._set_info(self, history)
                    
            ifupdate and self._update(out)
            
            # if temporary item exists
            if temp is not None:
                out.temp = temp
            return out
        return wrapper
    return _record


def same_dtype(asfloat=False):
    """
    Decorator to assure output image has the same dtype as the input
    image. 

    Parameters
    ----------
    asfloat : bool, optional
        If input image should be converted to float first, by default False
    """    
    def _same_dtype(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            dtype = self.dtype
            if asfloat and self.dtype.kind in "ui":
                self = self.astype(np.float32)
            out = func(self, *args, **kwargs)
            out = out.as_img_type(dtype)
            return out
        return wrapper
    return _same_dtype


def dims_to_spatial_axes(func):
    """
    Decorator to convert input `dims` to correct spatial axes.
    e.g.)
    dims=None (default) -> "yx" or "zyx" depend on the input image
    dims=2 -> "yx"
    dims=3 -> "zyx"
    dims="ty" -> "ty"
    """    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        dims = kwargs.get("dims", None)
        if dims is None or dims=="":
            dims = len([a for a in "zyx" if a in self._axes])
            if dims not in (2, 3):
                raise ValueError("Image must be 2 or 3 dimensional.")
            
        if isinstance(dims, int):
            s_axes = "".join([a for a in "zyx" if a in self.axes])[-dims:]
        elif isinstance(dims, str):
            s_axes = dims
        else:
            TypeError(f"'dims' must be None, int or str, but got {type(dims)}")
        kwargs["dims"] = s_axes # update input
        return func(self, *args, **kwargs)
    
    return wrapper

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