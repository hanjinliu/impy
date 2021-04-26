from functools import wraps
import numpy as np
from .func import add_axes
import trackpy as tp

def check_value(__op__):
    def wrapper(self, value):
        if isinstance(value, np.ndarray):
            value = value.astype("float32")
            if self.ndim >= 3 and value.shape == self.sizesof("yx"):
                value = add_axes(self.axes, self.shape, value)
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot multiply or divide negative value.")

        out = __op__(self, value)
        return out
    return wrapper
    
def record(append_history=True, record_label=False):
    """
    Record the name of ongoing function.
    """
    def _record(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # temporary record ongoing function
            self.ongoing = func.__name__
            if record_label:
                label_axes = self.labels.axes
                
            out = func(self, *args, **kwargs)
            
            self.ongoing = None
            del self.ongoing
            
            temp = getattr(out, "temp", None)
            
            if record_label:
                self.labels.axes = label_axes
                
            # view as ImgArray etc. if possible
            try:
                out = out.view(self.__class__)
            except AttributeError:
                pass
            
            # record history and update if needed
            ifupdate = kwargs.pop("update", False)
            
            if append_history:
                _args = list(map(safe_str, args))
                _kwargs = [f"{safe_str(k)}={safe_str(v)}" for k, v in kwargs.items()]
                history = f"{func.__name__}({','.join(_args + _kwargs)})"
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
            if asfloat:
                self = self.astype("float32")
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
        if dims is None:
            dims = len(self.spatial_shape)
            if dims not in (2, 3):
                raise ValueError("Image must be 2 or 3 dimensional.")
            
        if isinstance(dims, int):
            s_axes = "yx" if dims==2 else "zyx"
        elif isinstance(dims, str):
            s_axes = dims
        else:
            TypeError(f"'dims' must be None, int or str, but got {type(dims)}")
        kwargs["dims"] = s_axes # update input
        return func(self, *args, **kwargs)
    
    return wrapper

def need_labels(func):
    """
    Decorator to assure input image has label.
    """    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "labels"):
            raise AttributeError(f"Function {func.__name__} needs labels."
                                 " Add labels to the image first.")
        out = func(self, *args, **kwargs)
        return out
    return wrapper


def safe_str(obj):
    try:
        s = str(obj)
        if len(s) > 20:
            return str(type(obj))
        else:
            return s
    except Exception:
        return str(type(obj))
    
def tp_no_verbose(func):
    """
    Temporary suppress logging in trackpy.
    """    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        tp.quiet(suppress=True)
        out = func(self, *args, **kwargs)
        tp.quiet(suppress=False)
        return out
    return wrapper