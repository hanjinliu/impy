from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Literal, TypeVar, overload
from functools import wraps
import numpy as np
import inspect
from ..array_api import xp

if TYPE_CHECKING:
    from ..arrays import LazyImgArray, ImgArray
    from ..arrays.axesmixin import AxesMixin
    from typing_extensions import ParamSpec
    _P = ParamSpec("_P")
    _R = TypeVar("_R")

__all__ = [
    "check_input_and_output",
    "check_input_and_output_lazy",
    "same_dtype",
    "dims_to_spatial_axes",
]


@overload
def check_input_and_output(
    func: Literal[None],
    *, 
    inherit_label_info: bool = False,
    only_binary: bool = False,
    need_labels: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...

@overload
def check_input_and_output(
    func: Callable[_P, _R],
    *, 
    inherit_label_info: bool = False,
    only_binary: bool = False,
    need_labels: bool = False,
) -> Callable[_P, _R]:
    ...

def check_input_and_output(
    func=None,
    *, 
    inherit_label_info=False,
    only_binary=False,
    need_labels=False
):
    def f(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def _func(self: ImgArray, *args, **kwargs):
            # check requirements of the ongoing function.
            if only_binary and self.dtype != bool:
                raise TypeError(
                    f"Cannot run {func.__name__!r} with non-binary image."
                )
            if need_labels and not self.labels is not None:
                raise ValueError(
                    f"Function {func.__name__!r} needs labels. Add labels to the "
                    "image first."
                )
            
            out = func(self, *args, **kwargs)

            if type(out) in (np.ndarray, xp.ndarray):
                out = xp.asnumpy(out).view(self.__class__)

            ifupdate = kwargs.pop("update", False)
            if inherit_label_info:
                out.labels._set_info(self.labels)
            try:
                out._set_info(self)
            except AttributeError:
                pass
                    
            ifupdate and self._update(out)

            return out
        return _func
    return f if func is None else f(func)

@overload
def check_input_and_output_lazy(func: Callable[_P, _R], *, only_binary: bool = False) -> Callable[_P, _R]:
    ...

@overload
def check_input_and_output_lazy(func: Literal[None], *, only_binary: bool = False) -> Callable[[Callable[_P, _R], Callable[_P, _R]]]:
    ...
    
def check_input_and_output_lazy(func=None, *, only_binary=False):
    def f(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def _record(self: LazyImgArray, *args, **kwargs):
            if only_binary and self.dtype != bool:
                raise TypeError(f"Cannot run {func.__name__} with non-binary image.")
            
            out = func(self, *args, **kwargs)
            
            from dask import array as da
            if isinstance(out, da.core.Array):
                out = self.__class__(out)
            
            ifupdate = kwargs.pop("update", False)
            try:
                out._set_info(self)
            except AttributeError:
                pass
                
            if ifupdate:
                self.value = out.value
            
            return out
        return _record
    return f if func is None else f(func)

@overload
def same_dtype(func: Callable[_P, _R], asfloat: bool = False) -> Callable[_P, _R]:
    ...

@overload
def same_dtype(func: Literal[None], asfloat: bool = False) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...
    
def same_dtype(func=None, asfloat: bool = False):
    """
    Decorator to assure output image has the same dtype as the input image. 
    This decorator is compatible with both ImgArray and LazyImgArray.

    Parameters
    ----------
    asfloat : bool, optional
        If input image should be converted to float first, by default False
    """    
    def f(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def _same_dtype(self: ImgArray, *args, **kwargs):
            dtype = self.dtype
            if asfloat and self.dtype.kind in "ui":
                self = self.as_float()
            out: ImgArray = func(self, *args, **kwargs)
            out = out.as_img_type(dtype)
            return out
        return _same_dtype
    return f if func is None else f(func)


def dims_to_spatial_axes(func: Callable[_P, _R]) -> Callable[_P, _R]:
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
    def _dims_to_spatial_axes(self: AxesMixin, *args, **kwargs):
        dims = kwargs.get(
            "dims", 
            inspect.signature(func).parameters["dims"].default
        )
        if dims is None or dims == "":
            dims = len([a for a in "zyx" if a in self.axes])
            if dims not in (2, 3):
                raise ValueError(
                    f"Image spatial dimension must be 2 or 3, but {dims} was detected. If "
                    "image axes is not a standard one, such as 'tx' in kymograph, specify "
                    "the spatial axes by dims='tx' or dims='x'."
                    )
            
        if isinstance(dims, int):
            s_axes = [a for a in "zyx" if a in self.axes][-dims:]
        else:
            s_axes = list(dims)
        
        kwargs["dims"] = s_axes # update input
        return func(self, *args, **kwargs)
    
    return _dims_to_spatial_axes
