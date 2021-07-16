from __future__ import annotations
import numpy as np

def check_nd(x, ndim:int):
    if np.isscalar(x):
        x = (x,) * ndim
    elif len(x) != ndim:
        raise ValueError("length of parameter and dimension must match.")
    return x
    
def find_first_appeared(axes, include="", exclude=""):
    for a in axes:
        if a in include and not a in exclude:
            return a
    raise ValueError(f"Inappropriate axes: {axes}")

def del_axis(axes, axis) -> str:
    """
    axes: str or Axes object.
    axis: int.
    delete axis from axes.
    """
    new_axes = ""
    if isinstance(axis, int):
        axis = [axis]
    if axes is None:
        return None
    else:
        axes = str(axes)
    
    if isinstance(axis, (list, tuple)):
        for i, o in enumerate(axes):
            if i not in axis:
                new_axes += o
    elif isinstance(axis, str):
        new_axes = complement_axes(axis, axes)
        
    return new_axes

def add_axes(axes, shape, key, key_axes="yx"):
    """
    Stack `key` to make its shape key_axes-> axes.
    """    
    if shape == key.shape:
        return key
    key = np.asarray(key)
    for i, o in enumerate(axes):
        if o not in key_axes:
            key = np.stack([key]*(shape[i]), axis=i)
    return key




def complement_axes(axes, all_axes="ptzcyx"):
    c_axes = ""
    for a in all_axes:
        if a not in axes:
            c_axes += a
    return c_axes

def check_filter_func(f):
    if f is None:
        f = lambda x: True
    elif not callable(f):
        raise TypeError("`filt` must be callable.")
    return f


def largest_zeros(shape) -> np.ndarray:
    try:
        out = np.zeros(shape, dtype=np.uint64)
    except MemoryError:
        try:
            out = np.zeros(shape, dtype=np.uint32)
        except MemoryError:
            out = np.zeros(shape, dtype=np.uint16)
    return out
