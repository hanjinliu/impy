from __future__ import annotations
from functools import lru_cache
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..arrays.axesmixin import AxesMixin


def axes_included(img: AxesMixin, label: AxesMixin):
    """
    e.g.)
    img.axes = "tyx", label.axes = "yx" -> True
    img.axes = "tcyx", label.axes = "zyx" -> False
    
    """
    return img.axes.contains(label.axes)


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
    # key = np.asarray(key)
    for i, o in enumerate(axes):
        if o not in key_axes:
            key = np.stack([key]*(shape[i]), axis=i)
    return key

@lru_cache
def complement_axes(axes, all_axes="ptzcyx"):
    c_axes = ""
    for a in all_axes:
        if a not in axes:
            c_axes += a
    return c_axes


def switch_slice(axes, all_axes, ifin=np.newaxis, ifnot=":"):
    if ifnot == ":":
        ifnot = [slice(None)]*len(all_axes)
    elif not hasattr(ifnot, "__iter__"):
        ifnot = [ifnot]*len(all_axes)
        
    if not hasattr(ifin, "__iter__"):
        ifin = [ifin]*len(all_axes)
        
    sl = []
    for a, slin, slout in zip(all_axes, ifin, ifnot):
        if a in axes:
            sl.append(slin)
        else:
            sl.append(slout)
    sl = tuple(sl)
    return sl
