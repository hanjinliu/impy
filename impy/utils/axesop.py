from __future__ import annotations
import numpy as np
from ..axes import Axes, Axis, UndefAxis, AxisLike


def find_first_appeared(axes, include="", exclude=""):
    include = list(include)
    exclude = list(exclude)
    for a in axes:
        if a in include and not a in exclude:
            return a
    raise ValueError(f"Inappropriate axes: {axes}")

def add_axes(axes: Axes, shape: tuple[int, ...], key: np.ndarray, key_axes="yx"):
    """
    Stack `key` to make its shape key_axes-> axes.
    """
    key_axes = list(key_axes)
    if shape == key.shape:
        return key

    for i, o in enumerate(axes):
        if o not in key_axes:
            key = np.stack([key] * shape[i], axis=i)
    return key


def complement_axes(axes, all_axes="ptzcyx") -> list[AxisLike]:
    c_axes = []
    axes_list = list(axes)
    for a in all_axes:
        if a not in axes_list:
            c_axes.append(a)
    return c_axes


def switch_slice(axes, all_axes, ifin=np.newaxis, ifnot=":"):
    axes = list(axes)
    if ifnot == ":":
        ifnot = [slice(None)] * len(all_axes)
    elif not hasattr(ifnot, "__iter__"):
        ifnot = [ifnot] * len(all_axes)
        
    if not hasattr(ifin, "__iter__"):
        ifin = [ifin] * len(all_axes)
        
    sl = []
    for a, slin, slout in zip(all_axes, ifin, ifnot):
        if a in axes:
            sl.append(slin)
        else:
            sl.append(slout)
    sl = tuple(sl)
    return sl


def slice_axes(axes: Axes, key):
    ndim = len(axes)
    if isinstance(key, tuple):
        ndim += sum(k is None for k in key)
        rest = ndim - len(key)
        if any(k is ... for k in key):
            idx = key.index(...)
            _keys = key[:idx] + (slice(None),) * (rest + 1) + key[idx + 1:]
        else:
            _keys = key + (slice(None),) * rest
    elif isinstance(key, np.ndarray) or hasattr(key, "__array__"):
        if key.ndim == 1:
            new_axes = axes
        else:
            new_axes = [UndefAxis()] + axes[key.ndim:]
        return new_axes
    elif key is None:
        return [UndefAxis()] + axes
    elif key is ...:
        return axes
    else:
        _keys = (key,) +(slice(None),) * (ndim - 1)

    new_axes: list[Axis] = []
    list_idx: list[int] = []

    axes_iter = iter(axes)
    for sl in _keys:
        if sl is not None:
            a = next(axes_iter)
            if isinstance(sl, (slice, np.ndarray)):
                new_axes.append(a.slice_axis(sl))
            elif isinstance(sl, list):
                new_axes.append(a.slice_axis(sl))
                list_idx.append(a)
        else:
            new_axes.append(UndefAxis())  # new axis
        
    if len(list_idx) > 1:
        added = False
        out: list[Axis] = []
        for a in new_axes:
            if a not in list_idx:
                out.append(a)
            elif not added:
                out.append(UndefAxis())
                added = True
        new_axes = out

    return new_axes
