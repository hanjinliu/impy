from __future__ import annotations
import re
from functools import lru_cache
from typing import Any, Mapping
from .._types import Slices
from ..axes import Slicer, Axes

__all__ = ["str_to_slice", "axis_targeted_slicing", "solve_slicer"]

def _range_to_list(v: str) -> list[int]:
    """
    "1,3,5" -> [1,3,5]
    "2,4:6,9" -> [2,4,5,6,9]
    """
    if ":" in v:
        s, e = v.split(":")
        return list(range(int(s), int(e)))
    else:
        return [int(v)]

def _int_or_none(v: str) -> int | None:
    if v:
        return int(v)
    else:
        return None
    
def str_to_slice(v: str) -> list[int] | slice | int:
    v = re.sub(" ", "", v)
    if "," in v:
        sl = sum((_range_to_list(v) for v in v.split(",")), [])
    elif ":" in v:
        sl = slice(*map(_int_or_none, v.split(":")))
    else:
        sl = int(v)
    return sl


@lru_cache
def axis_targeted_slicing(axes: tuple[str, ...], string: str) -> Slices:
    """
    Make a conventional slices from an axis-targeted slicing string.

    Parameters
    ----------
    ndim : int
        Number of dimension of the array which will be sliced.
    axes : str
        Axes of input ndarray.
    string : str
        Axis-targeted slicing string. If an axis that does not exist in `axes` is
        contained, this function will raise ValueError.

    Returns
    -------
    slices
    """    
    keylist = re.sub(" ", "", string).split(";")
    dict_slicer: dict[str, Any] = {}
    
    for k in keylist:
        if k.count("=") != 1:
            raise ValueError(f"Informal axis-targeted slicing: {k!r}.")
        axis, sl_str = k.split("=")
        try:
            sl = str_to_slice(sl_str)
        except ValueError:
            raise ValueError(f"Informal axis-targeted slicing: {k!r}.")
        else:
            dict_slicer[axis] = sl
    
    return dict_to_slice(dict_slicer, axes)

def dict_to_slice(sl: dict[str, Any], axes: tuple[str, ...]):
    sl_list = [slice(None)] * len(axes)
    
    for k, v in sl.items():
        idx = axes.index(k)
        sl_list[idx] = v
    
    return tuple(sl_list)

def solve_slicer(key: Any, axes: Axes):
    if isinstance(key, str):
        key = axis_targeted_slicing(tuple(axes), key)
    
    elif isinstance(key, (Mapping, Slicer)):
        key = axes.create_slice(key)
    
    return key