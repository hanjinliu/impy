from __future__ import annotations
import re
from functools import lru_cache
from .._types import Slices

__all__ = ["str_to_slice", "axis_targeted_slicing"]

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
def axis_targeted_slicing(ndim: int, axes: tuple[str, ...], string: str) -> Slices:
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
    sl_list = [slice(None)]*ndim
    
    for k in keylist:
        if k.count("=") != 1:
            raise ValueError(f"Informal axis-targeted slicing: {k}")
        axis, sl_str = k.split("=")
        i = axes.index(axis)
        if i < 0:
            raise ValueError(f"Axis '{axis}' does not exist ({axes}).")
        try:
            sl_list[i] = str_to_slice(sl_str)
        except ValueError:
            raise ValueError(f"Informal axis-targeted slicing: {string}")
    
    return tuple(sl_list)