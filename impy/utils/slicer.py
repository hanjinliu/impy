from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
import math
from typing import Any, Mapping
from impy._types import Slices
from impy.axes import Slicer, Axes

__all__ = ["str_to_slice", "solve_slicer"]

class _Slicer(ABC):
    def str_to_slice(self, v: str) -> list[int] | slice | int:
        v = v.replace(" ", "")
        if "," in v:
            sl = sum((self._range_to_list(v) for v in v.split(",")), [])
        elif ":" in v:
            sl = slice(*map(self._int_or_none, v.split(":")))
        else:
            sl = self._int(v)
        return sl
    
    def _range_to_list(self, v: str) -> list[int]:    
        if ":" in v:
            s, e = v.split(":")
            return list(range(self._int(s), self._int(e)))
        else:
            return [self._int(v)]

    def _int_or_none(self, v: str) -> int | None:
        if v:
            return self._int(v)
        else:
            return None

    @abstractmethod
    def _int(self, v: str) -> int:
        """Convert a string to an integer."""

    
def str_to_slice(v: str) -> list[int] | slice | int:
    return ConstSlicer().str_to_slice(v)

class ConstSlicer(_Slicer):
    def _int(self, v: str) -> int:
        return int(v)

class EvalSlicer(_Slicer):
    def __init__(self, size: int):
        self._size = size

    def _int(self, v: str) -> int:
        ns = {
            "N": self._size,
            "int": int,
            "ceil": math.ceil,
            "floor": math.floor,
            "round": round,
            "__builtins__": {},
        }
        out = eval(v, ns)
        if not hasattr(out, "__index__"):
            raise ValueError(f"Invalid index: {v!r}.")
        return out.__index__()


def axis_targeted_slicing(
    axes: tuple[str, ...], 
    string: str,
    shape: tuple[int, ...] | None = None,
) -> Slices:
    """
    Make a conventional slices from an axis-targeted slicing string.

    Parameters
    ----------
    axes : str
        Axes of input ndarray.
    string : str
        Axis-targeted slicing string. If an axis that does not exist in `axes` is
        contained, this function will raise ValueError.

    Returns
    -------
    slices
    """    
    if "N" in string:
        return axis_targeted_slicing_eval(axes, string, shape)
    else:
        return axis_targeted_slicing_const(axes, string)

@lru_cache
def axis_targeted_slicing_const(axes: tuple[str, ...], string: str) -> Slices:
    keylist = string.replace(" ", "").split(";")
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
    
    return _dict_to_slice(dict_slicer, axes)

def axis_targeted_slicing_eval(
    axes: tuple[str, ...], 
    string: str, 
    shape: tuple[int, ...] | None = None,
) -> Slices:
    keylist = string.replace(" ", "").split(";")
    dict_slicer: dict[str, Any] = {}
    
    for k in keylist:
        if k.count("=") != 1:
            raise ValueError(f"Informal axis-targeted slicing: {k!r}.")
        axis, sl_str = k.split("=")
        try:
            size = shape[axes.index(axis)]
            sl = EvalSlicer(size).str_to_slice(sl_str)
        except ValueError:
            raise ValueError(f"Informal axis-targeted slicing: {k!r}.")
        else:
            dict_slicer[axis] = sl
    
    return _dict_to_slice(dict_slicer, axes)

def _dict_to_slice(sl: dict[str, Any], axes: tuple[str, ...]):
    sl_list = [slice(None)] * len(axes)
    
    for k, v in sl.items():
        idx = axes.index(k)
        sl_list[idx] = v
    
    return tuple(sl_list)

def solve_slicer(key: Any, axes: Axes, shape: tuple[int, ...] | None = None) -> Slices:
    if isinstance(key, str):
        key = axis_targeted_slicing(tuple(axes), key, shape)
    
    elif isinstance(key, (Mapping, Slicer)):
        key = axes.create_slice(key)
    
    return key
