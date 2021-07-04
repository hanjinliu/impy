from __future__ import annotations
import numpy as np
import re

__all__ = ["str_to_slice", "key_repr", "axis_targeted_slicing"]

def _range_to_list(v:str) -> list[int]:
    """
    "1,3,5" -> [1,3,5]
    "2,4:6,9" -> [2,4,5,6,9]
    """
    if ":" in v:
        s, e = v.split(":")
        return list(range(int(s), int(e)))
    else:
        return [int(v)]

def int_or_None(v:str) -> int|None:
    if v:
        return int(v)
    else:
        return None
    
def str_to_slice(v:str):
    # check if this works
    v = re.sub(" ", "", v)
    if "," in v:
        sl = sum((_range_to_list(v) for v in v.split(",")), [])
    elif ":" in v:
        sl = slice(*map(int_or_None, v.split(":")))
    else:
        sl = int(v)
    return sl

def key_repr(key):
    keylist = []
        
    if isinstance(key, tuple):
        _keys = key
    elif hasattr(key, "__array__"):
        _keys = ("array",)
    else:
        _keys = (key,)
    
    for s in _keys:
        if isinstance(s, (slice, list, np.ndarray)):
            keylist.append("*")
        elif s is None:
            keylist.append("new")
        elif s is ...:
            keylist.append("...")
        else:
            keylist.append(str(s))
    
    return ",".join(keylist)

def axis_targeted_slicing(arr:np.ndarray, axes:str, string:str):
    """
    e.g. 't=3:, z=1:5', 't=1, z=:7'
    """
    keylist = re.sub(" ", "", string).split(";")
    sl_list = [slice(None)]*arr.ndim
    try:
        for k in keylist:
            # e.g. k = "t=4:7"
            axis, sl_str = k.split("=")
            sl_list[axes.find(axis)] = str_to_slice(sl_str)
    except ValueError:
        raise ValueError(f"Informal axis-targeted slicing: {string}")
    return tuple(sl_list)