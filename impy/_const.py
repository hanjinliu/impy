from __future__ import annotations
import psutil
from typing import Any, MutableMapping

memory = psutil.virtual_memory()

MAX_GB_LIMIT = memory.total / 2 * 1e-9

class GlobalConstant(MutableMapping[str, Any]):
    _const: dict[str, Any]
    
    def __init__(self, **kwargs):
        object.__setattr__(self, "_const", dict(**kwargs))
    
    def __len__(self) -> int:
        return len(self._const)
    
    def __iter__(self):
        raise StopIteration

    def __getitem__(self, k):
        return self._const[k]

    def __setitem__(self, k: str, v):
        k = k.upper()
        if k == "MAX_GB":
            if not isinstance(v, (int, float)):
                raise TypeError("MAX_GB must be float.")
            elif v > MAX_GB_LIMIT:
                raise ValueError(f"Cannot exceed {MAX_GB_LIMIT} GB.")
        elif k == "ID_AXIS":
            if not isinstance(v, str):
                raise TypeError("ID_AXIS must be str.")
            elif len(v) != 1:
                raise ValueError("ID_AXIS must be single character.")
        elif k == "FONT_SIZE_FACTOR":
            if not isinstance(v, (int, float)):
                raise TypeError("FONT_SIZE_FACTOR must be float.")
        elif k == "RESOURCE":
            from .array_api import xp
            if v == "numpy":
                xp.setNumpy()
            elif v == "cupy":
                xp.setCupy()
            else:
                raise ValueError("RESOURCES must be either 'numpy' or 'cupy'.")
        elif k == "SCHEDULER":
            import dask
            dask.config.set(scheduler=v)
        else:
            raise RuntimeError("Cannot set new keys.")
        
        self._const[k] = v
    
    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def __delitem__(self, v):
        raise RuntimeError("Cannot delete any items.")
    
    def __repr__(self):
        return (
            f"""
                  MAX_GB    : {self['MAX_GB']:.2f} GB
                 ID_AXIS    : {self['ID_AXIS']}
            FONT_SIZE_FACTOR: {self['FONT_SIZE_FACTOR']}
                 RESOURCE   : {self['RESOURCE']}
                SCHEDULER   : {self['SCHEDULER']}
            """
        )
    
    def asdict(self) -> dict[str, Any]:
        return self._const.copy()

Const = GlobalConstant(
    MAX_GB = MAX_GB_LIMIT/2,
    ID_AXIS = "p",
    FONT_SIZE_FACTOR = 1.0,
    RESOURCE = "numpy",
    SCHEDULER = "threads",
)

class SetConst:
    n_ongoing = 0
    _locked_keys: set[str] = set()
    _old_dict: dict[str, Any] = dict()
    
    def __init__(self, dict_: dict[str, Any] | None =None, **kwargs):
        dict_ = dict_ or {}
        dict_.update(kwargs)
        self._kwargs = dict_
    
    def __enter__(self):
        self.__class__.n_ongoing += 1
        if self.__class__.n_ongoing == 1:
            self.__class__._old_dict = Const.asdict()
        for k, v in self._kwargs.items():
            if k not in self.__class__._locked_keys:
                self.__class__._locked_keys.add(k)
                Const[k] = v

    def __exit__(self, exc_type, exc_value, traceback):
        self.__class__.n_ongoing -= 1
        if self.__class__.n_ongoing == 0:
            for k, v in self.__class__._old_dict.items():
                Const[k] = v
            self._locked_keys.clear()

def use(resource, import_error: bool = False):
    """
    Use a resource (numpy or cupy) in this context.

    Parameters
    ----------
    resource : str
        Resource to use.
    import_error : bool, default is False
        If false, resource will not be switched to cupy if not available.
        Raise ImportError if true.
    """
    if not import_error and resource=="cupy":
        try:
            import cupy
        except ImportError:
            resource = "numpy"
    return SetConst(RESOURCE=resource)
