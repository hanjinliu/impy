
from functools import wraps
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

def cupy_dispatcher(function):
    @wraps(function)
    def func(*args, **kwargs):
        if xp.state == "cupy":
            args = (xp.asarray(a) if isinstance(a, np.ndarray) else a for a in args)
        out = function(*args, **kwargs)
        return out
    return func

from types import ModuleType
from scipy import ndimage as scipy_ndi
from typing import Callable

class XP:
    fft: ModuleType
    linalg: ModuleType
    random: ModuleType
    ndi: ModuleType
    asnumpy: Callable[[ArrayLike, DTypeLike], np.ndarray]
    asarray: Callable[[ArrayLike], ArrayLike]
    ndarray: type
    state: str
    
    def __init__(self):
        self.setNumpy()
    
    def __getattr__(self, key: str):
        return getattr(self._module, key)
    
    def setNumpy(self) -> None:
        self._module = np
        self.fft = np.fft
        self.linalg = np.linalg
        self.random = np.random
        self.ndi = scipy_ndi
        self.asnumpy = np.asarray
        self.asarray = np.asarray
        self.ndarray = np.ndarray
        self.state = "numpy"
        from ._const import Const
        Const["SCHEDULER"] = "threads"
    
    def setCupy(self) -> None:
        import cupy as cp
        def cp_asnumpy(arr, dtype=None):
            out = cp.asnumpy(arr)
            if dtype is None:
                return out
            return out.astype(dtype)
        from cupyx.scipy import fft as cp_fft
        from cupyx.scipy import ndimage as cp_ndi
        from cupy import linalg as cp_linalg
        from cupy import ndarray as cp_ndarray
        
        self._module = cp
        self.fft = cp_fft
        self.linalg = cp_linalg
        self.random = cp.random
        self.ndi = cp_ndi
        self.asnumpy = cp_asnumpy
        self.asarray = cp.asarray
        self.ndarray = cp_ndarray
        self.state = "cupy"
        
        from ._const import Const
        Const["SCHEDULER"] = "single-threaded"

xp = XP()
    