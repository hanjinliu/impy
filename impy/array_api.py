from functools import wraps
import numpy as np

def cupy_dispatcher(function):
    @wraps(function)
    def func(*args, **kwargs):
        if xp.state == "cupy":
            args = (xp.asarray(a) if isinstance(a, np.ndarray) else a for a in args)
        out = function(*args, **kwargs)
        return xp.asnumpy(out)
    return func

# CUDA <= ver.8 does not have gradient
def _gradient(a, axis=None):
    out = np.gradient(a.get(), axis=axis)
    return xp.asarray(out)

class XP:
    def __init__(self):
        self.state = ""
        self._reset_namespace()
        self.setNumpy()

    def _reset_namespace(self):
        self._signal = None
        self._fft = None

    @property
    def signal(self):
        if self._signal is not None:
            return self._signal
        if self.state == "numpy":
            import scipy.signal as scipy_sig
            self._signal = scipy_sig
        elif self.state == "cupy":
            import cupyx.scipy.signal as cp_sig
            self._signal = cp_sig
        else:
            raise ValueError(self.state)
        return self._signal

    @property
    def fft(self):
        if self._fft is not None:
            return self._fft
        if self.state == "numpy":
            import scipy.fft as scipy_fft
            self._fft = scipy_fft
        elif self.state == "cupy":
            import cupyx.scipy.fft as cp_fft
            self._fft = cp_fft
        else:
            raise ValueError(self.state)
        return self._fft

    def setNumpy(self) -> None:
        from scipy import ndimage as scipy_ndi

        if self.state == "numpy":
            return

        self._reset_namespace()
        self._module = np
        self.linalg = np.linalg
        self.random = np.random
        self.ndi = scipy_ndi
        self.asnumpy = np.asarray
        self.asarray = np.asarray
        self.ndarray = np.ndarray
        self.empty = np.empty
        self.zeros = np.zeros
        self.empty_like = np.empty_like
        self.zeros_like = np.zeros_like
        self.ones = np.ones
        self.full = np.full
        self.array = np.array
        self.exp = np.exp
        self.sin = np.sin
        self.cos = np.cos
        self.tan = np.tan
        self.sqrt = np.sqrt
        self.mean = np.mean
        self.max = np.max
        self.min = np.min
        self.median = np.median
        self.sum = np.sum
        self.prod = np.prod
        self.std = np.std
        self.meshgrid = np.meshgrid
        self.indices = np.indices
        self.cumsum = np.cumsum
        self.arange = np.arange
        self.linspace = np.linspace
        self.real = np.real
        self.imag = np.imag
        self.conjugate = np.conjugate
        self.angle = np.angle
        self.abs = np.abs
        self.mod = np.mod
        self.fix = np.fix
        self.round = np.round
        self.gradient = np.gradient
        self.tensordot = np.tensordot
        self.concatenate = np.concatenate
        self.stack = np.stack
        self.unravel_index = np.unravel_index
        self.argmax = np.argmax
        self.argmin = np.argmin
        self.pad = np.pad
        self.isnan = np.isnan
        self.eye = np.eye

        self.state = "numpy"
        from ._const import Const
        Const["SCHEDULER"] = "threads"

    def setCupy(self) -> None:
        if self.state == "cupy":
            return
        import cupy as cp
        def cp_asnumpy(arr, dtype=None):
            out = cp.asnumpy(arr)
            if dtype is None:
                return out
            return out.astype(dtype)

        from cupyx.scipy import ndimage as cp_ndi
        from cupy import linalg as cp_linalg

        self._reset_namespace()
        self._module = cp
        self.linalg = cp_linalg
        self.random = cp.random
        self.ndi = cp_ndi
        self.asnumpy = cp_asnumpy
        self.asarray = cp.asarray
        self.ndarray = cp.ndarray
        self.empty = cp.empty
        self.zeros = cp.zeros
        self.ones = cp.ones
        self.empty_like = cp.empty_like
        self.zeros_like = cp.zeros_like
        self.full = cp.full
        self.array = cp.array
        self.exp = cp.exp
        self.sin = cp.sin
        self.cos = cp.cos
        self.tan = cp.tan
        self.sqrt = cp.sqrt
        self.mean = cp.mean
        self.max = cp.max
        self.min = cp.min
        self.median = cp.median
        self.sum = cp.sum
        self.prod = cp.prod
        self.std = cp.std
        self.meshgrid = cp.meshgrid
        self.indices = cp.indices
        self.cumsum = cp.cumsum
        self.arange = cp.arange
        self.linspace = cp.linspace
        self.real = cp.real
        self.imag = cp.imag
        self.conjugate = cp.conjugate
        self.angle = cp.angle
        self.abs = cp.abs
        self.mod = cp.mod
        self.fix = cp.fix
        self.round = cp.round
        try:
            self.gradient = cp.gradient
        except AttributeError:
            self.gradient = _gradient
        self.tensordot = cp.tensordot
        self.concatenate = cp.concatenate
        self.stack = cp.stack
        self.unravel_index = cp.unravel_index
        self.argmax = cp.argmax
        self.argmin = cp.argmin
        self.pad = cp.pad
        self.isnan = cp.isnan
        self.eye = cp.eye
        self.state = "cupy"

        from ._const import Const
        Const["SCHEDULER"] = "single-threaded"

xp = XP()
