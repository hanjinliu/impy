try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    asnumpy = xp.asnumpy
    from cupyx.scipy import fft as xp_fft
    from cupyx.scipy import ndimage as xp_ndi
    from cupy import linalg as xp_linalg
    from cupy import ndarray as xp_ndarray
    import numpy as np
    _as_cupy = lambda a: xp.asarray(a) if isinstance(a, np.ndarray) else a
    
else:
    asnumpy = xp.asarray
    try:
        from scipy import fft as xp_fft
    except ImportError:
        from scipy import fftpack as xp_fft
    from scipy import ndimage as xp_ndi
    from numpy import linalg as xp_linalg
    from numpy import ndarray as xp_ndarray
    _as_cupy = lambda a: a

from functools import wraps

def cupy_dispatcher(function):
    @wraps(function)
    def func(*args, **kwargs):
        args = map(_as_cupy, args)
        out = function(*args, **kwargs)
        return out
    return func