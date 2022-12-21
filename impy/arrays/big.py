from __future__ import annotations

from functools import wraps
from .lazy import LazyImgArray


def wrap_method(method):
    @wraps(method)
    def wrapped(self: LazyImgArray, *args, **kwargs):
        out = method(self, *args, **kwargs)
        if isinstance(out, LazyImgArray):
            out.release()
        return out
    return wrapped

class BigImgArray(LazyImgArray):
    affine = wrap_method(LazyImgArray.affine)
    erosion = wrap_method(LazyImgArray.erosion)
    dilation = wrap_method(LazyImgArray.dilation)
    opening = wrap_method(LazyImgArray.opening)
    closing = wrap_method(LazyImgArray.closing)
    gaussian_filter = wrap_method(LazyImgArray.gaussian_filter)
    spline_filter = wrap_method(LazyImgArray.spline_filter)
    median_filter = wrap_method(LazyImgArray.median_filter)
    mean_filter = wrap_method(LazyImgArray.mean_filter)
    convolve = wrap_method(LazyImgArray.convolve)
    edge_filter = wrap_method(LazyImgArray.edge_filter)
    laplacian_filter = wrap_method(LazyImgArray.laplacian_filter)
    kalman_filter = wrap_method(LazyImgArray.kalman_filter)
    fft = wrap_method(LazyImgArray.fft)
    ifft = wrap_method(LazyImgArray.ifft)
    power_spectra = wrap_method(LazyImgArray.power_spectra)
    tiled_lowpass_filter = wrap_method(LazyImgArray.tiled_lowpass_filter)
    proj = wrap_method(LazyImgArray.proj)
    binning = wrap_method(LazyImgArray.binning)
    track_drift = wrap_method(LazyImgArray.track_drift)
    drift_correction = wrap_method(LazyImgArray.drift_correction)
    radon = wrap_method(LazyImgArray.radon)
    wiener = wrap_method(LazyImgArray.wiener)
    lucy = wrap_method(LazyImgArray.lucy)
    as_uint8 = wrap_method(LazyImgArray.as_uint8)
    as_uint16 = wrap_method(LazyImgArray.as_uint16)
    as_float = wrap_method(LazyImgArray.as_float)
    as_img_type = wrap_method(LazyImgArray.as_img_type)
    __neg__ = wrap_method(LazyImgArray.__neg__)
    __add__ = wrap_method(LazyImgArray.__add__)
    __iadd__ = wrap_method(LazyImgArray.__iadd__)
    __sub__ = wrap_method(LazyImgArray.__sub__)
    __isub__ = wrap_method(LazyImgArray.__isub__)
    __mul__ = wrap_method(LazyImgArray.__mul__)
    __imul__ = wrap_method(LazyImgArray.__imul__)
    __truediv__ = wrap_method(LazyImgArray.__truediv__)
    __itruediv__ = wrap_method(LazyImgArray.__itruediv__)
    __gt__ = wrap_method(LazyImgArray.__gt__)
    __ge__ = wrap_method(LazyImgArray.__ge__)
    __lt__ = wrap_method(LazyImgArray.__lt__)
    __lt__ = wrap_method(LazyImgArray.__lt__)
    __eq__ = wrap_method(LazyImgArray.__eq__)
    __ne__ = wrap_method(LazyImgArray.__ne__)
    __mod__ = wrap_method(LazyImgArray.__mod__)
    __floordiv__ = wrap_method(LazyImgArray.__floordiv__)
    