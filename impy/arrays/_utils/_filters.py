from typing import Callable, TYPE_CHECKING
import numpy as np
from ._linalg import hessian_eigval
from impy.array_api import xp, cupy_dispatcher
from scipy import ndimage as scipy_ndi


__all__ = ["binary_erosion",
    "erosion",
    "binary_dilation",
    "dilation",
    "binary_opening",
    "opening",
    "binary_closing",
    "closing",
    "gaussian_filter",
    "gaussian_filter_fourier",
    "median_filter",
    "convolve",
    "white_tophat",
    "gaussian_laplace",
    "spline_filter",
    "kalman_filter",
    "fill_hole",
    "mean_filter",
    "phase_mean_filter",
    "std_filter",
    "coef_filter",
    "dog_filter",
    "dog_filter_fourier",
    "doh_filter",
    "gabor_filter",
    "skeletonize",
    "population",
    "ncc_filter",
]

def get_func(function_name) -> Callable[..., np.ndarray]:
    def func(*args, **kwargs):
        _f = getattr(xp.ndi, function_name, getattr(scipy_ndi, function_name))
        return cupy_dispatcher(_f)(*args, **kwargs)
    func.__name__ = function_name
    return func

binary_erosion = get_func("binary_erosion")
erosion = get_func("grey_erosion")
binary_dilation = get_func("binary_dilation")
dilation = get_func("grey_dilation")
binary_opening = get_func("binary_opening")
opening = get_func("grey_opening")
binary_closing = get_func("binary_closing")
closing = get_func("grey_closing")
gaussian_filter = get_func("gaussian_filter")
median_filter = get_func("median_filter")
min_filter = get_func("minimum_filter")
max_filter = get_func("maximum_filter")
spline_filter = get_func("spline_filter")
convolve = get_func("convolve")
white_tophat = get_func("white_tophat")
gaussian_laplace = get_func("gaussian_laplace")
_fourier_gaussian = get_func("fourier_gaussian")

if TYPE_CHECKING:
    binary_erosion = scipy_ndi.binary_erosion
    erosion = scipy_ndi.grey_erosion
    binary_dilation = scipy_ndi.binary_dilation
    dilation = scipy_ndi.grey_dilation
    binary_opening = scipy_ndi.binary_opening
    opening = scipy_ndi.grey_opening
    binary_closing = scipy_ndi.binary_closing
    closing = scipy_ndi.grey_closing
    gaussian_filter = scipy_ndi.gaussian_filter
    median_filter = scipy_ndi.median_filter
    min_filter = scipy_ndi.minimum_filter
    max_filter = scipy_ndi.maximum_filter
    spline_filter = scipy_ndi.spline_filter
    convolve = scipy_ndi.convolve
    white_tophat = scipy_ndi.white_tophat
    gaussian_laplace = scipy_ndi.gaussian_laplace
    _fourier_gaussian = scipy_ndi.fourier_gaussian

def kalman_filter(img_stack, gain, noise_var):
    # data is 3D or 4D
    img_stack = xp.asarray(img_stack)
    out = xp.empty_like(img_stack)
    spatial_shape = img_stack.shape[1:]
    for t, img in enumerate(img_stack):
        if t == 0:
            estimate = img
            predicted_var = xp.full(spatial_shape, noise_var)
        else:
            kalman_gain = predicted_var / (predicted_var + noise_var)
            estimate = gain*estimate + (1.0 - gain)*img + kalman_gain*(img - estimate)
            predicted_var *= 1 - kalman_gain
        out[t] = estimate
    return out

def gaussian_filter_fourier(img, sigma: float):
    img_ft = xp.fft.fftn(xp.asarray(img))
    out_ft = xp.ndi.fourier_gaussian(img_ft, sigma)
    return xp.fft.ifftn(out_ft).real

def fill_hole(img: np.ndarray, mask: np.ndarray):
    from skimage.morphology import reconstruction
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.max()
    return reconstruction(seed, mask, method="erosion")

def mean_filter(img, selem, mode="reflect", cval=0.0):
    return convolve(img, selem/np.sum(selem), mode=mode, cval=cval)

def phase_mean_filter(img: np.ndarray, selem, a, mode="reflect", cval=0.0):
    out = xp.empty(img.shape, dtype=np.complex64)
    xp.exp(1j*a*img, out=out)
    convolve(out, selem, output=out, mode=mode, cval=cval)
    return xp.angle(out)/a

def std_filter(data, selem, mode="reflect", cval=0.0):
    selem = selem / np.sum(selem)
    x1 = convolve(data, selem, mode=mode, cval=cval)
    x2 = convolve(data**2, selem, mode=mode, cval=cval)
    std_img = _safe_sqrt(xp.asnumpy(x2 - x1**2), fill=0)
    return std_img

def coef_filter(data, selem, mode="reflect", cval=0.0):
    selem = selem / np.sum(selem)
    x1 = convolve(data, selem, mode=mode, cval=cval)
    x2 = convolve(data**2, selem, mode=mode, cval=cval)
    out = _safe_sqrt(xp.asnumpy(x2 - x1**2), fill=0)/xp.asnumpy(x1)
    return out

def dog_filter(img, low_sigma, high_sigma, mode="reflect", cval=0.0):
    filt_l = gaussian_filter(img, low_sigma, mode=mode, cval=cval)
    filt_h = gaussian_filter(img, high_sigma, mode=mode, cval=cval)
    return filt_l - filt_h

def dog_filter_fourier(img, low_sigma: float, high_sigma: float):
    img_ft = xp.fft.fftn(xp.asarray(img))
    filt_l = xp.ndi.fourier_gaussian(img_ft, low_sigma)
    filt_h = xp.ndi.fourier_gaussian(img_ft, high_sigma)
    return xp.fft.ifftn(filt_l - filt_h).real

def doh_filter(img, sigma, pxsize):
    eigval = hessian_eigval(img, sigma, pxsize)
    eigval[eigval>0] = 0
    det = xp.abs(xp.prod(eigval, axis=-1))
    return det

def gabor_filter(img: np.ndarray, ker: np.ndarray, mode="reflect", cval=0.0):
    out = xp.empty(img.shape, dtype=np.complex64)
    out.real[:] = convolve(img, ker.real, mode=mode, cval=cval)
    out.imag[:] = convolve(img, ker.imag, mode=mode, cval=cval)
    return out


def skeletonize(img, selem):
    from skimage import morphology
    skl = morphology.skeletonize_3d(img)
    if selem is not None:
        skl = morphology.binary_dilation(skl, selem)
    return skl

def population(img, selem):
    from skimage.filters import rank
    return rank.pop(img, selem, mask=img)

# Essentially identical to skimage.feature.match_template
# See skimage/feature/template.py
def ncc_filter(img: np.ndarray, template: np.ndarray, bg=0, mode="constant"):
    from scipy.signal import fftconvolve
    from skimage.feature import template as skimage_template
    ndim = template.ndim
    _win_sum = skimage_template._window_sum_2d if ndim == 2 else skimage_template._window_sum_3d
    pad_width = [(w, w) for w in template.shape]
    padimg = np.pad(img, pad_width=pad_width, mode=mode, constant_values=bg)

    corr = fftconvolve(padimg, template[(slice(None,None,-1),)*ndim], mode="valid")[(slice(1,-1,None),)*ndim]

    win_sum1 = _win_sum(padimg, template.shape)
    win_sum2 = _win_sum(padimg**2, template.shape)

    template_mean = np.mean(template)
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean)**2)

    var = (win_sum2 - win_sum1**2/template_volume) * template_ssd

    # zero division happens when perfectly matched
    response = np.ones_like(corr)
    mask = var > np.finfo(np.float32).eps
    response[mask] = (corr - win_sum1 * template_mean)[mask] / _safe_sqrt(var, fill=np.inf)[mask]
    slices = []
    for i in range(ndim):
        d0 = (template.shape[i] - 1) // 2
        d1 = d0 + img.shape[i]
        slices.append(slice(d0, d1))
    out = response[tuple(slices)]
    return out

def _safe_sqrt(a: np.ndarray, fill=0):
    out = np.full(a.shape, fill, dtype=np.float32)
    out = np.zeros_like(a)
    mask = a > 0
    out[mask] = np.sqrt(a[mask])
    return out
