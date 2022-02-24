import numpy as np
from ._skimage import *
from ._linalg import hessian_eigval
               
__all__ = ["binary_erosion",
           "erosion"
           "binary_dilation",
           "dilation",
           "binary_opening",
           "opening",
           "binary_closing",
           "closing",
           "gaussian_filter", 
           "median_filter",
           "convolve",
           "white_tophat",
           "gaussian_laplace",
           "kalman_filter",
           "fill_hole",
           "mean_filter",
           "phase_mean_filter",
           "std_filter",
           "coef_filter",
           "dog_filter",
           "doh_filter",
           "gabor_filter",
           "skeletonize",
           "population",
           "ncc_filter",
           ]


from ..._cupy import xp, cupy_dispatcher
from scipy import ndimage as scipy_ndi

def get_func(function_name):
    if hasattr(xp.ndi, function_name):
        _func = getattr(xp.ndi, function_name)    
        func = cupy_dispatcher(_func)
    else:
        func = getattr(scipy_ndi, function_name)
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
convolve = get_func("convolve")
white_tophat = get_func("white_tophat")
gaussian_laplace = get_func("gaussian_laplace")


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

def fill_hole(img, mask):
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.max()
    return skimage.morphology.reconstruction(seed, mask, method="erosion")

def mean_filter(img, selem):
    return convolve(img, selem/np.sum(selem))

def phase_mean_filter(img, selem, a):
    out = xp.empty(img.shape, dtype=xp.complex64)
    xp.exp(1j*a*img, out=out)
    convolve(out, selem, output=out)
    return xp.angle(out)/a

def std_filter(data, selem):
    selem = selem / np.sum(selem)
    x1 = convolve(data, selem)
    x2 = convolve(data**2, selem)
    std_img = _safe_sqrt(xp.asnumpy(x2 - x1**2), fill=0)
    return std_img

def coef_filter(data, selem):
    selem = selem / np.sum(selem)
    x1 = convolve(data, selem)
    x2 = convolve(data**2, selem)
    out = _safe_sqrt(xp.asnumpy(x2 - x1**2), fill=0)/xp.asnumpy(x1)
    return out
    
def dog_filter(img, low_sigma, high_sigma):
    filt_l = gaussian_filter(img, low_sigma)
    filt_h = gaussian_filter(img, high_sigma)
    return filt_l - filt_h

def doh_filter(img, sigma, pxsize):
    eigval = hessian_eigval(img, sigma, pxsize)
    eigval[eigval>0] = 0
    det = xp.abs(xp.prod(eigval, axis=-1))
    return det

def gabor_filter(img, ker):
    out = xp.empty(img.shape, dtype=np.complex64)
    out.real[:] = convolve(img, ker.real)
    out.imag[:] = convolve(img, ker.imag)
    return out


def skeletonize(img, selem):
    skl = skimage.morphology.skeletonize_3d(img)
    if selem is not None:
        skl = skimage.morphology.binary_dilation(skl, selem)
    return skl

def population(img, selem):
    return skfil.rank.pop(img, selem, mask=img)
    
def ncc_filter(img, template, bg, mode="constant"):
    from scipy.signal import fftconvolve
    ndim = template.ndim
    _win_sum = skfeat.template._window_sum_2d if ndim == 2 else skfeat.template._window_sum_3d
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
    mask = var > 0
    response[mask] = (corr - win_sum1 * template_mean)[mask] / _safe_sqrt(var, fill=np.inf)[mask]
    slices = []
    for i in range(ndim):
        d0 = (template.shape[i] - 1) // 2
        d1 = d0 + img.shape[i]
        slices.append(slice(d0, d1))
    out = response[tuple(slices)]
    return out
    
def _safe_sqrt(a, fill=0):
    out = np.full(a.shape, fill, dtype=np.float32)
    out = np.zeros_like(a)
    mask = a > 0
    out[mask] = np.sqrt(a[mask])
    return out
