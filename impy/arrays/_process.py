from ._skimage import *
from skimage.feature.corner import _symmetric_image
from skimage.feature.template import _window_sum_2d, _window_sum_3d
from scipy.signal import fftconvolve
import numpy as np

def directional_median_(args):
    sl, data, radius = args
    diam = 2*radius + 1
    directional_median_kernel = directional_median_kernel_2d if data.ndim == 2 else directional_median_kernel_3d
    kernels = directional_median_kernel(radius*2 + 1)
    data_var = np.stack([ndi.convolve(data**2, ker/diam, mode="reflect") - 
                         ndi.convolve(data, ker/diam, mode="reflect")**2 for ker in kernels])
    min_vars = np.argmin(data_var, axis=0)
    data_med = [ndi.median_filter(data, footprint=ker, mode="reflect") for ker in kernels]
    out = np.empty_like(data)
    for d in np.arange(len(kernels)):
        out = np.where(min_vars==d, data_med[d], out)
    
    return sl, out

def directional_median_kernel_2d(size):
    k1 = np.ones((1, size), dtype=np.uint8) # -
    k2 = np.eye(size)                       # \
    k3 = k1.T                               # |
    k4 = np.fliplr(k2)                      # /
    return [k1, k2, k3, k4]

def directional_median_kernel_3d(size):
    raise NotImplementedError
    
def hessian_eigh_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return sl, eigval, eigvec

def hessian_eigval_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return sl, eigval

def structure_tensor_eigh_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(tensor)
    return sl, eigval, eigvec

def structure_tensor_eigval_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(tensor)
    return sl, eigval

def label_(args):
    sl, data, connectivity = args
    labels = skmes.label(data, background=0, connectivity=connectivity)
    return sl, labels

    
def ncc_(img, template, bg):
    ndim = template.ndim
    _win_sum = _window_sum_2d if ndim == 2 else _window_sum_3d
    pad_width = [(w, w) for w in template.shape]
    padimg = np.pad(img, pad_width=pad_width, mode="constant", constant_values=bg)
    
    corr = fftconvolve(padimg, template[(slice(None,None,-1),)*ndim], mode="valid")[(slice(1,-1,None),)*ndim]
    
    win_sum1 = _win_sum(padimg, template.shape)
    win_sum2 = _win_sum(padimg**2, template.shape)
    
    template_mean = np.mean(template)
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean)**2)
    
    var = (win_sum2 - win_sum1**2/template_volume) * template_ssd
    response = (corr - win_sum1 * template_mean) / _safe_sqrt(var, fill=np.inf)
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

def _safe_div(a, b, eps=1e-8):
    out = np.zeros(a.shape, dtype=np.float32)
    mask = b > eps
    out[mask] = a[mask]/b[mask]
    return out