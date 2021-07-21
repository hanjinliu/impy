import numpy as np
from skimage.feature.template import _window_sum_2d, _window_sum_3d
from ._skimage import *
import scipy
    
def ncc(img, template, bg):
    from scipy.signal import fftconvolve
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

def adjust_bin(img, binsize, check_edges, dims, all_axes):
    shape = []
    scale = []
    for i, a in enumerate(all_axes):
        s = img.shape[i]
        if a in dims:
            b = binsize
            if s % b != 0:
                if check_edges:
                    raise ValueError(f"Cannot bin axis {a} with length {s} by bin size {binsize}")
                else:
                    img = img[(slice(None),)*i + (slice(None, s//b*b),)]
        else:
            b = 1
        shape += [s//b, b]
        scale.append(1/b)
    
    shape = tuple(shape)
    return img, shape, scale

def _safe_sqrt(a, fill=0):
    out = np.full(a.shape, fill, dtype=np.float32)
    out = np.zeros_like(a)
    mask = a > 0
    out[mask] = np.sqrt(a[mask])
    return out

