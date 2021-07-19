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


def affinefit(img, imgref, bins=256, order=1):
    as_3x3_matrix = lambda mtx: np.vstack((mtx.reshape(2,3), [0., 0., 1.]))
    from scipy.stats import entropy
    def normalized_mutual_information(img, imgref):
        """
        Y(A,B) = (H(A)+H(B))/H(A,B)
        See "Elegant SciPy"
        """
        hist, edges = np.histogramdd([np.ravel(img), np.ravel(imgref)], bins=bins)
        hist /= np.sum(hist)
        e1 = entropy(np.sum(hist, axis=0)) # Shannon entropy
        e2 = entropy(np.sum(hist, axis=1))
        e12 = entropy(np.ravel(hist)) # mutual entropy
        return (e1 + e2)/e12
    
    def cost_nmi(mtx, img, imgref):
        mtx = sktrans.AffineTransform(matrix=as_3x3_matrix(mtx))
        img_transformed = sktrans.warp(img, mtx, order=order)
        return -normalized_mutual_information(img_transformed, imgref)
    
    mtx0 = np.array([[1., 0., 0.],
                     [0., 1., 0.]]) # aberration makes little difference
    
    result = scipy.optimize.minimize(cost_nmi, mtx0, args=(np.asarray(img), np.asarray(imgref)),
                                     method="Powell")
    mtx_opt = as_3x3_matrix(result.x)
    return mtx_opt

def check_matrix(matrices):
    """
    Check Affine transformation matrix
    """    
    mtx = []
    for m in matrices:
        if np.isscalar(m): 
            if m == 1:
                mtx.append(m)
            else:
                raise ValueError(f"Only `1` is ok, but got {m}")
            
        elif m.shape != (3, 3) or not np.allclose(m[2,:2], 0):
            raise ValueError(f"Wrong Affine transformation matrix:\n{m}")
        
        else:
            mtx.append(m)
    return mtx