
import numpy as np
from ._skimage import *

def kalman_filter(img, gain, noise_var):
    # data is 3D or 4D
    out = np.empty_like(img)
    spatial_shape = img.shape[1:]
    for t, img in enumerate(img):
        if t == 0:
            estimate = img
            predicted_var = np.full(spatial_shape, noise_var)
        else:
            kalman_gain = predicted_var / (predicted_var + noise_var)
            estimate = gain*estimate + (1.0 - gain)*img + kalman_gain*(img - estimate)
            predicted_var *= 1 - kalman_gain
        out[t] = estimate
    return out

def fill_hole(img, mask):
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.max()
    return skmorph.reconstruction(seed, mask, method="erosion")

def mean_filter(img, selem):
    return ndi.convolve(img, selem/np.sum(selem))

def phase_mean_filter(img, selem, a):
    out = np.empty_like(img, dtype=np.complex64)
    np.exp(1j*a*img, out=out)
    ndi.convolve(out, selem, output=out)
    return np.angle(out)/a

def std_filter(data, selem):
    selem = selem / np.sum(selem)
    x1 = ndi.convolve(data, selem)
    x2 = ndi.convolve(data**2, selem)
    std_img = _safe_sqrt(x2 - x1**2, fill=0)
    return std_img

def coef_filter(data, selem):
    selem = selem / np.sum(selem)
    x1 = ndi.convolve(data, selem)
    x2 = ndi.convolve(data**2, selem)
    out = _safe_sqrt(x2 - x1**2, fill=0)/x1
    return out
    
def dog_filter(img, low_sigma, high_sigma):
    return ndi.gaussian_filter(img, low_sigma) - ndi.gaussian_filter(img, high_sigma)

def doh_filter(img, sigma, pxsize):
    hessian_elements = skfeat.hessian_matrix(img, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = skfeat.corner._symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    eigval[eigval>0] = 0
    det = np.abs(np.product(eigval, axis=-1))
    return det


def gabor_filter(img, ker):
    out = np.empty_like(img, dtype=np.complex64)
    out.real[:] = ndi.convolve(img, ker.real)
    out.imag[:] = ndi.convolve(img, ker.imag)
    return out


def skeletonize(img, selem):
    skl = skmorph.skeletonize_3d(img)
    if selem is not None:
        skl = skmorph.binary_dilation(skl, selem)
    return skl

def population(img, selem):
    return skfil.rank.pop(img, selem, mask=img)
    
def _safe_sqrt(a, fill=0):
    out = np.full(a.shape, fill, dtype=np.float32)
    out = np.zeros_like(a)
    mask = a > 0
    out[mask] = np.sqrt(a[mask])
    return out
