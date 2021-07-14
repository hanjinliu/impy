from ._skimage import *
from skimage.feature.corner import _symmetric_image
from skimage.feature.template import _window_sum_2d, _window_sum_3d
from scipy.signal import fftconvolve
import numpy as np

    
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
