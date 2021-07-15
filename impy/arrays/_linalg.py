import numpy as np
from ._skimage import *
from skimage.feature.corner import _symmetric_image

def structure_tensor_eigval(img, sigma, pxsize):
    tensor_elements = skfeat.structure_tensor(img, sigma, order="xy",
                                              mode="reflect")
    return _solve_hermitian_eigval(tensor_elements, pxsize)

def structure_tensor_eigh(img, sigma, pxsize):
    tensor_elements = skfeat.structure_tensor(img, sigma, order="xy",
                                              mode="reflect")
    return _solve_hermitian_eigs(tensor_elements, pxsize)

def hessian_eigval(img, sigma, pxsize):
    hessian_elements = skfeat.hessian_matrix(img, sigma=sigma, order="xy",
                                             mode="reflect")
    return _solve_hermitian_eigval(hessian_elements, pxsize)

def hessian_eigh(img, sigma, pxsize):
    hessian_elements = skfeat.hessian_matrix(img, sigma=sigma, order="xy",
                                             mode="reflect")
    return _solve_hermitian_eigs(hessian_elements, pxsize)


def _solve_hermitian_eigval(elems, pxsize):
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(elems)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return eigval

def _solve_hermitian_eigs(elems, pxsize):
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(elems)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return np.concatenate([eigval[..., np.newaxis], eigvec], axis=-1)

def eigs_post_process(eigs, axes):
    eigval = eigs[..., 0]
    eigvec = eigs[..., 1:]
    
    eigval.axes = str(axes) + "l"
    eigval = eigval.sort_axes()
    
    eigvec.axes = str(axes) + "rl"
    eigvec = eigvec.sort_axes()
    
    return eigval, eigvec