import numpy as np
from skimage.feature.corner import _symmetric_image
from ._skimage import skfeat
from ..._const import Const

if Const["RESOURCE"] == "cupy":
    from ..._cupy import cupy as cp
    def eigh(a):
        a = cp.asarray(a, dtype=a.dtype)
        val, vec = cp.linalg.eigh(a)
        return val.get(), vec.get()

    def eigvalsh(a):
        a = cp.asarray(a, dtype=a.dtype)
        val = cp.linalg.eigvalsh(a)
        return val.get()
else:
    eigh = np.linalg.eigh
    eigvalsh = np.linalg.eigvalsh

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
    eigval = eigvalsh(hessian)
    return eigval

def _solve_hermitian_eigs(elems, pxsize):
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(elems)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = eigh(hessian)
    return np.concatenate([eigval[..., np.newaxis], eigvec], axis=-1)

def eigs_post_process(eigs, axes):
    eigval = eigs[..., 0]
    eigvec = eigs[..., 1:]
    
    eigval.axes = str(axes) + "l"
    eigval = eigval.sort_axes()
    
    eigvec.axes = str(axes) + "rl"
    eigvec = eigvec.sort_axes()
    
    return eigval, eigvec