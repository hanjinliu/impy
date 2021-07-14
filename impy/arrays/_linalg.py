import numpy as np
from dask import array as da
from ._skimage import *
from skimage.feature.corner import _symmetric_image

def structure_tensor_eigval(img, sigma, pxsize):
    tensor_elements = skfeat.structure_tensor(img, sigma, order="xy",
                                              mode="reflect")
    return _solve_hermitian(tensor_elements, pxsize, np.linalg.eigvalsh)

def structure_tensor_eigh(img, sigma, pxsize):
    tensor_elements = skfeat.structure_tensor(img, sigma, order="xy",
                                              mode="reflect")
    return _solve_hermitian(tensor_elements, pxsize, _eigh)

def hessian_eigval(img, sigma, pxsize):
    hessian_elements = skfeat.hessian_matrix(img, sigma=sigma, order="xy",
                                             mode="reflect")
    return _solve_hermitian(hessian_elements, pxsize, np.linalg.eigvalsh)

def hessian_eigh(img, sigma, pxsize):
    hessian_elements = skfeat.hessian_matrix(img, sigma=sigma, order="xy",
                                             mode="reflect")
    return _solve_hermitian(hessian_elements, pxsize, _eigh)


def _eigh(arr: da.core.Array):
    return da.apply_gufunc(np.linalg.eigh, "(i,j)->(i),(i,j)", arr)

def _solve_hermitian(elems, pxsize, solver):
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(elems)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    return solver(hessian)