from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from skimage.feature.corner import _symmetric_image
from ._skimage import skfeat
from impy.array_api import xp

if TYPE_CHECKING:
    from impy.arrays.imgarray import ImgArray
    from impy.axes import Axes

def eigh(a: np.ndarray):
    a = xp.asarray(a, dtype=a.dtype)
    val, vec = xp.linalg.eigh(a)
    return val, vec

def eigvalsh(a: np.ndarray):
    a = xp.asarray(a, dtype=a.dtype)
    val = xp.linalg.eigvalsh(a)
    return val

def structure_tensor_eigval(img: np.ndarray, sigma: float, pxsize: float):
    tensor_elements = skfeat.structure_tensor(
        img, sigma, mode="reflect"
    )
    return _solve_hermitian_eigval(tensor_elements, pxsize)

def structure_tensor_eigh(img: np.ndarray, sigma: float, pxsize: float):
    tensor_elements = skfeat.structure_tensor(
        img, sigma, mode="reflect"
    )
    return _solve_hermitian_eigs(tensor_elements, pxsize)

def hessian_eigval(img: np.ndarray, sigma: float, pxsize: float):
    hessian_elements = skfeat.hessian_matrix(
        img, sigma=sigma, mode="reflect", use_gaussian_derivatives=False,
    )
    return _solve_hermitian_eigval(hessian_elements, pxsize)

def hessian_eigh(img: np.ndarray, sigma: float, pxsize: float):
    hessian_elements = skfeat.hessian_matrix(
        img, sigma=sigma, mode="reflect", use_gaussian_derivatives=False,
    )
    return _solve_hermitian_eigs(hessian_elements, pxsize)


def _solve_hermitian_eigval(elems, pxsize) -> np.ndarray:
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(elems)
    hessian *= (pxsize.reshape(-1, 1) * pxsize.reshape(1, -1))
    eigval = eigvalsh(hessian)
    return eigval

def _solve_hermitian_eigs(elems, pxsize):
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(elems)
    hessian *= (pxsize.reshape(-1, 1) * pxsize.reshape(1, -1))
    eigval, eigvec = eigh(hessian)
    return np.concatenate([eigval[..., np.newaxis], eigvec], axis=-1)

def eigs_post_process(eigs: ImgArray, axes: Axes, self: ImgArray):
    eigval = eigs[..., 0]
    eigvec = eigs[..., 1:]
    eigval: ImgArray = np.moveaxis(eigval, -1, 0)
    eigvec: ImgArray = np.moveaxis(eigvec, [-2, -1], [0, 1])
    
    val_axes = ["base"] + axes
    vec_axes = ["dim", "base"] + axes
    nspatial = eigvec.shape[0]
    eigval._set_info(self, new_axes=val_axes)
    eigvec._set_info(self, new_axes=vec_axes)
    eigvec.axes["dim"].labels = tuple(map(str, eigvec.axes[-nspatial:]))
    return eigval, eigvec
