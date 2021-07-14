import numpy as np
from ._skimage import *
from skimage.feature.corner import _symmetric_image

def structure_tensor_eigval(img, sigma, pxsize):
    tensor_elements = skfeat.structure_tensor(img, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(tensor)
    return eigval


def hessian_eigval(img, sigma, pxsize):
    hessian_elements = skfeat.hessian_matrix(img, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return eigval