from skimage import morphology as skmorph
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import restoration as skres
from skimage import measure as skmes
from skimage.feature.corner import _symmetric_image
from skimage import feature as skfeat
from scipy import ndimage as ndi
import numpy as np

def affine_(args):
    sl, data, mx, order = args
    return (sl, sktrans.warp(data, mx, order=order))

def median_(args):
    sl, data, selem = args
    return (sl, skfil.rank.median(data, selem))

def mean_(args):
    sl, data, selem = args
    return (sl, skfil.rank.mean(data, selem))

def gaussian_(args):
    sl, data, sigma = args
    return (sl, ndi.gaussian_filter(data, sigma))

def entropy_(args):
    sl, data, selem = args
    return (sl, skfil.rank.entropy(data, selem))

def enhance_contrast_(args):
    sl, data, selem = args
    return (sl, skfil.rank.enhance_contrast(data, selem))

def difference_of_gaussian_(args):
    sl, data, low_sigma, high_sigma = args
    return (sl, skfil.difference_of_gaussians(data, low_sigma, high_sigma))

def rolling_ball_(args):
    sl, data, radius, smooth = args
    if smooth:
        _, ref = mean_((sl, data, np.ones((3, 3))))
        back = skres.rolling_ball(ref, radius=radius)
        tozero = (back > data)
        back[tozero] = data[tozero]
    else:
        back = skres.rolling_ball(data, radius=radius)
    
    return (sl, data - back)

def sobel_(args):
    sl, data = args
    return (sl, skfil.sobel(data))
    
def opening_(args):
    sl, data, selem = args
    return (sl, skmorph.opening(data, selem))

def binary_opening_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_opening(data, selem))

def closing_(args):
    sl, data, selem = args
    return (sl, skmorph.closing(data, selem))

def binary_closing_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_closing(data, selem))

def erosion_(args):
    sl, data, selem = args
    return (sl, skmorph.erosion(data, selem))

def binary_erosion_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_erosion(data, selem))

def dilation_(args):
    sl, data, selem = args
    return (sl, skmorph.dilation(data, selem))

def binary_dilation_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_dilation(data, selem))

def tophat_(args):
    sl, data, selem = args
    return (sl, skmorph.white_tophat(data, selem))

def skeletonize_(args):
    sl, data = args
    return (sl, skmorph.skeletonize_3d(data))

def hessian_eigh_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return (sl, eigval, eigvec)

def hessian_eigval_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return (sl, eigval)

def structure_tensor_eigh_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(tensor)
    return (sl, eigval, eigvec)

def structure_tensor_eigval_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(tensor)
    return (sl, eigval)

def label_(args):
    sl, data, connectivity = args
    labels = skmes.label(data, background=0, connectivity=connectivity)
    return (sl, labels)

def distance_transform_edt_(args):
    sl, data = args
    return (sl, ndi.distance_transform_edt(data))

def fill_hole_(args):
    sl, data, mask = args
    seed = np.copy(data)
    seed[1:-1, 1:-1] = data.max()
    return (sl, skmorph.reconstruction(seed, mask, method="erosion"))