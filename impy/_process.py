from skimage import morphology as skmorph
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import restoration as skres
from skimage import measure as skmes
from skimage.feature.corner import _symmetric_image
from skimage import feature as skfeat
from scipy import ndimage as ndi
import numpy as np

def _affine(args):
    sl, data, mx, order = args
    return (sl, sktrans.warp(data, mx, order=order))

def _median(args):
    sl, data, selem = args
    return (sl, skfil.rank.median(data, selem))

def _mean(args):
    sl, data, selem = args
    return (sl, skfil.rank.mean(data, selem))

def _gaussian(args):
    sl, data, sigma = args
    return (sl, ndi.gaussian_filter(data, sigma))

def _entropy(args):
    sl, data, selem = args
    return (sl, skfil.rank.entropy(data, selem))

def _enhance_contrast(args):
    sl, data, selem = args
    return (sl, skfil.rank.enhance_contrast(data, selem))

def _difference_of_gaussian(args):
    sl, data, low_sigma, high_sigma = args
    return (sl, skfil.difference_of_gaussians(data, low_sigma, high_sigma))

def _rolling_ball(args):
    sl, data, radius, smooth = args
    if smooth:
        _, ref = _mean((sl, data, np.ones((3, 3))))
        back = skres.rolling_ball(ref, radius=radius)
        tozero = (back > data)
        back[tozero] = data[tozero]
    else:
        back = skres.rolling_ball(data, radius=radius)
    
    return (sl, data - back)

def _sobel(args):
    sl, data = args
    return (sl, skfil.sobel(data))
    
def _opening(args):
    sl, data, selem = args
    return (sl, skmorph.opening(data, selem))

def _binary_opening(args):
    sl, data, selem = args
    return (sl, skmorph.binary_opening(data, selem))

def _closing(args):
    sl, data, selem = args
    return (sl, skmorph.closing(data, selem))

def _binary_closing(args):
    sl, data, selem = args
    return (sl, skmorph.binary_closing(data, selem))

def _erosion(args):
    sl, data, selem = args
    return (sl, skmorph.erosion(data, selem))

def _binary_erosion(args):
    sl, data, selem = args
    return (sl, skmorph.binary_erosion(data, selem))

def _dilation(args):
    sl, data, selem = args
    return (sl, skmorph.dilation(data, selem))

def _binary_dilation(args):
    sl, data, selem = args
    return (sl, skmorph.binary_dilation(data, selem))

def _tophat(args):
    sl, data, selem = args
    return (sl, skmorph.white_tophat(data, selem))

def _skeletonize(args):
    sl, data = args
    return (sl, skmorph.skeletonize_3d(data))

def _hessian_eigh(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return (sl, eigval, eigvec)

def _hessian_eigval(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return (sl, eigval)

def _structure_tensor_eigh(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(tensor)
    return (sl, eigval, eigvec)

def _structure_tensor_eigval(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(tensor)
    return (sl, eigval)

def _label(args):
    sl, data, connectivity = args
    labels = skmes.label(data, background=0, connectivity=connectivity)
    return (sl, labels)

def _distance_transform_edt(args):
    sl, data = args
    return (sl, ndi.distance_transform_edt(data))

def _fill_hole(args):
    sl, data, mask = args
    seed = np.copy(data)
    seed[1:-1, 1:-1] = data.max()
    return (sl, skmorph.reconstruction(seed, mask, method="erosion"))