import numpy as np
import scipy
from collections import namedtuple
from ._skimage import sktrans
from ..._cupy import xp, xp_ndi

__all__ = ["compose_affine_matrix", 
           "decompose_affine_matrix",
           "affinefit", 
           "check_matrix",
           "warp"
           ]

AffineTransformationParameters = namedtuple(typename="AffineTransformationParameters", 
                                            field_names=["translation", "rotation", "scale", "shear"]
                                            )

def warp(img, matrix, cval=0, mode="constant", output_shape=None, order=1):
    img = xp.asarray(img, dtype=img.dtype)
    matrix = xp.asarray(matrix)
    out = xp_ndi.affine_transform(img, matrix, cval=cval, mode=mode, output_shape=output_shape, 
                                  order=order, prefilter=order>1)
    return out

def compose_affine_matrix(scale=None, translation=None, rotation=None, shear=None, ndim:int=2):
    # These two modules returns consistent matrix in the two dimensional case.
    if ndim == 2:
        af = sktrans.AffineTransform(scale=scale, translation=translation, rotation=rotation, shear=shear)
        mx = af.params
    else:
        from napari.utils.transforms import Affine
        if scale is None:
            scale = [1]*ndim
        elif np.isscalar(scale):
            scale = [scale]*ndim
        if translation is None:
            translation = [0]*ndim
        elif np.isscalar(translation):
            translation = [translation]*ndim
        
        af = Affine(scale=scale, translate=translation, rotate=rotation, shear=shear)
        mx = af.affine_matrix
    
    return mx


def decompose_affine_matrix(matrix:np.ndarray):
    ndim = matrix.shape[0] - 1
    if ndim == 2:
        af = sktrans.AffineTransform(matrix=matrix)
        out = AffineTransformationParameters(translation=af.translation, rotation=af.rotation, 
                                             scale=af.scale, shear=af.shear)
    else:
        from napari.utils.transforms import Affine
        af = Affine(affine_matrix=matrix)
        out = AffineTransformationParameters(translation=af.translate, rotation=af.rotate, 
                                             scale=af.scale, shear=af.shear)
    return out
        


def calc_corr(img0, img1, matrix, corr_func):
    """
    Calculate value of corr_func(img0, matrix(img1)).
    """
    img1_transformed = warp(img1, matrix)
    return corr_func(img0, img1_transformed)

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
        mtx = as_3x3_matrix(mtx)
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