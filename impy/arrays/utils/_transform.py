import numpy as np
from ._skimage import sktrans
import scipy

def compose_affine_matrix(scale=None, translation=None, rotation=None, shear=None):    
    params = list(param for param in (scale, rotation, shear, translation) if param is not None)
    ndim = len(params[0])
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