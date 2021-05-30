import numpy as np
from scipy import optimize as opt
from scipy.stats import entropy
from skimage import transform as sktrans

def affinefit(img, imgref, bins=256, order=1):
    as_3x3_matrix = lambda mtx: np.vstack((mtx.reshape(2,3), [0., 0., 1.]))
    
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
        mtx = sktrans.AffineTransform(matrix=as_3x3_matrix(mtx))
        img_transformed = sktrans.warp(img, mtx, order=order)
        return -normalized_mutual_information(img_transformed, imgref)
    
    mtx0 = np.array([[1., 0., 0.],
                     [0., 1., 0.]]) # aberration makes little difference
    
    result = opt.minimize(cost_nmi, mtx0, args=(np.asarray(img), np.asarray(imgref)), method="Powell")
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