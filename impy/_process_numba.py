from impy.utilcls import ArrayDict
import numpy as np
from trackpy.try_numba import try_numba_jit
from scipy.stats import entropy

__all__ = ["glcm_props_"]

@try_numba_jit(signature_or_function="void(uint8[:], f4[:], i8, i8, uint8[:,:], f4[:,:,:,:])", nopython=True)
def _calc_glcm(distances, angles, levels, radius, patch, glcm):
    glcm[:,:,:,:] = 0
    for a_idx in range(angles.size):
        angle = angles[a_idx]
        for d_idx in range(distances.size):
            d = distances[d_idx]
            offset_row = round(np.sin(angle) * d)
            offset_col = round(np.cos(angle) * d)
            start_row = max(0, -offset_row)
            end_row = min(2*radius+1, 2*radius+1 - offset_row)
            start_col = max(0, -offset_col)
            end_col = min(2*radius+1, 2*radius+1 - offset_col)
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    i = patch[r, c]
                    # compute the location of the offset pixel
                    row = r + offset_row
                    col = c + offset_col
                    j = patch[row, col]
                    if 0 <= i < levels and 0 <= j < levels:
                        glcm[i, j, d_idx, a_idx] += 1.

def glcm_props_(data, distances, angles, levels, radius, properties):
    outshape = (len(distances), len(angles)) + \
        (data.shape[0]-2*radius, data.shape[1]-2*radius)
        
    propdict = {"contrast": contrast_,
                "dissimilarity": dissimilarity_,
                "idm": idm_,
                "asm": asm_,
                "energy": energy_,
                "max": max_,
                "entropy": entropy_,
                "correlation": correlation_,
                "mean_ref": mean_ref_,
                "mean_neighbor": mean_neighbor_,
                "std_ref": std_ref_,
                "std_neighbor": std_neighbor_,
                }
        
    propout = ArrayDict()
    for prop in properties:
        if isinstance(prop, str):
            propout[prop] = np.empty(outshape, dtype=np.float32)
        else:
            propout[prop.__name__] = np.empty(outshape, dtype=np.float32)
    # placeholder
    glcm = np.empty((levels, levels, len(distances), len(angles)), dtype=np.float32)
    
    ref, nei = np.indices((levels, levels), dtype=np.float32)
    ref = ref[:, :, np.newaxis, np.newaxis]
    nei = nei[:, :, np.newaxis, np.newaxis]
    
    for x in range(outshape[3]):
        for y in range(outshape[2]):
            # calc glcm
            patch = data[y:y+2*radius+1, x:x+2*radius+1]
            _calc_glcm(distances, angles, levels, radius, patch, glcm)
            # calc props
            for prop in properties:
                if isinstance(prop, str):
                    propout[prop][:,:,y,x] = propdict[prop](glcm, ref, nei)
                else:
                    # only callable
                    propout[prop.__name__][:,:,y,x] = prop(glcm, ref, nei)
    
    return propout

def contrast_(glcm, ref, nei):
    return np.sum((ref-nei)**2 * glcm, axis=(0,1))

def dissimilarity_(glcm, ref, nei):
    return np.sum(glcm * np.abs(ref-nei), axis=(0,1))

def asm_(glcm, ref, nei):
    return np.sum(glcm**2, axis=(0,1))

def idm_(glcm, ref, nei):
    return np.sum(glcm/(1+(ref-nei)**2), axis=(0,1))
    
def energy_(glcm, ref, nei):
    return np.sqrt(np.sum(glcm**2, axis=(0,1)))

def max_(glcm, ref, nei):
    return np.max(glcm, axis=(0,1))

def entropy_(glcm, ref, nei):
    prob = glcm / np.sum(glcm, axis=(0,1), keepdims=True)
    return entropy(prob, axis=(0,1))

def correlation_(glcm, ref, nei):
    diffy = ref - np.sum(glcm * ref, axis=(0,1))
    diffx = nei - np.sum(glcm * nei, axis=(0,1))
    
    stdy = np.sqrt(np.sum(glcm*diffy**2, axis=(0,1)))
    stdx = np.sqrt(np.sum(glcm*diffx**2, axis=(0,1)))
    cov = np.sum(glcm*diffx*diffy, axis=(0,1))
    
    out = np.empty(glcm.shape[2:], dtype=np.float32)
    mask_0 = np.logical_or(stdx<1e-15, stdy<1e-15)
    mask_1 = ~mask_0
    out[mask_0] = 1

    # handle the standard case
    out[mask_1] = cov[mask_1] / (stdx[mask_1] * stdy[mask_1])
    return out

def mean_ref_(glcm, ref, nei):
    return np.sum(glcm*ref, axis=(0,1))

def mean_neighbor_(glcm, ref, nei):
    return np.sum(glcm*nei, axis=(0,1))

def std_ref_(glcm, ref, nei):
    return np.std(glcm*ref, axis=(0,1))

def std_neighbor_(glcm, ref, nei):
    return np.std(glcm*nei, axis=(0,1))

