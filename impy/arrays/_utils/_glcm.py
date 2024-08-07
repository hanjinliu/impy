import numpy as np
import scipy
from impy.collections import DataDict

__all__ = ["glcm_props_", "check_glcm"]

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
    return scipy.stats.entropy(prob, axis=(0,1))

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


def glcm_props_(data, distances, angles, levels, radius, properties):
    outshape = (len(distances), len(angles)) + \
        (data.shape[0]-2*radius, data.shape[1]-2*radius)

    propout = DataDict()
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

    from ._process_numba import _calc_glcm

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

def check_glcm(self, bins, rescale_max):
    if bins is None:
        if self.dtype == bool:
            bins = 2
        else:
            bins = 256
    elif bins > 256:
        raise ValueError("`bins` must be smaller than 256.")

    if self.dtype == np.uint16:
        self = self.as_uint8()
    elif self.dtype == np.uint8:
        pass
    else:
        raise TypeError(f"Cannot calculate comatrix of {self.dtype} image.")

    imax = np.iinfo(self.dtype).max

    if rescale_max:
        scale = int(imax/self.max())
        self *= scale

    if (imax+1) % bins != 0 or bins > imax+1:
        raise ValueError(f"`bins` must be a divisor of {imax+1} (max value of {self.dtype}).")
    self = self // ((imax+1) // bins)

    return self, bins, rescale_max
