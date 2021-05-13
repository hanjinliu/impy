from numba import jit
import numpy as np

@jit("void(uint8[:], f4[:], i8, i8, uint16[:,:], f4[:,:,:,:])", nopython=True)
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
                        glcm[i, j, d_idx, a_idx] += 1

def glcm_filter_(data, distances, angles, levels, radius):
    outshape = (len(distances), len(angles)) + \
        (data.shape[0]-2*radius, data.shape[1]-2*radius)
    # four properties
    contrast = np.empty(outshape, dtype=np.float32)
    dissimilarity = np.empty(outshape, dtype=np.float32)
    homogeneity = np.empty(outshape, dtype=np.float32)
    energy = np.empty(outshape, dtype=np.float32)
    # placeholder
    glcm = np.empty((levels, levels, len(distances), len(angles)), dtype=np.float32)
    
    dyx = np.empty(glcm.shape, dtype=np.uint16)
    for j in range(glcm.shape[0]):
        for i in range(glcm.shape[1]):
            dyx[j,i,:,:] = abs(j - i)
    
    for x in range(outshape[3]):
        for y in range(outshape[2]):
            # calc glcm
            patch = data[y:y+2*radius+1, x:x+2*radius+1]
            _calc_glcm(distances, angles, levels, radius, patch, glcm)
            # calc props
            contrast[:,:,y,x] = np.sum(dyx**2 * glcm, axis=(0,1))
            dissimilarity[:,:,y,x] = np.sum(glcm * dyx, axis=(0,1))
            homogeneity[:,:,y,x] = np.sum(glcm/(1+dyx**2), axis=(0,1))
            energy[:,:,y,x] = np.sum(glcm**2, axis=(0,1))
    
    return contrast, dissimilarity, homogeneity, energy

