import numpy as np
from numba import jit

__all__ = ["glcm_props_"]

@jit(signature_or_function="void(uint8[:], f4[:], i8, i8, uint8[:,:], f4[:,:,:,:])", nopython=True, cache=True)
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
