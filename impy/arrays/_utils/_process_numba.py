import numpy as np
from numba import jit

@jit("void(uint8[:], f4[:], i8, i8, uint8[:,:], f4[:,:,:,:])", nopython=True, cache=True)
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

@jit(nopython=True)
def _specify_square_2d(arr, coords, radius, label_offset=1):
    npoints = coords.shape[0]
    ny, nx = arr.shape
    ry, rx = radius
    for i in range(npoints):
        coord = coords[i,:]
        yc = coord[0]
        xc = coord[1]
        y0 = max( 0, yc-ry)
        y1 = min(ny, yc+ry+1)        
        x0 = max( 0, xc-rx)
        x1 = min(nx, xc+rx+1)
        arr[y0:y1, x0:x1] = i + label_offset

@jit(nopython=True)
def _specify_square_3d(arr, coords, radius, label_offset=1):
    npoints = coords.shape[0]
    nz, ny, nx = arr.shape
    rz, ry, rx = radius
    for i in range(npoints):
        coord = coords[i,:]
        zc = coord[0]
        yc = coord[1]
        xc = coord[2]
        z0 = max( 0, zc-rz)
        z1 = min(nz, zc+rz+1)        
        y0 = max( 0, yc-ry)
        y1 = min(ny, yc+ry+1)        
        x0 = max( 0, xc-rx)
        x1 = min(nx, xc+rx+1)
        arr[z0:z1, y0:y1, x0:x1] = i + label_offset

@jit(nopython=True)
def _specify_circ_2d(arr, coords, radius, label_offset=1):
    npoints = coords.shape[0]
    ny, nx = arr.shape
    ry, rx = radius
    for i in range(npoints):
        coord = coords[i,:]
        yc = coord[0]
        xc = coord[1]
        y0 = max( 0, yc-ry)
        y1 = min(ny, yc+ry+1)        
        x0 = max( 0, xc-rx)
        x1 = min(nx, xc+rx+1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if ((x-xc)/rx)**2 + ((y-yc)/ry)**2 <= 1:
                    arr[y,x] = i + label_offset

@jit(nopython=True)
def _specify_circ_3d(arr, coords, radius, label_offset=1):
    npoints = coords.shape[0]
    nz, ny, nx = arr.shape
    rz, ry, rx = radius
    for i in range(npoints):
        coord = coords[i,:]
        zc = coord[0]
        yc = coord[1]
        xc = coord[2]
        z0 = max( 0, zc-rz)
        z1 = min(nz, zc+rz+1)        
        y0 = max( 0, yc-ry)
        y1 = min(ny, yc+ry+1)        
        x0 = max( 0, xc-rx)
        x1 = min(nx, xc+rx+1)
        for z in range(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if ((x-xc)/rx)**2 + ((y-yc)/ry)**2 + ((z-zc)/rz)**2 <= 1:
                        arr[z,y,x] = i + label_offset
