import numpy as np
from dask import array as da
from ..._cupy import xp, xp_fft, asnumpy, xp_ndarray

# skimage/registration/_phase_cross_correlation.py 

def subpixel_pcc(f0:xp_ndarray, f1:xp_ndarray, upsample_factor:int):
    shape = f0.shape
    product = f0 * f1.conj()
    cross_correlation = xp_fft.ifftn(product)
    maxima = xp.unravel_index(xp.argmax(abs2(cross_correlation)),
                              cross_correlation.shape)
    midpoints = xp.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = xp.asarray(maxima, dtype=xp.float32)
    shifts[shifts > midpoints] -= xp.array(shape)[shifts > midpoints]
    # Initial shift estimate in upsampled grid
    shifts = xp.fix(shifts * upsample_factor) / upsample_factor
    if upsample_factor > 1:
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = xp.fix(upsampled_region_size / 2.0)
        upsample_factor = xp.array(upsample_factor, dtype=xp.float32)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset
                                           ).conj()
        # Locate maximum and map back to original pixel grid
        maxima = xp.unravel_index(xp.argmax(abs2(cross_correlation)),
                                    cross_correlation.shape)

        maxima = xp.asarray(maxima, dtype=xp.float32) - dftshift

        shifts = shifts + maxima / upsample_factor
    return asnumpy(shifts)

def subpixel_pcc_dask(f0:da.core.Array, f1:da.core.Array, upsample_factor:int):
    shape = f0.shape
    product = f0 * f1.conj()
    cross_correlation = da.fft.ifftn(product).astype(np.complex64)
    maxima = da.unravel_index(da.argmax(abs2(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = da.stack(maxima).astype(np.float32)
    shifts = da.where(shifts > midpoints, shifts - da.array(shape), shifts)
    
    # Initial shift estimate in upsampled grid
    shifts = da.fix(shifts * upsample_factor) / upsample_factor
    if upsample_factor > 1:
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=xp.float32)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft_dask(product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset
                                           ).conj()
        # Locate maximum and map back to original pixel grid
        maxima = da.unravel_index(da.argmax(abs2(cross_correlation)),
                                    cross_correlation.shape)

        maxima = da.asarray(maxima, dtype=np.float32) - dftshift

        shifts = shifts + maxima / upsample_factor
    return shifts


def _upsampled_dft_dask(data, upsampled_region_size, upsample_factor, axis_offsets):
    # if people pass in an integer, expand it to a list of equal-sized sections
    upsampled_region_size = [upsampled_region_size,] * data.ndim

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((da.arange(ups_size) - ax_offset)[:, xp.newaxis]
                  * da.fft.fftfreq(n_items, upsample_factor))
        kernel = da.exp(-2j * xp.pi * kernel)

        data = da.tensordot(kernel, data, axes=(1, -1))
    return data

def _upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    # if people pass in an integer, expand it to a list of equal-sized sections
    upsampled_region_size = [upsampled_region_size,] * data.ndim

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((xp.arange(ups_size) - ax_offset)[:, xp.newaxis]
                  * xp.fft.fftfreq(n_items, upsample_factor))
        kernel = xp.exp(-2j * xp.pi * kernel)

        data = xp.tensordot(kernel, data, axes=(1, -1))
    return data

def abs2(a):
    return a.real**2 + a.imag**2