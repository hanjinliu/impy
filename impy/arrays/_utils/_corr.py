from __future__ import annotations
from re import S
import numpy as np
from ..._cupy import xp, xp_fft, xp_ndarray

# Modified from skimage/registration/_phase_cross_correlation.py
# Compatible between numpy/cupy.

def subpixel_pcc(
    f0: xp_ndarray, 
    f1: xp_ndarray, 
    upsample_factor: int,
    max_shifts: tuple[int, ...] | None = None,
) -> xp_ndarray:
    product = f0 * f1.conj()
    cross_correlation = xp_fft.ifftn(product)
    power = abs2(cross_correlation)
    if max_shifts is not None:
        shifted_power = xp_fft.fftshift(power)
        centers = tuple(s//2 for s in power.shape)
        slices = tuple(
            slice(c-shift, c+shift + 1, None) for c, shift in zip(centers, max_shifts)
        )
        power = xp_fft.ifftshift(shifted_power[slices])
        
    maxima = xp.unravel_index(xp.argmax(power), power.shape)
    midpoints = xp.array([np.fix(axis_size / 2) for axis_size in power.shape])

    shifts = xp.asarray(maxima, dtype=xp.float32)
    shifts[shifts > midpoints] -= xp.array(power.shape)[shifts > midpoints]
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
    return shifts

def _upsampled_dft(
    data: xp_ndarray, 
    upsampled_region_size: np.ndarray, 
    upsample_factor: int, 
    axis_offsets: xp_ndarray
) -> xp_ndarray:
    # if people pass in an integer, expand it to a list of equal-sized sections
    upsampled_region_size = [upsampled_region_size,] * data.ndim

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((xp.arange(ups_size) - ax_offset)[:, xp.newaxis]
                  * xp.fft.fftfreq(n_items, upsample_factor))
        kernel = xp.exp(-2j * xp.pi * kernel)

        data = xp.tensordot(kernel, data, axes=(1, -1))
    return data

def abs2(a: xp_ndarray) -> xp_ndarray:
    return a.real**2 + a.imag**2