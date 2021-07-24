import numpy as np
from ..._cupy import xp, xp_fft, asnumpy

def subpixel_pcc(f0, f1, upsample_factor):
    shape = f0.shape
    product = f0 * f1.conj()
    cross_correlation = xp_fft.ifftn(product)
    maxima = xp.unravel_index(xp.argmax(xp.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = xp.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = xp.stack(maxima).astype(xp.float32)
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
        maxima = xp.unravel_index(xp.argmax(xp.abs(cross_correlation)),
                                    cross_correlation.shape)

        maxima = xp.stack(maxima).astype(xp.float32) - dftshift

        shifts = shifts + maxima / upsample_factor
    return asnumpy(shifts)


def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((xp.arange(ups_size) - ax_offset)[:, xp.newaxis]
                  * xp.fft.fftfreq(n_items, upsample_factor))
        kernel = xp.exp(-2j * xp.pi * kernel)

        data = xp.tensordot(kernel, data, axes=(1, -1))
    return data