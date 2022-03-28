from __future__ import annotations
import numpy as np
from functools import lru_cache
from ...array_api import xp

# Phase Cross Correlation
# Modified from skimage/registration/_phase_cross_correlation.py
# Compatible between numpy/cupy and supports maximum shifts.

# TODO: if max_shifts is small, DFT should be used in place of FFT

def subpixel_pcc(
    f0: xp.ndarray, 
    f1: xp.ndarray, 
    upsample_factor: int,
    max_shifts: tuple[float, ...] | None = None,
):
    product = f0 * f1.conj()
    cross_correlation = xp.fft.ifftn(product)
    power = abs2(cross_correlation)
    if max_shifts is not None:
        max_shifts = xp.asarray(max_shifts)
        power = crop_by_max_shifts(power, max_shifts, max_shifts)
        
    maxima = xp.unravel_index(xp.argmax(power), power.shape)
    midpoints = xp.array([np.fix(axis_size / 2) for axis_size in power.shape])

    shifts = xp.asarray(maxima, dtype=np.float32)
    shifts[shifts > midpoints] -= xp.array(power.shape)[shifts > midpoints]
    # Initial shift estimate in upsampled grid
    shifts = xp.fix(shifts * upsample_factor) / upsample_factor
    if upsample_factor > 1:
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = xp.fix(upsampled_region_size / 2.0)
        upsample_factor = xp.array(upsample_factor, dtype=np.float32)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset
                                           ).conj()
        # Locate maximum and map back to original pixel grid
        power = abs2(cross_correlation)
        
        if max_shifts is not None:
            _upsampled_left_shifts = (shifts + max_shifts) * upsample_factor
            _upsampled_right_shifts = (max_shifts - shifts) * upsample_factor
            power = crop_by_max_shifts(power, _upsampled_left_shifts, _upsampled_right_shifts)
            
        maxima = xp.unravel_index(xp.argmax(power), power.shape)
        maxima = xp.asarray(maxima, dtype=np.float32) - dftshift
        shifts = shifts + maxima / upsample_factor
    return shifts

def _upsampled_dft(
    data: xp.ndarray, 
    upsampled_region_size: np.ndarray, 
    upsample_factor: int, 
    axis_offsets: xp.ndarray
) -> xp.ndarray:
    # if people pass in an integer, expand it to a list of equal-sized sections
    upsampled_region_size = [upsampled_region_size,] * data.ndim

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((xp.arange(ups_size) - ax_offset)[:, np.newaxis]
                  * xp.fft.fftfreq(n_items, upsample_factor))
        kernel = xp.exp(-2j * np.pi * kernel)

        data = xp.tensordot(kernel, data, axes=(1, -1))
    return data

def abs2(a: xp.ndarray) -> xp.ndarray:
    return a.real**2 + a.imag**2

def crop_by_max_shifts(power: xp.ndarray, left, right):
    shifted_power = xp.fft.fftshift(power)
    centers = tuple(s//2 for s in power.shape)
    slices = tuple(
        slice(max(c - int(shiftl), 0), min(c + int(shiftr) + 1, s), None) 
        for c, shiftl, shiftr, s in zip(centers, left, right, power.shape)
    )
    return xp.fft.ifftshift(shifted_power[slices])


def subpixel_ncc(
    img0: xp.ndarray, 
    img1: xp.ndarray, 
    upsample_factor: int,
    max_shifts: tuple[float, ...] | None = None,
):
    ndim = img1.ndim
    if max_shifts is not None:
        max_shifts = tuple(max_shifts)
    pad_width, sl = _get_padding_params(img0.shape, img1.shape, max_shifts)
    padimg = xp.pad(img0[sl], pad_width=pad_width, mode="constant", constant_values=0)
    
    corr = xp.signal.fftconvolve(
        padimg, img1[(slice(None, None, -1),)*ndim], mode="valid"
    )[(slice(1, -1, None),)*ndim]
    
    _win_sum = _window_sum_2d if ndim == 2 else _window_sum_3d
    win_sum1 = _win_sum(padimg, img1.shape)
    win_sum2 = _win_sum(padimg**2, img1.shape)
    
    template_mean = xp.mean(img1)
    template_volume = xp.prod(xp.array(img1.shape))
    template_ssd = xp.sum((img1 - template_mean)**2)
    
    var = (win_sum2 - win_sum1**2 / template_volume) * template_ssd
    
    # zero division happens when perfectly matched
    response = xp.zeros_like(corr)
    mask = var > 0
    response[mask] = (corr - win_sum1 * template_mean)[mask] / _safe_sqrt(var, fill=np.inf)[mask]
    
    if max_shifts is None:
        pad_width_eff = (3,) * ndim
    else:
        pad_width_eff = tuple((s - int(m) * 2 - 1)//2 for m, s in zip(max_shifts, response.shape))
    sl_res = tuple(slice(w, -w, None) for w in pad_width_eff)
    response_center = response[sl_res]
    maxima = xp.unravel_index(xp.argmax(response_center), response_center.shape)
    midpoints = xp.asarray(response_center.shape) // 2
    
    if upsample_factor > 1:
        coords = _create_mesh(upsample_factor, maxima, max_shifts, midpoints, pad_width_eff, xp.state)
        local_response = xp.ndi.map_coordinates(
            response, coords, order=3, mode="reflect", prefilter=True
        )
        local_maxima = xp.unravel_index(xp.argmax(local_response), local_response.shape)
        zncc = local_response[local_maxima]
        shifts = xp.array(maxima) - midpoints + xp.array(local_maxima) / upsample_factor - 1
    else:
        zncc = response[maxima]
        shifts = xp.array(maxima) - midpoints
        
    return xp.asarray(shifts, dtype=np.float32), zncc

# Identical to skimage.feature.template, but compatible between numpy and cupy.
def _window_sum_2d(image, window_shape):
    window_sum = xp.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])
    window_sum = xp.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def _window_sum_3d(image, window_shape):
    window_sum = _window_sum_2d(image, window_shape)
    window_sum = xp.cumsum(window_sum, axis=2)
    window_sum = (window_sum[:, :, window_shape[2]:-1]
                  - window_sum[:, :, :-window_shape[2] - 1])

    return window_sum

def _safe_sqrt(a: xp.ndarray, fill=0):
    out = xp.full(a.shape, fill, dtype=np.float32)
    out = xp.zeros_like(a)
    mask = a > 0
    out[mask] = xp.sqrt(a[mask])
    return out

@lru_cache(maxsize=12)
def _get_padding_params(
    shape0: tuple[int, ...], 
    shape1: tuple[int, ...],
    max_shifts: tuple[int, ...] | None,
) -> tuple[list[tuple[int, ...]], list[slice] | slice]:
    if max_shifts is None:
        pad_width = [(w, w) for w in shape1]
        sl = slice(None)
    else:
        pad_width: list[tuple[int, ...]] = []
        sl: list[slice] = []
        for w, s0, s1 in zip(max_shifts, shape0, shape1):
            w_int = int(np.ceil(w + 3 - (s0 - s1)/2))
            if w_int >= 0:
                pad_width.append((w_int,) * 2)
                sl.append(slice(None))
            else:
                pad_width.append((0,) * 2)
                sl.append(slice(-w_int, w_int, None))
        sl = tuple(sl)
    
    return pad_width, sl


def _create_mesh(
    upsample_factor: int,
    maxima: tuple[int, ...], 
    max_shifts: tuple[int, ...] | None,
    midpoints: tuple[int, ...],
    pad_width_eff: tuple[int, ...],
    state: str = "numpy",  # just for caching
):
    if max_shifts is not None:
        shifts = np.array(maxima, dtype=np.float32) - np.array(midpoints, dtype=np.float32)
        max_shifts = np.array(max_shifts, dtype=np.float32)
        left = -shifts - max_shifts
        right = -shifts + max_shifts
        local_shifts = tuple(
            [int(round(max(shiftl, -1) * upsample_factor)),
             int(round(min(shiftr, 1) * upsample_factor))]
            for shiftl, shiftr in zip(left, right)
        )
    else:
        local_shifts = ([-upsample_factor, upsample_factor],) * len(maxima)
    mesh = xp.meshgrid(
        *[xp.arange(s0, s1+1)/upsample_factor + m + w
          for (s0, s1), m, w in zip(local_shifts, maxima, pad_width_eff)], 
        indexing="ij",
    )
    return xp.stack(mesh, axis=0)
