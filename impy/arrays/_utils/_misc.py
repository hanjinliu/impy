from __future__ import annotations
import numpy as np
from impy.array_api import xp

def adjust_bin(
    img: np.ndarray,
    binsize: int,
    check_edges: bool,
    dims: str,
    all_axes: str
) -> tuple[np.ndarray, tuple[int, ...], float]:
    """Adjust bin size of an image prior to calling `binning`."""
    shape = []
    scale = []
    sl = []
    dims = list(dims)
    for i, a in enumerate(all_axes):
        s = img.shape[i]
        if a in dims:
            b = binsize
            if s % b != 0:
                if check_edges:
                    raise ValueError(f"Cannot bin axis {a} with length {s} by bin size {binsize}")
                else:
                    sl.append(slice(None, s//b*b))
            else:
                sl.append(slice(None))
        else:
            b = 1
            sl.append(slice(None))
        shape += [s//b, b]
        scale.append(1/b)
    sl = tuple(sl)
    shape = tuple(shape)
    return img[sl], shape, scale


def make_rotated_axis(src, dst):
    dr = dst - src
    d = np.sqrt(sum(dr**2))
    n = int(round(d))
    return np.linspace(src, src + dr / d * (n - 1), n)

def make_rotated_axis_by_vec(src, unit_vec, n: int):
    return np.linspace(src, src + unit_vec * (n - 1), n)

def make_pad(pad_width, dims, all_axes, **kwargs):
    """More flexible padding than `np.pad`."""
    pad_width_ = []

    # for consistency with scipy-format
    if "cval" in kwargs.keys():
        kwargs["constant_values"] = kwargs["cval"]
        kwargs.pop("cval")

    if hasattr(pad_width, "__iter__") and len(pad_width) == len(dims):
        pad_iter = iter(pad_width)
        for a in all_axes:
            if a in dims:
                pad_width_.append(next(pad_iter))
            else:
                pad_width_.append((0, 0))

    elif isinstance(pad_width, int):
        for a in all_axes:
            if a in dims:
                pad_width_.append((pad_width, pad_width))
            else:
                pad_width_.append((0, 0))
    else:
        raise TypeError(f"pad_width must be iterable or int, but got {type(pad_width)}")

    return pad_width_

def dft(img: np.ndarray, exps: list[np.ndarray] = None):
    """Discrete Fourier Transform."""
    img = xp.asarray(img)
    for ker in reversed(exps):
        # K_{kx} * I_{zyx}
        img = xp.tensordot(xp.asarray(ker), img, axes=(1, -1))
    return img

def inpaint_mean(img: np.ndarray, mask: np.ndarray):
    """Inpaint missing values with mean of surrounding pixels."""
    img = xp.asarray(img)
    mask = xp.asarray(mask)
    labels, nfeatures = xp.ndi.label(mask)
    ndim = img.ndim
    out = img.copy()
    for i in range(nfeatures):
        mask_i = labels == i + 1
        expanded_i = xp.ndi.binary_dilation(mask_i , xp.ones((3,) * ndim))
        border = expanded_i ^ mask_i
        out[mask_i] = xp.mean(img[border])
    return out
