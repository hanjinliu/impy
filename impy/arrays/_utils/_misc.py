import numpy as np
from ..._cupy import xp

def adjust_bin(img, binsize, check_edges, dims, all_axes):
    shape = []
    scale = []
    sl = []
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
    return np.linspace(src, src+dr/d*(n-1), n)

def make_pad(pad_width, dims, all_axes, **kwargs):
    """
    More flexible padding than `np.pad`.
    """
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

def dft(img: xp.ndarray, exps: list[xp.ndarray] = None):
    img = xp.asarray(img)
    for ker in reversed(exps):
        # K_{kx} * I_{zyx}
        img = xp.tensordot(ker, img, axes=(1, -1))
    return img