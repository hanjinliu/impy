import numpy as np


def adjust_bin(img, binsize, check_edges, dims, all_axes):
    shape = []
    scale = []
    for i, a in enumerate(all_axes):
        s = img.shape[i]
        if a in dims:
            b = binsize
            if s % b != 0:
                if check_edges:
                    raise ValueError(f"Cannot bin axis {a} with length {s} by bin size {binsize}")
                else:
                    img = img[(slice(None),)*i + (slice(None, s//b*b),)]
        else:
            b = 1
        shape += [s//b, b]
        scale.append(1/b)
    
    shape = tuple(shape)
    return img, shape, scale


