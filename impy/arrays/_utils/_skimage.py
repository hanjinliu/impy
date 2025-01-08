from __future__ import annotations

import numpy as np

from functools import reduce, lru_cache

# same as the function in skimage.filters._fft_based (available in scikit-image >= 0.19)
@lru_cache(maxsize=4)
def _get_ND_butterworth_filter(
    shape: tuple[int, ...],
    cutoff: float,
    order: int,
    high_pass: bool,
    real: bool,
):
    if cutoff == 0:
        if high_pass:
            wfilt = np.ones(shape, dtype=np.float32)
        else:
            wfilt = np.zeros(shape, dtype=np.float32)
        return wfilt
    ranges = []
    for d, fc in zip(shape, cutoff):
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1, dtype=np.float32) / (d*fc)
        ranges.append(np.fft.ifftshift(axis ** 2))
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    q2 = reduce(np.add, np.meshgrid(*ranges, indexing="ij", sparse=True))
    wfilt = 1 / (1 + q2**order)
    if high_pass:
        wfilt = 1 - wfilt
    return wfilt
