from __future__ import annotations
# skimage.morphology takes very long time to import. Here it is not imported explicitly, and is always
# accessed by `skimage.morphology.some_function`.
import skimage
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import exposure as skexp
from skimage import measure as skmes
from skimage import segmentation as skseg
from skimage import restoration as skres
from skimage import feature as skfeat
from skimage import registration as skreg
from skimage import graph as skgraph
from skimage import util as skutil

from functools import reduce, lru_cache
from ..._cupy import xp

# same as the function in skimage.filters._fft_based (available in scikit-image >= 0.19)
@lru_cache
def _get_ND_butterworth_filter(shape: tuple[int, ...], cutoff: float, order: int, 
                               high_pass: bool, real: bool):
    ranges = []
    for d, fc in zip(shape, cutoff):
        axis = xp.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d*fc)
        ranges.append(xp.fft.ifftshift(axis ** 2))
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    q2 = reduce(xp.add, xp.meshgrid(*ranges, indexing="ij", sparse=True))
    wfilt = 1 / (1 + q2**order)
    if high_pass:
        wfilt = 1 - wfilt
    return wfilt