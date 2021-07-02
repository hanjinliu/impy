from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import exposure as skexp
from skimage import measure as skmes
from skimage import segmentation as skseg
from skimage import restoration as skres
from skimage import feature as skfeat
from skimage import registration as skreg
from skimage import morphology as skmorph
from skimage import graph as skgraph
from scipy import ndimage as ndi

# same as the function in skimage.filters._fft_based (available in scikit-image >= 0.19)
def _get_ND_butterworth_filter(shape, cutoff, order, high_pass, real):
    import functools
    import numpy as np
    ranges = []
    for d, fc in zip(shape, cutoff):
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d*fc)
        ranges.append(np.fft.ifftshift(axis ** 2))
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    q2 = functools.reduce(np.add, np.meshgrid(*ranges, indexing="ij", sparse=True))
    wfilt = 1 / (1 + q2**order)
    if high_pass:
        wfilt = 1 - wfilt
    return wfilt