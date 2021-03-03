import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from skimage import registration as skreg
from skimage import transform as sktrans
from ..func import record

"""
Maybe this is better...
https://scikit-image.org/docs/0.13.x/auto_examples/transform/plot_register_translation.html
"""

__all__ = ["drift_correction"]

@record
def track_drift(self, axis="t", **kwargs):
    """
    Calculate (x,y) change based on cross correlation.
    """
    if (self.ndim != 3):
        raise TypeError(f"input must be three dimensional, but got {self.shape}")

    # slow drift needs large upsampling numbers
    corr_kwargs = {"upsample_factor": 10}
    corr_kwargs.update(kwargs)
    
    # self.ongoing = "drift tracking"
    shift_list = [[0.0, 0.0]]
    last_img = None
    for i, (_, img) in enumerate(self.iter(axis)):
        if (last_img is not None):
            shift, _, _ = skreg.phase_cross_correlation(last_img, img, **corr_kwargs)
            shift_total = shift + shift_list[-1]            # list + ndarray -> ndarray
            shift_list.append(shift_total)
            last_img = img
        else:
            last_img = img
    

    result = np.fliplr(shift_list) # shift is (y,x) order in skreg
    show_drift(result)
    # del self.ongoing
    return result

def show_drift(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, title="drift")
    ax.plot(result[:, 0], result[:, 1], marker="+", color="red")
    ax.grid()
    # delete the default axes and let x=0 and y=0 be new ones.
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # let the interval of x-axis and that of y-axis be equal.
    ax.set_aspect("equal")
    # set the x/y-tick intervals to 1.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    return None

@record
def drift_correction(self, shift=None, ref=None):
    """
    shift: (N, 2) array, optional.
    x,y coordinates of drift. If None, this parameter will be determined by the
    track drift() function, using self or ref if indicated. 

    ref: ImgArray object, optional
    The reference 3D image to determine drift.

    e.g.
    >>> drift = [[ dx1, dy1], [ dx2, dy2], ... ]
    >>> img = img0.drift_correction()
    """
    if (shift is None):
        # determine 'ref'
        if (ref is None):
            ref = self
        elif (not isinstance(ref, self.__class__)):
            raise TypeError(f"'ref' must be ImgArray object, but got {type(ref)}")
        elif (ref.axes != "tyx"):
            raise ValueError(f"Cannot track drift using {ref.axes} image")

        shift = track_drift(ref, axis="t")

    elif (shift.shape[1] != 2):
        raise TypeError(f"Invalid shift shape: {shift.shape}")
    elif (shift.shape[0] != self.sizeof("t")):
        raise TypeError(f"Length inconsistency between image and shift")

    out = np.empty(self.shape)
    for t, img in self.as_uint16().iter("tzc"):
        if (type(t) is int):
            tr = -shift[t]
        else:
            tr = -shift[t[0]]
        mx = sktrans.AffineTransform(translation=tr)
        out[t] = sktrans.warp(img.astype("float64"), mx)
    out = out.view(self.__class__).as_uint16()

    out._set_info(self, "Drift-Correction")
    out.temp = shift
    return out