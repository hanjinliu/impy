import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    from skimage.feature import register_translation as phase_cross_correlation
from skimage import transform as sktrans
from ..func import record, same_dtype


__all__ = ["drift_correction"]

@record
def track_drift(self, axis="t", **kwargs):
    """
    Calculate (x,y) change based on cross correlation.
    """
    if self.ndim != 3:
        raise TypeError(f"input must be three dimensional, but got {self.shape}")

    # slow drift needs large upsampling numbers
    corr_kwargs = {"upsample_factor": 10}
    corr_kwargs.update(kwargs)
    
    # self.ongoing = "drift tracking"
    shift_list = [[0.0, 0.0]]
    last_img = None
    for _, img in self.iter(axis):
        if last_img is not None:
            shift, _, _ = phase_cross_correlation(last_img, img, **corr_kwargs)
            shift_total = shift + shift_list[-1]            # list + ndarray -> ndarray
            shift_list.append(shift_total)
            last_img = img
        else:
            last_img = img
    

    result = np.fliplr(shift_list) # shift is (y,x) order in skreg
    return result

def _show_drift(result):
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

@same_dtype(True)
@record
def drift_correction(self, shift=None, ref=None, order=1, show_drift=True):
    """
    shift: (N, 2) array, optional.
        x,y coordinates of drift. If None, this parameter will be determined by the
        track drift() function, using self or ref if indicated. 

    ref: ImgArray object, optional
        The reference 3D image to determine drift.
    
    order: int, optional
        The order of interpolation. See skimage.transform.warp.

    e.g.
    >>> drift = [[ dx1, dy1], [ dx2, dy2], ... ]
    >>> img = img0.drift_correction()
    """
    if shift is None:
        # determine 'ref'
        if ref is None:
            ref = self
        elif not isinstance(ref, self.__class__):
            raise TypeError(f"'ref' must be ImgArray object, but got {type(ref)}")
        elif ref.axes != "tyx":
            raise ValueError(f"Cannot track drift using {ref.axes} image")

        shift = track_drift(ref, axis="t")
        if show_drift:
            _show_drift(shift)
        self.ongoing = "drift_correction"

    elif shift.shape[1] != 2:
        raise TypeError(f"Invalid shift shape: {shift.shape}")
    elif shift.shape[0] != self.sizeof("t"):
        raise TypeError(f"Length inconsistency between image and shift")

    out = np.empty(self.shape)
    for sl, img in self.iter("ptzc"):
        if type(sl) is int:
            tr = -shift[sl]
        else:
            tr = -shift[sl[0]]
        mx = sktrans.AffineTransform(translation=tr)
        out[sl] = sktrans.warp(img.astype("float32"), mx, order=order)
    out = out.view(self.__class__)

    out._set_info(self, "Drift-Correction")
    out.temp = shift
    return out