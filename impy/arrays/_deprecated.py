from __future__ import annotations
import warnings
import numpy as np
from typing import TYPE_CHECKING

from ._utils._skimage import skfeat, skreg
from ._utils import _misc, _docs, _transform, _structures

from ..utils.axesop import complement_axes, find_first_appeared
from ..utils.deco import check_input_and_output, dims_to_spatial_axes
from ..utils.gauss import GaussianParticle
from ..utils.misc import check_nd

from ..collections import DataDict
from .._types import nDFloat, Dims
from ..array_api import xp
from ..axes import AxisLike

if TYPE_CHECKING:
    from ..frame import MarkerFrame
    from .imgarray import ImgArray

@dims_to_spatial_axes
def gauss_sm(
    self: ImgArray,
    coords = None,
    radius = 4,
    sigma = 1.5, 
    filt = None,
    percentile: float = 95,
    *,
    return_all: bool = False,
    dims = None
):
    """
    Calculate positions of particles in subpixel precision using Gaussian fitting.

    Parameters
    ----------
    coords : MarkerFrame or (N, 2) array, optional
        Coordinates of peaks. If None, this will be determined by find_sm.
    radius : int, default is 4.
        Fitting range. Rectangular image with size 2r+1 x 2r+1 will be send to Gaussian
        fitting function.
    sigma : float, default is 1.5
        Expected standard deviation of particles.
    filt : callable, optional
        For every slice `sl`, label is added only when filt(`input`) == True is satisfied.
        This discrimination is conducted before Gaussian fitting so that stringent filter
        will save time.
    percentile, dims :
        Passed to peak_local_max()
    return_all : bool, default is False
        If True, fitting results are all returned as Frame Dict.
    {dims}

    Returns
    -------
    MarkerFrame, if return_all == False
        Gaussian centers.
    DataDict with keys {means, sigmas, errors}, if return_all == True
        Dictionary that contains means, standard deviations and fitting errors.
    """        
    warnings.warn("'gauss_sm' is deprecated. Use 'centroid_sm' instead.", DeprecationWarning)
    from scipy.linalg import pinv as pseudo_inverse
    from ..frame import MarkerFrame
    
    if coords is None:
        coords = self.find_sm(sigma=sigma, dims=dims, percentile=percentile)
    else:
        coords = _check_coordinates(coords, self)    
    ndim = len(dims)
    filt = check_filter_func(filt)
    
    radius = np.asarray(check_nd(radius, ndim))
    
    shape = self.sizesof(dims)
    
    means = []  # fitting results of means
    sigmas = [] # fitting results of sigmas
    errs = []   # fitting errors of means
    ab = []
    for crd in coords.values:
        center = tuple(crd[-ndim:])
        label_sl = tuple(crd[:-ndim])
        sl = _specify_one(center, radius, shape) # sl = (..., z,y,x)
        input_img = self.value[label_sl][sl]
        if input_img.size == 0 or not filt(input_img):
            continue
        
        gaussian = GaussianParticle(initial_sg=sigma)
        res = gaussian.fit(input_img, method="BFGS")
        
        if (gaussian.mu_inrange(0, radius*2) and 
            gaussian.sg_inrange(sigma/3, sigma*3) and
            gaussian.a > 0):
            gaussian.shift(center - radius)
            # calculate fitting error with Jacobian
            if return_all:
                jac = res.jac[:2].reshape(1,-1)
                cov = pseudo_inverse(jac.T @ jac)
                err = np.sqrt(np.diag(cov))
                sigmas.append(label_sl + tuple(gaussian.sg))
                errs.append(label_sl + tuple(err))
                ab.append(label_sl + (gaussian.a, gaussian.b))
            
            means.append(label_sl + tuple(gaussian.mu))
            
    kw = dict(columns=coords.col_axes, dtype=np.float32)
    
    if return_all:
        out = DataDict(means = MarkerFrame(means, **kw).as_standard_type(),
                        sigmas = MarkerFrame(sigmas, **kw).as_standard_type(),
                        errors = MarkerFrame(errs, **kw).as_standard_type(),
                        intensities = MarkerFrame(ab, 
                                                    columns=str(coords.col_axes)[:-ndim]+"ab",
                                                    dtype=np.float32))
        
        out.means.set_scale(coords.scale)
        out.sigmas.set_scale(coords.scale)
        out.errors.set_scale(coords.scale)
            
    else:
        out = MarkerFrame(means, **kw)
        out.set_scale(coords.scale)
                        
    return out


@dims_to_spatial_axes
@check_input_and_output
def corner_peaks(
    self: ImgArray,
    *, 
    min_distance:int = 1,
    percentile: float | None = None, 
    topn: int = np.inf,
    topn_per_label: int = np.inf,
    exclude_border: bool = True,
    use_labels: bool = True, 
    dims: Dims = None
) -> MarkerFrame:
    """
    Find local corner maxima. Slightly different from peak_local_max.

    Parameters
    ----------
    min_distance : int, default is 1
        Minimum distance allowed for each two peaks. This parameter is slightly
        different from that in ``skimage.feature.peak_local_max`` because here float
        input is allowed and every time footprint is calculated.
    percentile : float, optional
        Percentile to compute absolute threshold.
    topn : int, optional
        Maximum number of peaks **for each iteration**.
    topn_per_label : int, default is np.inf
        Maximum number of peaks per label.
    use_labels : bool, default is True
        If use self.labels when it exists.
    {dims}
        
    Returns
    -------
    MarkerFrame
        DataFrame with columns same as axes of self. For example, if self.axes is "tcyx" then
        return value has "t", "c", "y" and "x" columns, and sub-frame at t=0, c=0 contains all
        the coordinates of corners in the slice at t=0, c=0.
    """        
    warnings.warn("'corner_peaks' is deprecated and will be removed.", DeprecationWarning)
    # separate spatial dimensions and others
    ndim = len(dims)
    dims_list = list(dims)
    c_axes = complement_axes(dims, self.axes)
    c_axes_list = list(c_axes)
    
    if isinstance(exclude_border, bool):
        exclude_border = int(min_distance) if exclude_border else False
    
    thr = None if percentile is None else np.percentile(self.value, percentile)
    
    import pandas as pd
    from ..frame import MarkerFrame

    df_all: list[pd.DataFrame] = []
    for sl, img in self.iter(c_axes, israw=True, exclude=dims):
        # skfeat.corner_peaks overwrite something so we need to give copy of img.
        if use_labels and img.labels is not None:
            labels = xp.asnumpy(img.labels)
        else:
            labels = None
        
        indices = skfeat.corner_peaks(
            xp.asnumpy(img),
            footprint=xp.asnumpy(_structures.ball_like(min_distance, ndim)),
            threshold_abs=thr,
            num_peaks=topn,
            num_peaks_per_label=topn_per_label,
            labels=labels,
            exclude_border=exclude_border
        )
        indices = pd.DataFrame(indices, columns=dims_list)
        indices[c_axes_list] = sl
        df_all.append(indices)
        
    out = pd.concat(df_all, axis=0)
    out = MarkerFrame(out, columns=self.axes, dtype="uint16")
    out.set_scale(self)
    return out

@_docs.write_docs
@dims_to_spatial_axes
@check_input_and_output
def find_corners(
    self,
    sigma: nDFloat = 1,
    k: float = 0.05,
    *, 
    dims: Dims = None
) -> ImgArray:
    """
    Corner detection using Harris response.

    Parameters
    ----------
    {sigma}
    k : float, optional
        Sensitivity factor to separate corners from edges, typically in range [0, 0.2].
        Small values of k result in detection of sharp corners.
    {dims}

    Returns
    -------
    MarkerFrame
        Coordinates of corners. For details see ``corner_peaks`` method.
    """        
    warnings.warn("'find_corners' is deprecated and will be removed.", DeprecationWarning)
    res = self.gaussian_filter(sigma=1).corner_harris(sigma=sigma, k=k, dims=dims)
    out = res.corner_peaks(min_distance=3, percentile=97, dims=dims)
    return out

@check_input_and_output
def track_template(self: ImgArray, template:np.ndarray, bg=None, along: AxisLike = "t") -> MarkerFrame:
    """
    Tracking using template matching. For every time frame, matched region is interpreted as a
    new template and is used for the next template. To avoid slight shifts accumulating to the
    template image, new template image will be fitteg to the old one by phase cross correlation.

    Parameters
    ----------
    template : np.ndarray
        Template image. Must be 2 or 3 dimensional. 
    bg : float, optional
        Background intensity. If not given, it will calculated as the minimum value of 
        the original image.
    along : str, default is "t"
        Which axis will be the time axis.

    Returns
    -------
    MarkerFrame
        Centers of matched templates.
    """        
    warnings.warn("'track_template' is deprecated and will be removed.", DeprecationWarning)
    template = _check_template(template)
    template_new = template
    t_shape = np.array(template.shape)
    bg = _check_bg(self, bg)
    dims = "yx" if template.ndim == 2 else "zyx"
    # check along
    if along is None:
        along = find_first_appeared("tpzc", include=self.axes, exclude=dims)
    elif len(along) != 1:
        raise ValueError("`along` must be single character.")
    
    if complement_axes(dims, self.axes) != along:
        raise ValueError(f"Image axes do not match along ({along}) and template dimensions ({dims})")
    
    ndim = len(dims)
    rem_edge_sl = tuple(slice(s//2, -s//2) for s in t_shape)
    pos = []
    shift = np.zeros(ndim, dtype=np.float32)
    for sl, img in self.as_float().iter(along):
        template_old = _translate_image(template_new, shift, cval=bg)
        _, resp = _misc.ncc(img, template_old, bg)
        resp_crop = resp[rem_edge_sl]
        peak = np.unravel_index(np.argmax(resp_crop), resp_crop.shape) + t_shape//2
        pos.append(peak)
        sl = []
        for i in range(ndim):
            d0 = peak[i] - t_shape[i]//2
            d1 = d0 + t_shape[i]
            sl.append(slice(d0, d1, None))
        template_new = img[tuple(sl)]
        shift = skreg.phase_cross_correlation(template_old, template_new, 
                                                return_error=False, upsample_factor=10)
    
    from ..frame import MarkerFrame
    pos = np.array(pos)
    pos = np.hstack([np.arange(self.sizeof(along), dtype=np.uint16).reshape(-1,1), pos])
    pos = MarkerFrame(pos, columns=along+dims)
    
    return pos


def _translate_image(img, shift, order=1, cval=0):
    ndim = len(shift)
    mx = _transform.compose_affine_matrix(translation=-np.asarray(shift), ndim=ndim)
    return _transform.warp(img, mx, order=order, cval=cval)


def _specify_one(center, radius, shape:tuple) -> tuple[slice]:
    sl = tuple(slice(max(0, xc-r), min(xc+r+1, sh), None) 
                        for xc, r, sh in zip(center, radius, shape))
    return sl

def check_filter_func(f):
    if f is None:
        f = lambda x: True
    elif not callable(f):
        raise TypeError("`filt` must be callable.")
    return f

def _check_bg(img: ImgArray, bg):
    # determine bg
    if bg is None:
        bg = img.min()
    elif isinstance(bg, str) and bg.endswith("%"):
        bg = np.percentile(img.value, float(bg[:-1]))
    elif not np.isscalar(bg):
        raise TypeError("Wrong type of `bg`.")
    return bg

def _check_template(template):
    if not isinstance(template, np.ndarray):
        raise TypeError(f"`template` must be np.ndarray, but got {type(template)}")
    elif template.ndim not in (2, 3):
        raise ValueError("`template must be 2 or 3 dimensional.`")
    template = np.asarray(template).astype(np.float32, copy=True)
    return template

def _check_coordinates(coords, img: ImgArray, dims: Dims = None):
    from ..frame import MarkerFrame
    if not isinstance(coords, MarkerFrame):
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        elif coords.ndim != 2:
            raise ValueError("Input cannot be interpreted as coordinate(s).")
        if dims is None:
            ndim = coords.shape[1]
            if ndim == img.ndim:
                dims = img.axes
            else:
                dims = complement_axes("c", img.axes)[-ndim:]
        coords = MarkerFrame(coords, columns=dims, dtype=np.uint16)
        coords.set_scale(img)
    
    return coords


Funcs = {
    "gauss_sm": gauss_sm,
    "corner_peaks": corner_peaks,
    "find_corners": find_corners,
    "track_template": track_template,
}
