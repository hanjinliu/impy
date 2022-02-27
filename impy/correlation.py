from __future__ import annotations
import numpy as np
from warnings import warn
from .core import asarray as ip_asarray
from .arrays import ImgArray, PropArray
from .arrays._utils import _docs
from .arrays._utils._transform import polar2d
from .arrays._utils._corr import subpixel_pcc
from .utils.axesop import complement_axes, add_axes
from .utils.utilcls import Progress
from .utils.deco import dims_to_spatial_axes
from .array_api import xp
from ._types import Dims

__all__ = ["fsc", "fourier_shell_correlation", "ncc", "zncc", "fourier_ncc", "fourier_zncc",
           "nmi", "pcc_maximum", "ft_pcc_maximum", "polar_pcc_maximum",
           "pearson_coloc", "manders_coloc"]

@_docs.write_docs
def fsc(
    img0: ImgArray, 
    img1: ImgArray, 
    dfreq: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Calculate Fourier Shell Correlation (FSC; or Fourier Ring Correlation, FRC, for 2-D images) 
    between two images. FSC is defined as:
    
    .. math::
    
        FSC(r) = \frac{Re(\sum_{r<r'<r+dr}[F_0(r') \cdot \bar{F_1}(r)])}
        {\sqrt{\sum_{r<r'<r+dr}|F_0(r')|^2 \cdot \sum_{r<r'<r+dr}|F_1(r')|^2}}
    
    In this function, frequency domain will be binned like this:
    
    .. code-block::

        |---|---|---|---|---|
        0  0.1 0.2 0.3 0.4 0.5
    
    and frequencies calculated in each bin will be 0.05, 0.15, ..., 0.45.
    
    Parameters
    ----------
    {inputs_of_correlation}
    dfreq : float, default is 0.02
        Difference of frequencies. This value will be the width of bins.
                
    Returns
    -------
    Two np.ndarray
        The first array is frequency, and the second array is FSC.
    """    
    img0, img1 = _check_inputs(img0, img1)
    
    shape = img0.shape
    dims = img0.axes
    
    freqs = xp.meshgrid(*[xp.fft.fftshift(xp.fft.fftfreq(s)) for s in shape])
    r = xp.sqrt(sum(f**2 for f in freqs))
    
    with Progress("fsc"):
        # make radially separated labels
        labels = (r/dfreq).astype(np.uint16)
        nlabels = int(xp.asnumpy(labels.max()))
        
        out = xp.empty(nlabels, dtype=np.float32)
        def radial_sum(arr):
            arr = xp.asarray(arr)
            return xp.ndi.sum_labels(arr, labels=labels, index=xp.arange(1, nlabels+1))

        f0 = img0.fft(dims=dims)
        f1 = img1.fft(dims=dims)
        
        cov = f0.real*f1.real + f0.imag*f1.imag
        pw0 = f0.real**2 + f0.imag**2
        pw1 = f1.real**2 + f1.imag**2
    
        out = radial_sum(cov)/xp.sqrt(radial_sum(pw0) * radial_sum(pw1))
    freq = (np.arange(len(out)) + 0.5) * dfreq
    return freq, out

# alias
fourier_shell_correlation = fsc


def _ncc(img0: ImgArray, img1: ImgArray, dims: Dims):
    # Basic Normalized Cross Correlation with batch processing
    n = np.prod(img0.sizesof(dims))
    if isinstance(dims, str):
        dims = tuple(img0.axisof(a) for a in dims)
    img0 = xp.asarray(img0)
    img1 = xp.asarray(img1)
    corr = xp.sum(img0 * img1, axis=dims) / (
        xp.std(img0, axis=dims)*xp.std(img1, axis=dims)) / n
    return xp.asnumpy(corr)


def _masked_ncc(img0: ImgArray, img1: ImgArray, dims: Dims, mask: ImgArray):
    if mask.ndim < img0.ndim:
        mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
    n = np.prod(img0.sizesof(dims))
    img0ma = np.ma.array(img0.value, mask=mask)
    img1ma = np.ma.array(img1.value, mask=mask)
    axis = tuple(img0.axisof(a) for a in dims)
    return np.ma.sum(img0ma * img1ma, axis=axis) / (
        np.ma.std(img0ma, axis=axis)*np.ma.std(img1ma, axis=axis)) / n


def _zncc(img0: ImgArray, img1: ImgArray, dims: Dims):
    # Basic Zero-Normalized Cross Correlation with batch processing.
    # Inputs must be already zero-normalized.
    if isinstance(dims, str):
        dims = tuple(img0.axisof(a) for a in dims)
    img0 = xp.asarray(img0)
    img1 = xp.asarray(img1)
    corr = xp.sum(img0 * img1, axis=dims) / (
        xp.sqrt(xp.sum(img0**2, axis=dims)*xp.sum(img1**2, axis=dims)))
    return xp.asnumpy(corr)


def _masked_zncc(img0: ImgArray, img1: ImgArray, dims: Dims, mask: ImgArray):
    if mask.ndim < img0.ndim:
        mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
    img0ma = np.ma.array(img0.value, mask=mask)
    img1ma = np.ma.array(img1.value, mask=mask)
    axis = tuple(img0.axisof(a) for a in dims)
    return np.sum(img0ma * img1ma, axis=axis) / (
        np.sqrt(np.sum(img0ma**2, axis=axis)*np.sum(img1ma**2, axis=axis)))

@_docs.write_docs
@dims_to_spatial_axes
def ncc(
    img0: ImgArray, 
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    squeeze: bool = True, 
    *, 
    dims: Dims = None
) -> PropArray | float:
    """
    Normalized Cross Correlation.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate 
        correlation.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    with Progress("ncc"):
        img0, img1 = _check_inputs(img0, img1)
        if mask is None:
            corr = _ncc(img0, img1, dims)
        else:
            corr = _masked_ncc(img0, img1, dims, mask)
    return _make_corr_output(corr, img0, "ncc", squeeze, dims)

@_docs.write_docs
@dims_to_spatial_axes
def zncc(
    img0: ImgArray, 
    img1: ImgArray, 
    mask: ImgArray | None = None,
    squeeze: bool = True,
    *,
    dims: Dims = None
) -> PropArray | float:
    """
    Zero-Normalized Cross Correlation.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate 
        correlation.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    with Progress("zncc"):
        img0, img1 = _check_inputs(img0, img1)
        img0zn = img0 - np.mean(img0, axis=dims, keepdims=True)
        img1zn = img1 - np.mean(img1, axis=dims, keepdims=True)
        if mask is None:
            corr = _zncc(img0zn, img1zn, dims)
        else:
            corr = _masked_zncc(img0zn, img1zn, dims, mask)
    return _make_corr_output(corr, img0, "zncc", squeeze, dims)

# alias
pearson_coloc = zncc

@_docs.write_docs
@dims_to_spatial_axes
def nmi(
    img0: ImgArray, 
    img1: ImgArray,
    mask: ImgArray | None = None,
    bins: int = 100, 
    squeeze: bool = True,
    *, 
    dims: Dims = None
) -> PropArray | float:
    r"""
    Normalized Mutual Information.
    
    :math:`Y(A, B) = \frac{H(A) + H(B)}{H(A, B)}`
                   
    See "Elegant SciPy"
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate 
        correlation.
    bins : int, default is 100
        Number of bins to construct histograms.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """
    from scipy.stats import entropy

    img0, img1 = _check_inputs(img0, img1)
    c_axes = complement_axes(dims, img0.axes)
    out = np.empty(img0.sizesof(c_axes), dtype=np.float32)
    if mask.ndim < img0.ndim:
        mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
    for sl, img0_, img1_ in iter2(img0, img1, c_axes):
        mask_ = mask[sl]
        hist, edges = np.histogramdd([np.ravel(img0_[mask_]),
                                      np.ravel(img1_[mask_])], bins=bins)
        hist /= np.sum(hist)
        e1 = entropy(np.sum(hist, axis=0)) # Shannon entropy
        e2 = entropy(np.sum(hist, axis=1))
        e12 = entropy(np.ravel(hist)) # mutual entropy
        out[sl] = (e1 + e2)/e12
    return _make_corr_output(out, img0, "nmi", squeeze, dims)

@_docs.write_docs
@dims_to_spatial_axes
def fourier_ncc(
    img0: ImgArray, 
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    squeeze: bool = True, 
    *, 
    dims: Dims = None
) -> PropArray | float:
    """
    Normalized Cross Correlation in Fourier space.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate
        correlation.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    with Progress("fourier_ncc"):
        img0, img1 = _check_inputs(img0, img1)
        f0 = np.sqrt(img0.power_spectra(dims=dims, zero_norm=True))
        f1 = np.sqrt(img1.power_spectra(dims=dims, zero_norm=True))
        if mask is None:
            corr = _ncc(f0, f1, dims)
        else:
            corr = _masked_ncc(f0, f1, dims, mask)
    return _make_corr_output(corr, img0, "fourier_ncc", squeeze, dims)

@_docs.write_docs
@dims_to_spatial_axes
def fourier_zncc(
    img0: ImgArray, 
    img1: ImgArray,
    mask: ImgArray | None = None, 
    squeeze: bool = True, 
    *, 
    dims: Dims = None
) -> PropArray | float:
    """
    Zero-Normalized Cross Correlation in Fourier space.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate 
        correlation.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    with Progress("fourier_zncc"):
        img0, img1 = _check_inputs(img0, img1)
        f0 = np.sqrt(img0.power_spectra(dims=dims, zero_norm=True))
        f1 = np.sqrt(img1.power_spectra(dims=dims, zero_norm=True))
        f0 -= np.mean(f0, axis=dims, keepdims=True)
        f1 -= np.mean(f1, axis=dims, keepdims=True)
        if mask is None:
            corr = _zncc(f0, f1, dims)
        else:
            corr = _masked_zncc(f0, f1, dims, mask)
    return _make_corr_output(corr, img0, "fourier_zncc", squeeze, dims)

@_docs.write_docs
def pcc_maximum(
    img0: ImgArray, 
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    upsample_factor: int = 10,
    max_shifts: int | tuple[int, ...] | None = None
) -> np.ndarray:
    """
    Calculate lateral shift between two images. 
    
    Same as ``skimage.registration.phase_cross_correlation`` but some additional parameters 
    are supported.

    Parameters
    ----------
    {inputs_of_correlation}
    upsample_factor : int, default is 10
        Up-sampling factor when calculating phase cross correlation.
    max_shifts : int, tuple of int, optional
        Maximum shifts in each dimension. If a single integer is given, it is interpreted 
        as maximum shifts in all dimensions. No upper bound of shifts if not given.
    
    Returns
    -------
    np.ndarray
        Shift in pixel.
    """    
    if img0 is img1:
        return np.zeros(img0.ndim)
    with Progress("pcc_maximum"):
        img0, img1 = _check_inputs(img0, img1)
        ft0 = img0.fft(dims=img0.axes)
        ft1 = img1.fft(dims=img1.axes)
        if mask is not None:
            ft0[mask] = 0
        if isinstance(max_shifts, (int, float)):
            max_shifts = (max_shifts,) * img0.ndim
        shift = subpixel_pcc(
            xp.asarray(ft0.value), 
            xp.asarray(ft1.value),
            upsample_factor, 
            max_shifts=max_shifts
        )
    return xp.asnumpy(shift)

@_docs.write_docs
def ft_pcc_maximum(
    img0: ImgArray,
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    upsample_factor: int = 10,
    max_shifts: float | tuple[float, ...] | None = None
) -> np.ndarray:
    """
    Calculate lateral shift between two images.
    
    This function takes Fourier transformed images as input. If you have to repetitively
    use a same template image, this function is faster.

    Parameters
    ----------
    {inputs_of_correlation}
    upsample_factor : int, default is 10
        Up-sampling factor when calculating phase cross correlation.
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as maximum shifts
        in all dimensions. No upper bound of shifts if not given.

    Returns
    -------
    np.ndarray
        Shift in pixel.
    """    
    with Progress("ft_pcc_maximum"):
        _check_dimensions(img0, img1)
        if mask is not None:
            img0 = img0.copy()
            img0[mask] = 0
        if isinstance(max_shifts, (int, float)):
            max_shifts = (max_shifts,) * img0.ndim
        shift = subpixel_pcc(
            xp.asarray(img0.value), 
            xp.asarray(img1.value),
            upsample_factor,
            max_shifts=max_shifts,
        )
    return xp.asnumpy(shift)

@_docs.write_docs
def polar_pcc_maximum(
    img0: ImgArray,
    img1: ImgArray,
    upsample_factor: int = 10,
    max_degree: int = None,
) -> float:
    """
    Calculate rotational shift between two images using polar Fourier transformation.

    Parameters
    ----------
    {inputs_of_correlation}
    upsample_factor : int, default is 10
        Up-sampling factor when calculating phase cross correlation.
    max_degree : int, tuple of int, optional
        Maximum rotation in degree.

    Returns
    -------
    float
        Rotation in degree
    """    
    img0, img1 = _check_inputs(img0, img1)
    if img0.ndim != 2:
        raise TypeError("Currently only 2D image is supported.")
    if max_degree is None:
        max_degree = 180
    rmax = min(img0.shape)
    with Progress("polar_pcc_maximum_2d"):
        imgp = ip_asarray(polar2d(img0, rmax, np.pi/180))
        imgrotp = ip_asarray(polar2d(img1, rmax, np.pi/180))
        max_shifts = (max_degree, 1)
        shift = pcc_maximum(imgp, imgrotp, upsample_factor=upsample_factor,
                            max_shifts=max_shifts)
    # Here, `shift` satisfies `img0.rotate(-shift[0]) == img1`
    return shift[0]


@_docs.write_docs
@dims_to_spatial_axes
def manders_coloc(
    img0: ImgArray, 
    img1: np.ndarray,
    *,
    squeeze: bool = True,
    dims: Dims = None
) -> PropArray | float:
    r"""
    Manders' correlation coefficient. This is defined as following:
    
    :math:`r = \frac{\sum_{i \in I_{ref}} I_i}{\sum_{i} I_i}`
    
    This value is NOT independent of background intensity. You need to correctly subtract
    background from self. This value is NOT interchangable between channels.
    
    Parameters
    ----------
    {inputs_of_correlation}
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation coefficient(s).
    """        
    if img1.dtype != bool:
        raise TypeError("`ref` must be a binary image.")
    if img0.shape != img1.shape:
        raise ValueError(f"Shape mismatch. `img0` has shape {img0.shape} but `img1` "
                         f"has shape {img1.shape}")
    if img0.axes != img1.axes:
        warn(f"Axes mismatch. `img0` has axes {img0.axes} but `img1` has axes {img1.axes}. "
              "Result may be wrong due to this mismatch.", UserWarning)
    img0 = img0.as_float()
    total = np.sum(img0, axis=dims)
    img0 = img0.copy()
    img0[~img1] = 0
    
    coeff = np.sum(img0, axis=dims) / total
    return _make_corr_output(coeff, img0, "manders_coloc", squeeze, dims)
    

def iter2(img0: ImgArray, img1: ImgArray, axes: str, israw: bool = False, exclude: str = ""):
    for (sl, i0), (sl, i1) in zip(img0.iter(axes, israw=israw, exclude=exclude),
                                  img1.iter(axes, israw=israw, exclude=exclude)):
        yield sl, i0, i1
        
def _check_inputs(img0: ImgArray, img1: ImgArray):
    _check_dimensions(img0, img1)

    img0 = img0.as_float()
    img1 = img1.as_float()
        
    return img0, img1

def _check_dimensions(img0: ImgArray, img1: ImgArray):
    if img0.shape != img1.shape:
        raise ValueError(f"Shape mismatch. `img0` has shape {img0.shape} but `img1` "
                         f"has shape {img1.shape}")
    if img0.axes != img1.axes:
        warn(f"Axes mismatch. `img0` has axes {img0.axes} but `img1` has axes {img1.axes}. "
              "Result may be wrong due to this mismatch.", UserWarning)
    return

def _make_corr_output(corr: np.ndarray, refimg: ImgArray, propname: str, squeeze: bool, dims: str):
    if corr.ndim == 0 and squeeze:
        corr = corr[()]
    else:
        corr = PropArray(corr, name=refimg.name, axes=complement_axes(dims, refimg.axes), 
                        dirpath=refimg.dirpath, metadata=refimg.metadata, 
                        propname=propname, dtype=np.float32)
    return corr
