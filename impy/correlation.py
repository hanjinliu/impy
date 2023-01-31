from __future__ import annotations
import numpy as np
from warnings import warn
from .core import asarray as ip_asarray, circular_mask
from .arrays import ImgArray, PropArray
from .arrays._utils import _docs
from .arrays._utils._transform import polar2d
from .arrays._utils._corr import subpixel_pcc, subpixel_ncc, draw_pcc_landscape, draw_ncc_landscape
from .utils.axesop import complement_axes, add_axes
from .utils.deco import dims_to_spatial_axes
from .array_api import xp
from ._types import Dims

__all__ = [
    "fsc",
    "fourier_shell_correlation",
    "ncc",
    "zncc",
    "fourier_ncc",
    "fourier_zncc",
    "nmi",
    "pcc_maximum",
    "pcc_maximum_with_corr",
    "ft_pcc_maximum",
    "ft_pcc_maximum_with_corr", 
    "polar_pcc_maximum",
    "polar_pcc_maximum_with_corr",
    "zncc_maximum",
    "zncc_maximum_with_corr",
    "pcc_landscape", 
    "ft_pcc_landscape", 
    "zncc_landscape",
    "pearson_coloc",
    "manders_coloc",
]

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
    
    freqs = xp.meshgrid(*[xp.fft.fftshift(xp.fft.fftfreq(s)) for s in shape], indexing="ij")
    r = xp.sqrt(sum(f**2 for f in freqs))

    # make radially separated labels
    labels = (r/dfreq).astype(np.uint16)
    nlabels = int(xp.asnumpy(labels.max()))
    
    out = xp.empty(nlabels, dtype=np.float32)
    def radial_sum(arr):
        arr = xp.asarray(arr)
        return xp.ndi.sum_labels(arr, labels=labels, index=xp.arange(0, nlabels))

    f0 = img0.fft(dims=dims)
    f1 = img1.fft(dims=dims)
    
    cov = f0.real*f1.real + f0.imag*f1.imag
    pw0 = f0.real**2 + f0.imag**2
    pw1 = f1.real**2 + f1.imag**2

    out = radial_sum(cov)/xp.sqrt(radial_sum(pw0) * radial_sum(pw1))
    freq = (np.arange(len(out)) + 0.5) * dfreq
    return freq, xp.asnumpy(out)

# alias
fourier_shell_correlation = fsc


def _ncc(img0: ImgArray, img1: ImgArray, dims: Dims):
    # Basic Normalized Cross Correlation with batch processing
    n = np.prod(img0.sizesof(dims))
    dims = tuple(img0.axisof(a) for a in dims)
    img0 = xp.asarray(img0)
    img1 = xp.asarray(img1)
    corr = xp.sum(img0 * img1, axis=dims) / (
        xp.std(img0, axis=dims)*xp.std(img1, axis=dims)) / n
    return xp.asnumpy(corr)


def _masked_ncc(img0: ImgArray, img1: ImgArray, dims: Dims, mask: xp.ndarray):
    n = np.prod(img0.sizesof(dims))
    img0ma = np.ma.array(img0.value, mask=mask)
    img1ma = np.ma.array(img1.value, mask=mask)
    axis = tuple(img0.axisof(a) for a in dims)
    return np.ma.sum(img0ma * img1ma, axis=axis) / (
        np.ma.std(img0ma, axis=axis)*np.ma.std(img1ma, axis=axis)) / n


def _zncc(img0: xp.ndarray, img1: xp.ndarray, dims: tuple[int, ...]):
    # Basic Zero-mean Normalized Cross Correlation with batch processing.
    # Inputs must be already zero-normalized.
    corr = xp.sum(img0 * img1, axis=dims) / (
        xp.sqrt(xp.sum(img0**2, axis=dims)*xp.sum(img1**2, axis=dims)))
    return corr


def _masked_zncc(img0: xp.ndarray, img1: xp.ndarray, dims: tuple[int, ...], mask: ImgArray):
    if xp.state == "cupy":
        img0 = img0*mask
        img1 = img1*mask
        out = xp.sum(img0*img1, axis=dims) / (
            xp.sqrt(xp.sum(img0**2, axis=dims)*xp.sum(img1**2, axis=dims)))
    else:
        img0ma = np.ma.array(img0, mask=mask)
        img1ma = np.ma.array(img1, mask=mask)
        out = np.ma.sum(img0ma * img1ma, axis=dims) / (
            np.sqrt(np.ma.sum(img0ma**2, axis=dims)*np.ma.sum(img1ma**2, axis=dims)))
    return out

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
    img0, img1 = _check_inputs(img0, img1)
    if mask is None:
        corr = _ncc(img0, img1, dims)
    else:
        if img0.ndim > mask.ndim:
            mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
        corr = _masked_ncc(img0, img1, dims, np.asarray(mask))
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
    Zero-mean Normalized Cross Correlation.
    
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
    img0, img1 = _check_inputs(img0, img1)
    _dims = tuple(img0.axisof(a) for a in dims)
    _img0 = xp.asarray(img0.value)
    _img1 = xp.asarray(img1.value)
    img0zn = _img0 - xp.mean(_img0, axis=_dims, keepdims=True)
    img1zn = _img1 - xp.mean(_img1, axis=_dims, keepdims=True)
    if mask is None:
        corr = _zncc(img0zn, img1zn, _dims)
    else:
        if img0.ndim > mask.ndim:
            mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
        corr = _masked_zncc(img0zn, img1zn, _dims, xp.asarray(np.asarray(mask)))
    return _make_corr_output(xp.asnumpy(corr), img0, "zncc", squeeze, dims)

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
    img0, img1 = _check_inputs(img0, img1)
    f0 = np.sqrt(img0.power_spectra(dims=dims, zero_norm=True))
    f1 = np.sqrt(img1.power_spectra(dims=dims, zero_norm=True))
    if mask is None:
        corr = _ncc(f0, f1, dims)
    else:
        if img0.ndim > mask.ndim:
            mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
        corr = _masked_ncc(f0, f1, dims, xp.asarray(np.asarray(mask)))
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
    Zero-mean Normalized Cross Correlation in Fourier space.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account 
        when calculate correlation.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    img0, img1 = _check_inputs(img0, img1)
    pw0 = xp.asarray(img0.power_spectra(dims=dims, zero_norm=True).value)
    pw1 = xp.asarray(img1.power_spectra(dims=dims, zero_norm=True).value)
    
    _dims = tuple(img0.axisof(a) for a in dims)
    f0 = xp.sqrt(pw0)
    f1 = xp.sqrt(pw1)
    f0 -= xp.mean(f0, axis=_dims, keepdims=True)
    f1 -= xp.mean(f1, axis=_dims, keepdims=True)
    if mask is None:
        corr = _zncc(f0, f1, _dims)
    else:
        if img0.ndim > mask.ndim:
            mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
        corr = _masked_zncc(f0, f1, _dims, xp.asarray(np.asarray(mask)))
    return _make_corr_output(xp.asnumpy(corr), img0, "fourier_zncc", squeeze, dims)

@_docs.write_docs
def pcc_maximum(
    img0: ImgArray, 
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    upsample_factor: int = 10,
    max_shifts: int | tuple[int, ...] | None = None
) -> np.ndarray:
    """
    Calculate lateral shift between two images using phase cross correlation.
    
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
    return pcc_maximum_with_corr(img0, img1, mask, upsample_factor, max_shifts)[0]

@_docs.write_docs
def pcc_maximum_with_corr(
    img0: ImgArray, 
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    upsample_factor: int = 10,
    max_shifts: int | tuple[int, ...] | None = None
) -> np.ndarray:
    """
    Calculate lateral shift between two images using phase cross correlation.
    
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
    np.ndarray and float
        Shift in pixel and phase cross correlation
    """    
    if img0 is img1:
        return np.zeros(img0.ndim)
    img0, img1 = _check_inputs(img0, img1)
    ft0 = img0.fft(dims=img0.axes)
    ft1 = img1.fft(dims=img1.axes)
    if mask is not None:
        ft0[mask] = 0
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    shift, pcc = subpixel_pcc(
        xp.asarray(ft0.value), 
        xp.asarray(ft1.value),
        upsample_factor, 
        max_shifts=max_shifts
    )
    return xp.asnumpy(shift), float(pcc)

@_docs.write_docs
def ft_pcc_maximum(
    img0: ImgArray,
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    upsample_factor: int = 10,
    max_shifts: float | tuple[float, ...] | None = None
) -> np.ndarray:
    """
    Calculate lateral shift between two images using phase cross correlation.
    
    This function takes Fourier transformed images as input. If you have to repetitively
    use a same template image, this function is faster.

    Parameters
    ----------
    {inputs_of_correlation}
    upsample_factor : int, default is 10
        Up-sampling factor when calculating phase cross correlation.
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as
        maximum shifts in all dimensions. No upper bound of shifts if not given.

    Returns
    -------
    np.ndarray
        Shift in pixel.
    """    
    return ft_pcc_maximum_with_corr(img0, img1, mask, upsample_factor, max_shifts)[0]

@_docs.write_docs
def ft_pcc_maximum_with_corr(
    img0: ImgArray,
    img1: ImgArray, 
    mask: ImgArray | None = None, 
    upsample_factor: int = 10,
    max_shifts: float | tuple[float, ...] | None = None
) -> np.ndarray:
    """
    Calculate lateral shift between two images using phase cross correlation.
    
    This function takes Fourier transformed images as input. If you have to repetitively
    use a same template image, this function is faster.

    Parameters
    ----------
    {inputs_of_correlation}
    upsample_factor : int, default is 10
        Up-sampling factor when calculating phase cross correlation.
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as
        maximum shifts in all dimensions. No upper bound of shifts if not given.

    Returns
    -------
    np.ndarray and float
        Shift in pixel and phase cross correlation.
    """    
    _check_dimensions(img0, img1)
    if mask is not None:
        img0 = img0.copy()
        img0[mask] = 0
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    shift, pcc = subpixel_pcc(
        xp.asarray(img0.value), 
        xp.asarray(img1.value),
        upsample_factor,
        max_shifts=max_shifts,
    )
    return xp.asnumpy(shift), float(pcc)

@_docs.write_docs
def pcc_landscape(
    img0: ImgArray, 
    img1: ImgArray,
    max_shifts: int | tuple[int, ...] | None = None,
):
    """
    Create landscape of phase cross correlation.

    Parameters
    ----------
    {inputs_of_correlation}
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as
        maximum shifts in all dimensions. No upper bound of shifts if not given.

    Returns
    -------
    ImgArray
        Landscape image.
    """    
    _check_dimensions(img0, img1)
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    
    ft0 = img0.fft(dims=img0.axes)
    ft1 = img1.fft(dims=img1.axes)
    landscape = draw_pcc_landscape(
        xp.asarray(ft0.value), 
        xp.asarray(ft1.value),
        max_shifts=max_shifts
    )
    return ip_asarray(xp.asnumpy(landscape), axes=img1.axes)

@_docs.write_docs
def ft_pcc_landscape(
    img0: ImgArray, 
    img1: ImgArray,
    max_shifts: int | tuple[int, ...] | None = None,
):
    """
    Create landscape of phase cross correlation.

    This function takes Fourier transformed images as input. If you have to repetitively
    use a same template image, this function is faster.
    
    Parameters
    ----------
    {inputs_of_correlation}
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as
        maximum shifts in all dimensions. No upper bound of shifts if not given.

    Returns
    -------
    ImgArray
        Landscape image.
    """    
    _check_dimensions(img0, img1)
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim

    landscape = draw_pcc_landscape(
        xp.asarray(img0.value), 
        xp.asarray(img1.value),
        max_shifts=max_shifts
    )
    return ip_asarray(xp.asnumpy(landscape), axes=img1.axes)

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
    return polar_pcc_maximum_with_corr(img0, img1, upsample_factor, max_degree)[0]

@_docs.write_docs
def polar_pcc_maximum_with_corr(
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
    float and float
        Rotation in degree and phase cross correlation
    """    
    return _polar_corr(pcc_maximum_with_corr, img0, img1, upsample_factor, max_degree)


@_docs.write_docs
def polar_zncc_maximum(
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
    return polar_zncc_maximum_with_corr(img0, img1, upsample_factor, max_degree)[0]

@_docs.write_docs
def polar_zncc_maximum_with_corr(
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
    float and float
        Rotation in degree and phase cross correlation
    """    
    return _polar_corr(zncc_maximum_with_corr, img0, img1, upsample_factor, max_degree)


def _polar_corr(
    corr_fn,
    img0: ImgArray,
    img1: ImgArray,
    upsample_factor: int = 10,
    max_degree: int = None,
) -> float:
    """Calculate rotational shift between two images using polar Fourier transformation."""    
    img0, img1 = _check_inputs(img0, img1)
    if img0.ndim != 2:
        raise TypeError("Currently only 2D image is supported.")
    if max_degree is None:
        max_degree = 180
    rmax = min(img0.shape) // 2
    mask = circular_mask(rmax, img0.shape, soft=True, out_value=False)
    imgp = ip_asarray(xp.asnumpy(polar2d(img0 * mask, rmax, np.pi/180)))
    imgrotp = ip_asarray(xp.asnumpy(polar2d(img1 * mask, rmax, np.pi/180)))
    max_shifts = (max_degree, 1)
    shift, pcc = corr_fn(
        imgp, imgrotp, upsample_factor=upsample_factor, max_shifts=max_shifts
    )
    # Here, `shift` satisfies `img0.rotate(-shift[0]) == img1`
    return shift[0], pcc

@_docs.write_docs
def zncc_maximum(
    img0: ImgArray, 
    img1: ImgArray,
    upsample_factor: int = 10,
    max_shifts: int | tuple[int, ...] | None = None
) -> np.ndarray:
    """
    Calculate lateral shift between two images using zero-mean normalized cross correlation.
    
    Similar to :func:`pcc_maximum`, this function can determine shift at sub-pixel precision.
    Since ZNCC uses real space, this function performs better than PCC when the input images
    have frequency loss.
    Unlike :func:`zncc_maximum_with_corr`, this function only returns the optimal shift.
    Parameters
    ----------
    {inputs_of_correlation}
    upsample_factor : int, default is 10
        Up-sampling factor when calculating cross correlation. Convolution image will be
        up-sampled by third-order interpolation.
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as
        maximum shifts in all dimensions. No upper bound of shifts if not given.
    Returns
    -------
    np.ndarray
        Shift in pixel .
    """    
    return zncc_maximum_with_corr(img0, img1, upsample_factor, max_shifts)[0]

@_docs.write_docs
def zncc_maximum_with_corr(
    img0: ImgArray, 
    img1: ImgArray,
    upsample_factor: int = 10,
    max_shifts: int | tuple[int, ...] | None = None
) -> tuple[np.ndarray, float]:
    """
    Calculate lateral shift between two images using zero-mean normalized cross correlation.
    
    Similar to :func:`pcc_maximum`, this function can determine shift at sub-pixel precision.
    Since ZNCC uses real space, this function performs better than PCC when the input images
    have frequency loss.
    Parameters
    ----------
    {inputs_of_correlation}
    upsample_factor : int, default is 10
        Up-sampling factor when calculating cross correlation. Convolution image will be
        up-sampled by third-order interpolation.
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as
        maximum shifts in all dimensions. No upper bound of shifts if not given.
    Returns
    -------
    np.ndarray and float
        Shift in pixel and ZNCC value.
    """    
    if img0 is img1:
        return np.zeros(img0.ndim), 1.
    img0, img1 = img0.astype(np.float32), img1.astype(np.float32)
    img0z = img0 - img0.mean()
    img1z = img1 - img1.mean()
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    shift, zncc = subpixel_ncc(
        xp.asarray(np.asarray(img0z)), 
        xp.asarray(np.asarray(img1z)),
        upsample_factor, 
        max_shifts=max_shifts
    )
    return xp.asnumpy(shift), float(zncc)

@_docs.write_docs
def zncc_landscape(
    img0: ImgArray, 
    img1: ImgArray,
    max_shifts: int | tuple[int, ...] | None = None,
):
    """
    Create landscape of zero-mean normalized cross correlation.

    Parameters
    ----------
    {inputs_of_correlation}
    max_shifts : float, tuple of float, optional
        Maximum shifts in each dimension. If a single scalar is given, it is interpreted as
        maximum shifts in all dimensions. No upper bound of shifts if not given.

    Returns
    -------
    ImgArray
        Landscape image.
    """    
    img0, img1 = img0.astype(np.float32), img1.astype(np.float32)
    img0z = img0 - img0.mean()
    img1z = img1 - img1.mean()
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    landscape = draw_ncc_landscape(
        xp.asarray(np.asarray(img0z)), 
        xp.asarray(np.asarray(img1z)),
        max_shifts=max_shifts
    )
    return ip_asarray(xp.asnumpy(landscape), axes=img1.axes)

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
        corr = corr.item()
    else:
        corr = PropArray(corr, name=refimg.name, axes=complement_axes(dims, refimg.axes), 
                        source=refimg.source, metadata=refimg.metadata, 
                        propname=propname, dtype=np.float32)
    return corr
