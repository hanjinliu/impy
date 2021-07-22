from __future__ import annotations
from scipy import ndimage as ndi
import numpy as np
from functools import partial
from warnings import warn
from .arrays import ImgArray, PropArray
from .arrays.utils import _docs
from .utils.axesop import *
from .utils.utilcls import Progress
from .utils.deco import dims_to_spatial_axes

__all__ = ["fsc", "fourier_shell_correlation", "ncc", "zncc", "fourier_ncc", "fourier_zncc",
           "pearson_coloc", "manders_coloc"]

@_docs.write_docs
@dims_to_spatial_axes
def fsc(img0:ImgArray, img1:ImgArray, nbin:int=32, r_max:float=None, *, squeeze:bool=True,
        dims=None) -> PropArray:
    """
    Calculate Fourier Shell Correlation (FSC; or Fourier Ring Correlation, FRC, for 2-D images) 
    between two images. FSC is defined as:
    
                      Re{Σ'[F0(r') x F1(r)*]}
        FSC(r) = --------------------------------- (Σ' is summation of all r < r'< r+dr)
                  sqrt{Σ'|F0(r')|^2 x Σ'|F1(r')|}
                  
    Parameters
    ----------
    {inputs_of_correlation}
    nbin : int, default is 32
        Number of bins.
    r_max : float, optional
        Maximum radius to make profile. Region 0 <= r < r_max will be split into `nbin` rings
        (or shells). **Scale must be considered** because scales of each axis may vary.
    {squeeze}
    {dims}
                
    Returns
    -------
    PropArray
        FSC stored in x-axis by default. If input images have tzcyx-axes, then an array with 
        tcx-axes will be returned. Make sure x-axis no longer means length in x because images
        are Fourier transformed.
    """    
    img0, img1 = _check_inputs(img0, img1)
    
    spatial_shape = img0.sizesof(dims)
    inds = np.indices(spatial_shape)
    
    center = [s/2 for s in spatial_shape]
    
    r = np.sqrt(sum(((x - c)/img0.scale[a])**2 for x, c, a in zip(inds, center, dims)))
    r_lim = r.max()
        
    # check r_max
    if r_max is None:
        r_max = r_lim
    elif r_max > r_lim or r_max <= 0:
        raise ValueError(f"`r_max` must be in range of 0 < r_max <= {r_lim} with this image.")
    
    with Progress("fsc"):
        # make radially separated labels
        r_rel = r/r_max
        labels = (nbin * r_rel).astype(np.uint16)
        labels[r_rel >= 1] = 0
        
        c_axes = complement_axes(dims, img0.axes)
        
        out = PropArray(np.empty(img0.sizesof(c_axes)+(labels.max(),)), dtype=np.float32, axes=c_axes+dims[-1], 
                        dirpath=img0.dirpath, metadata=img0.metadata, propname="fsc")
        radial_sum = partial(ndi.sum_labels, labels=labels, index=np.arange(1, labels.max()+1))
        f0 = img0.fft(dims=dims)
        f1 = img1.fft(dims=dims)
        
        for sl, f0_, f1_ in iter2(f0, f1, c_axes, exclude=dims):
            cov = f0_.real*f1_.real + f0_.imag*f1_.imag
            pw0 = f0_.real**2 + f0_.imag**2
            pw1 = f1_.real**2 + f1_.imag**2
        
            out[sl] = radial_sum(cov)/np.sqrt(radial_sum(pw0)*radial_sum(pw1))
        
    if out.ndim == 0 and squeeze:
        out = out[()]
    
    return out

# alias
fourier_shell_correlation = fsc


def _ncc(img0:ImgArray, img1:ImgArray, dims):
    # Basic Normalized Cross Correlation with batch processing
    n = np.prod(img0.sizesof(dims))
    return np.sum(img0 * img1, axis=dims) / (
        np.std(img0, axis=dims)*np.std(img1, axis=dims)) / n

def _masked_ncc(img0:ImgArray, img1:ImgArray, dims, mask:ImgArray):
    if mask.ndim < img0.ndim:
        mask = add_axes(img0.axes, img0.shape, mask, mask.axes)
    n = np.prod(img0.sizesof(dims))
    img0ma = np.ma.array(img0, mask=mask)
    img1ma = np.ma.array(img1, mask=mask)
    axis = tuple(img0.axisof(a) for a in dims)
    return np.ma.sum(img0ma * img1ma, axis=axis) / (
        np.ma.std(img0ma, axis=axis)*np.ma.std(img1ma, axis=axis)) / n

@_docs.write_docs
@dims_to_spatial_axes
def ncc(img0:ImgArray, img1:ImgArray, mask:ImgArray|None=None, squeeze:bool=True, *, 
        dims=None) -> PropArray|float:
    """
    Normalized Cross Correlation.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate correlation.
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
def zncc(img0:ImgArray, img1:ImgArray, mask:ImgArray|None=None, squeeze:bool=True, *,
         dims=None) -> PropArray|float:
    """
    Zero-Normalized Cross Correlation.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate correlation.
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
            corr = _ncc(img0zn, img1zn, dims)
        else:
            corr = _masked_ncc(img0zn, img1zn, dims, mask)
    return _make_corr_output(corr, img0, "zncc", squeeze, dims)


@_docs.write_docs
@dims_to_spatial_axes
def fourier_ncc(img0:ImgArray, img1:ImgArray, mask:ImgArray|None=None, squeeze:bool=True, *, 
                dims=None) -> PropArray|float:
    """
    Normalized Cross Correlation in Fourier space.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate correlation.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    with Progress("fourier_ncc"):
        img0, img1 = _check_inputs(img0, img1)
        f0 = np.sqrt(img0.power_spectra(dims=dims))
        f1 = np.sqrt(img1.power_spectra(dims=dims))
        if mask is None:
            corr = _ncc(f0, f1, dims)
        else:
            corr = _masked_ncc(f0, f1, dims, mask)
    return _make_corr_output(corr, img0, "fourier_ncc", squeeze, dims)

@_docs.write_docs
@dims_to_spatial_axes
def fourier_zncc(img0:ImgArray, img1:ImgArray, mask:ImgArray|None=None, squeeze:bool=True, *,
                 dims=None) -> PropArray|float:
    """
    Zero-Normalized Cross Correlation in Fourier space.
    
    Parameters
    ----------
    {inputs_of_correlation}
    mask : boolean ImgArray, optional
        If provided, True regions will be masked and will not be taken into account when calculate correlation.
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    with Progress("fourier_zncc"):
        img0, img1 = _check_inputs(img0, img1)
        f0 = np.sqrt(img0.power_spectra(dims=dims))
        f1 = np.sqrt(img1.power_spectra(dims=dims))
        f0 -= np.mean(f0, axis=dims, keepdims=True)
        f1 -= np.mean(f1, axis=dims, keepdims=True)
        if mask is None:
            corr = _ncc(f0, f1, dims)
        else:
            corr = _masked_ncc(f0, f1, dims, mask)
    return _make_corr_output(corr, img0, "fourier_zncc", squeeze, dims)

@_docs.write_docs
@dims_to_spatial_axes
def pcc(img0:ImgArray, img1:ImgArray, squeeze:bool=True, *,
        dims=None) -> PropArray|float:
    """
    Phase Cross Correlation.
    
    Parameters
    ----------
    {inputs_of_correlation}
    {squeeze}
    {dims}

    Returns
    -------
    PropArray or float
        Correlation value(s).
    """    
    with Progress("pcc"):
        img0, img1 = _check_inputs(img0, img1)
        f0 = img0.fft(dims=dims)
        f1 = img1.fft(dims=dims)
        cov = f0 * f1.conj()
        corr_ft = cov/np.abs(cov)
        corr = corr_ft.ifft(real=True, dims=dims)
    return _make_corr_output(corr, img0, "pcc", squeeze, dims)

@_docs.write_docs
@dims_to_spatial_axes
def pearson_coloc(img0:ImgArray, img1:ImgArray, mask:np.ndarray=None, squeeze:bool=True, *,
                  dims=None) -> PropArray|float:
    """
    Masked Pearson's correlation coefficient. This is defined as following:
    
                  Σ[(Ai - Amean)(Bi - Bmean)]
        r = -----------------------------------------
             sqrt{Σ[(Ai - Amean)^2 Σ(Bi - Bmean)^2]}
    
    This value is independent of constant background intensity and the scale of intensity, 
    while is strongly affected by outliers.

    Parameters
    ----------
    {inputs_of_correlation}
    {squeeze}
    {dims}
    
    Returns
    -------
    PropArray or float
        Correlation coefficient(s).
    
    Examples
    --------
    (1) Make a mask by thresholding and calculate correlation.
    >>> mask = ~img.threshold()
    >>> coeff = img.pcc(mask=mask) 
    """        
    img0, img1 = _check_inputs(img0, img1)
    sumaxes = tuple(img0.axisof(a) for a in dims)
    img0_norm = img0 - np.mean(img0)
    img1_norm = img1 - np.mean(img1)
    if mask is not None:
        img0_norm[mask] = 0
        img1_norm[mask] = 0
    cov = np.sum(img0_norm * img1_norm, axis=sumaxes)
    var0 = np.sum(img0_norm**2, axis=sumaxes)
    var1 = np.sum(img1_norm**2, axis=sumaxes)
    out = cov / np.sqrt(var0 * var1)
    
    return _make_corr_output(out, img0, "pearson_coloc", squeeze, dims)


@_docs.write_docs
@dims_to_spatial_axes
def manders_coloc(img0:ImgArray, img1:np.ndarray, *, squeeze:bool=True, dims=None) -> PropArray|float:
    """
    Manders' correlation coefficient. This is defined as following:
    
             Σ(Ai[ref])
        r = -----------
               Σ(Ai)
    
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
    img0, img1 = _check_inputs(img0, img1)
    sumaxes = tuple(img0.axisof(a) for a in dims)
    total = np.sum(img0.value, axis=sumaxes)
    img0 = img0.copy()
    img0[~img1] = 0
    
    coeff = np.sum(img0.value, axis=sumaxes) / total
    return _make_corr_output(coeff, img0, "manders_coloc", squeeze, dims)
    

def iter2(img0, img1, axes, israw=False, exclude=""):
    for (sl, i0), (sl, i1) in zip(img0.iter(axes, israw=israw, exclude=exclude),
                                  img1.iter(axes, israw=israw, exclude=exclude)):
        yield sl, i0, i1
        
def _check_inputs(img0, img1):
    if img0.shape != img1.shape:
        raise ValueError(f"Shape mismatch. `img0` has shape {img0.shape} but `img1` "
                         f"has shape {img1.shape}")
    if img0.axes != img1.axes:
        warn(f"Axes mismatch. `img0` has axes {img0.axes} but `img1` has axes {img1.axe}. "
              "Result may be wrong due to this mismatch.", UserWarning)
    img0 = img0.as_float()
    img1 = img1.as_float()
        
    return img0, img1

def _make_corr_output(corr:np.ndarray, refimg:ImgArray, propname:str, squeeze:bool, dims:str):
    if corr.ndim == 0 and squeeze:
        corr = corr[()]
    else:
        corr = PropArray(corr, name=refimg.name, axes=complement_axes(dims, refimg.axes), 
                        dirpath=refimg.dirpath, metadata=refimg.metadata, 
                        propname=propname, dtype=np.float32)
    return corr
