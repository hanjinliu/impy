from __future__ import annotations
from scipy import ndimage as ndi
import numpy as np
from functools import partial
from .deco import dims_to_spatial_axes
from .arrays import ImgArray, PropArray
from .func import *
from .utilcls import *
from ._const import SetConst
from warnings import warn

# TODO:
__all__ = ["fsc", "fourier_shell_correlation", "angular_correlation", "pearson_coloc", "manders_coloc"]

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
    img0 : ImgArray
        First image
    img1 : ImgArray
        Second image
    nbin : int, default is 32
        Number of bins.
    r_max : float, optional
        Maximum radius to make profile. Region 0 <= r < r_max will be split into `nbin` rings
        (or shells). **Scale must be considered** because scales of each axis may vary.
    squeeze : bool, default is True
        If True and output can be converted to scalar, then a float value will be returned.
    dims : str or int, optional
        Spatial dimensions.
            
    Returns
    -------
    PropArray
        FSC stored in x-axis by default. If input images have tzcyx-axes, then an array with 
        tcx-axes will be returned. Make sure x-axis no longer means length in x because images
        are Fourier transformed.
    """    
    _assert_same_dims(img0, img1)
    
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

fourier_shell_correlation = fsc

@dims_to_spatial_axes
def angular_correlation(img0:ImgArray, img1:ImgArray, deg:float, center="center", *, squeeze:bool=True,
                        dims="yx") -> PropArray|float:
    """
    Parameters
    ----------
    img0 : ImgArray
        First image.
    img1 : ImgArray
        Second image. This image will be rotated around the center with degree `deg`.
    deg : float
        Degree (not radian!) to rotate.
    center : array-like of float, default is the center of image.
        Rotation center.
    squeeze : bool, default is True
        If True and output can be converted to scalar, then a float value will be returned.
    dims : int or str, default is "yx"
        Spatial dimensions.

    Returns
    -------
    PropArray or float
        Correlation.
        
    Reference
    ---------
    Blestel, S., Kervrann, C., & Chrétien, D. (2009). A Fourier-Based method for detecting curved microtubule 
    centers: Application to straightening of cryo-Electron microscope images. Proceedings - 2009 IEEE
    International Symposium on Biomedical Imaging: From Nano to Macro, ISBI 2009, 3(1), 298–301.
    https://doi.org/10.1109/ISBI.2009.5193043
    """    
    _assert_same_dims(img0, img1)
    sl = []
    for a in img0.axes:
        if a in dims:
            sl.append(np.newaxis)
        else:
            sl.append(slice(None))
    sl = tuple(sl)
    
    with SetConst("SHOW_PROGRESS", False):
        f1 = np.sqrt(img0.power_spectra(dims=dims))
        f2 = np.sqrt(img1.rotate(deg, center=center).power_spectra(dims=dims))
        f1 -= np.mean(f1, axis=dims)[sl]
        f2 -= np.mean(f2, axis=dims)[sl]
        cov = (f1 - np.mean(f1, axis=dims))*(f2 - np.mean(f2, axis=dims))
        corr = np.sum(cov) / (np.std(f1)*np.std(f2))

    if corr.ndim == 0 and squeeze:
        corr = corr[()]
    else:
        corr = PropArray(corr, name=img0.name, axes=complement_axes(dims, img0.axes), 
                         dirpath=img0.dirpath, metadata=img0.metadata, 
                         propname="angular_correlation", dtype=np.float32)
    
    return corr

@dims_to_spatial_axes
def pearson_coloc(img0:ImgArray, img1:ImgArray, mask:np.ndarray=None, *, squeeze:bool=True, 
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
    mask : np.ndarray, optional
        If given, pixels with True value will not be account for correlation. If MetaArray,
        this array will be broadcasted.
    along : str, optional
        Which axis will be the channel axis.
    squeeze : bool, default is True
        If True and output can be converted to scalar, then a float value will be returned.
    dims : int or str, optional
        Spatial dimensions.

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
    _assert_same_dims(img0, img1)
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
    
    if out.ndim == 0 and squeeze:
        out = out[()]
    else:
        out = PropArray(out, name=img0.name, axes=complement_axes(dims, img0.axes), 
                        dirpath=img0.dirpath, metadata=img0.metadata, 
                        propname="pearson_coloc", dtype=np.float32)
    return out


@dims_to_spatial_axes
def manders_coloc(img:ImgArray, ref:np.ndarray, *, squeeze:bool=True, dims=None) -> PropArray|float:
    """
    Manders' correlation coefficient. This is defined as following:
    
             Σ(Ai[ref])
        r = -----------
               Σ(Ai)
    
    This value is NOT independent of background intensity. You need to correctly subtract
    background from self. This value is NOT interchangable between channels.
    
    Parameters
    ----------
    img : ImgArray
        Input image.
    ref : np.ndarray
        Reference image to calculate coefficent. If MetaArray, this array will be broadcasted.
    squeeze : bool, default is True
        If True and output can be converted to scalar, then a float value will be returned.
    dims : int or str, optional
        Spatial dimensions.

    Returns
    -------
    PropArray or float
        Correlation coefficient(s).
    """        
    if ref.dtype != bool:
        raise TypeError("`ref` must be a binary image.")
    
    sumaxes = tuple(img.axisof(a) for a in dims)
    total = np.sum(img.value, axis=sumaxes)
    img = img.copy()
    img[~ref] = 0
    
    out = np.sum(img.value, axis=sumaxes) / total
    
    if out.ndim == 0 and squeeze:
        out = out[()]
    else:
        out = PropArray(out, name=img.name, axes=complement_axes(dims, img.axes), 
                        dirpath=img.dirpath, metadata=img.metadata, 
                        propname="manders_coloc", dtype=np.float32)
    return out

def iter2(img0, img1, axes, israw=False, exclude=""):
    for (sl, i0), (sl, i1) in zip(img0.iter(axes, israw=israw, exclude=exclude),
                                  img1.iter(axes, israw=israw, exclude=exclude)):
        yield sl, i0, i1
        
def _assert_same_dims(img0, img1):
    if img0.shape != img1.shape:
        raise ValueError(f"Shape mismatch. `img0` has shape {img0.shape} but `img1` "
                         f"has shape {img1.shape}")
    if img0.axes != img1.axes:
        warn(f"Axes mismatch. `img0` has axes {img0.axes} but `img1` has axes {img1.axe}. "
              "Result may be wrong due to this mismatch.")
    return None
    