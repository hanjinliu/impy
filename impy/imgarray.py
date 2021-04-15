from __future__ import annotations
import itertools
import numpy as np
import os
import glob
import collections
from skimage import io
from skimage import morphology as skmorph
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import restoration as skres
from skimage import exposure as skexp
from skimage import measure as skmes
from skimage import segmentation as skseg
from skimage.feature.corner import _symmetric_image
from skimage import feature as skfeat
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft
from scipy import ndimage as ndi
from scipy import optimize as opt
from .func import *
from .deco import *
from .gauss import GaussianBackground, GaussianParticle
from .labeledarray import LabeledArray
from .label import Label
from .axes import Axes, ImageAxesError
from .proparray import PropArray
from .utilcls import *

def _affine(args):
    sl, data, mx, order = args
    return (sl, sktrans.warp(data, mx, order=order))

def _median(args):
    sl, data, selem = args
    return (sl, skfil.rank.median(data, selem))

def _mean(args):
    sl, data, selem = args
    return (sl, skfil.rank.mean(data, selem))

def _gaussian(args):
    sl, data, sigma = args
    return (sl, ndi.gaussian_filter(data, sigma))

def _entropy(args):
    sl, data, selem = args
    return (sl, skfil.rank.entropy(data, selem))

def _enhance_contrast(args):
    sl, data, selem = args
    return (sl, skfil.rank.enhance_contrast(data, selem))

def _difference_of_gaussian(args):
    sl, data, low_sigma, high_sigma = args
    return (sl, skfil.difference_of_gaussians(data, low_sigma, high_sigma))

def _rolling_ball(args):
    sl, data, radius, smooth = args
    if smooth:
        _, ref = _mean((sl, data, np.ones((3, 3))))
        back = skres.rolling_ball(ref, radius=radius)
        tozero = (back > data)
        back[tozero] = data[tozero]
    else:
        back = skres.rolling_ball(data, radius=radius)
    
    return (sl, data - back)

def _sobel(args):
    sl, data = args
    return (sl, skfil.sobel(data))
    
def _opening(args):
    sl, data, selem = args
    return (sl, skmorph.opening(data, selem))

def _binary_opening(args):
    sl, data, selem = args
    return (sl, skmorph.binary_opening(data, selem))

def _closing(args):
    sl, data, selem = args
    return (sl, skmorph.closing(data, selem))

def _binary_closing(args):
    sl, data, selem = args
    return (sl, skmorph.binary_closing(data, selem))

def _erosion(args):
    sl, data, selem = args
    return (sl, skmorph.erosion(data, selem))

def _binary_erosion(args):
    sl, data, selem = args
    return (sl, skmorph.binary_erosion(data, selem))

def _dilation(args):
    sl, data, selem = args
    return (sl, skmorph.dilation(data, selem))

def _binary_dilation(args):
    sl, data, selem = args
    return (sl, skmorph.binary_dilation(data, selem))

def _tophat(args):
    sl, data, selem = args
    return (sl, skmorph.white_tophat(data, selem))

def _skeletonize(args):
    sl, data = args
    return (sl, skmorph.skeletonize_3d(data))

def _hessian_eigh(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return (sl, eigval, eigvec)

def _hessian_eigval(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return (sl, eigval)

def _structure_tensor_eigh(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(tensor)
    return (sl, eigval, eigvec)

def _structure_tensor_eigval(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(tensor)
    return (sl, eigval)

def _label(args):
    sl, data, connectivity = args
    labels = skmes.label(data, background=0, connectivity=connectivity)
    return (sl, labels)

def _distance_transform_edt(args):
    sl, data = args
    return (sl, ndi.distance_transform_edt(data))

    
class ImgArray(LabeledArray):
    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 history=None, metadata=None, lut=None):
        super().__init__(obj, name=name, axes=axes, dirpath=dirpath, 
                         history=history, metadata=metadata, lut=lut)

    @dims_to_spatial_axes
    @same_dtype(True)
    @record()
    def affine(self, *, dims=None, order:int=1, **kwargs) -> ImgArray:
        """
        Affine transformation
        kwargs: matrix, scale, rotation, shear, translation
        """
        if dims != 2:
            raise ValueError("dims != 2 version have yet been implemented")
        mx = sktrans.AffineTransform(**kwargs)
        out = self.parallel(_affine, complement_axes(dims), mx, order)
        return out
    
    @dims_to_spatial_axes
    @same_dtype(True)
    @record()
    def translate(self, translation=None, *, dims=2) -> ImgArray:
        """
        Simple translation of image, i.e. (x, y) -> (x+dx, y+dy)
        """
        mx = sktrans.AffineTransform(translation=translation)
        out = self.parallel(_affine, complement_axes(dims), mx)
        return out

    @same_dtype(True)
    @record()
    def rescale(self, scale:float=1/16, dims=None, order:int=None) -> ImgArray:
        """
        Rescale image.

        Parameters
        ----------
        scale : float, optional
            scale of the new image.
        dims : int or str, optional
            axes to rescale.
        order : float, optional
            order of rescaling.
        """        
        scale_ = [scale if a in dims else 1 for a in self.axes]
        out = sktrans.rescale(self.value, scale_, order=order, anti_aliasing=False)
        return out
    
    @record()
    def gaussfit(self, scale:float=1/16, p0=None, show_result:bool=True) -> ImgArray:
        """
        Fit the image to 2-D Gaussian.

        Parameters
        ----------
        scale : float, optional
            Scale of rough image (to speed up fitting).
        p0 : list or None, optional
            Initial parameters.

        Returns
        -------
        ImgArray
            Fit image.

        Raises
        ------
        TypeError
            If self is not two dimensional.
        """
        if self.ndim != 2:
            raise TypeError(f"input must be two dimensional, but got {self.shape}")
        
        rough = self.rescale(scale).value.astype("float32")
        gaussian = GaussianBackground(p0)
        result = gaussian.fit(rough)
        gaussian.rescale(1/scale)
        fit = gaussian.generate(self.shape).view(self.__class__)
        fit.temp = dict(params=gaussian.params, result=result)
        
        # show fitting result
        if show_result:
            x0 = self.shape[1]//2
            y0 = self.shape[0]//2
            plt.figure(figsize=(6,4))
            plt.subplot(2,1,1)
            plt.title("x-direction")
            plt.plot(self[y0].value, color="gray", alpha=0.5, label="raw image")
            plt.plot(fit[y0], color="red", label="fit")
            plt.subplot(2,1,2)
            plt.title("y-direction")
            plt.plot(self[:,x0].value, color="gray", alpha=0.5, label="raw image")
            plt.plot(fit[:,x0], color="red", label="fit")
            plt.tight_layout()
            plt.show()
        return fit
    
    @record()
    def gaussfit_particle(self, center, width=9, p0=None) -> ImgArray:
        # TODO: more useful way? Like peak_local_max().
        center = np.array(center)
        gaussian = GaussianParticle(p0)
        x0, y0 = center - width // 2
        x1, y1 = center + (width+1) // 2
        result = gaussian.fit(self.value[x0:x1, y0:y1])
        gaussian.shift([x0, y0])
        fit = gaussian.generate(self.shape).view(__class__)
        fit.temp = dict(params=gaussian.params, result=result)

        return fit
    
    @record()
    def affine_correction(self, ref=None, bins:int=256, 
                          order:int=3, prefilter:bool=True, axis:str="c") -> ImgArray:
        """
        Correct chromatic aberration using Affine transformation. Input matrix is
        determined by maximizing normalized mutual information.
        
        Parameters
        ----------
        ref : int, optional
            Reference image-stack to calculate Affine transformation matrices, or
            matrices themselves.
        bins : int, optional
            Number of bins that is generated on calculating mutual information, 
            by default 256
        order : int, optional
            Interporation order, by default 3
        prefilter : bool
            If median filter is applied to all images before fitting. This does not
            change original images. By default True.
        axis : str
            Along which axis correction will be performed, by default "c".
            Chromatic aberration -> "c"
            Some drift during time lapse movie -> "t"

        Returns
        -------
        ImgArray
            Corrected image.
        """
        def check_c_axis(self):
            if not hasattr(self, "axes"):
                raise AttributeError("Image dose not have axes.")
            elif axis not in self.axes:
                raise ValueError("Image does not have channel axis.")
            elif self.sizeof(axis) < 2:
                raise ValueError("Image must have two channels or more.")
        
        check_c_axis(self)
        
        mtx = None
        
        # check `ref`
        if ref is None:
            # correct self
            ref = self
            
        elif isinstance(ref, np.ndarray):
            # ref is single Affine transformation matrix or a reference image stack.
            if ref.shape == (3, 3) and np.allclose(ref[2,:2], 0):
                mtx = [1, ref]
            else:
                check_c_axis(ref)
                
        elif isinstance(ref, (list, tuple)):
            # ref is a list of Affine transformation matrix
            mtx = []
            for m in ref:
                if isinstance(m, (int, float)): 
                    if m == 1:
                        mtx.append(m)
                    else:
                        raise ValueError(f"Only `1` is ok, but got {m}")
                    
                elif m.shape != (3, 3) or not np.allclose(m[2,:2], 0):
                    raise ValueError(f"Wrong Affine transformation matrix:\n{m}")
                
                else:
                    mtx.append(m)
        
        else:
            raise TypeError("`ref` must be image or (list of) Affine transformation matrices.")
        
        
        # Determine matrices by fitting
        if mtx is None:
            if prefilter:
                imgs = [img for img in ref.median_filter(radius=1).split(axis)]
            else:
                imgs = [img for img in ref.split(axis)]
                
            print("fitting ... ", end="")
            mtx = [1] + [affinefit(img, imgs[0], bins, order) for img in imgs[1:]]
        
        if len(mtx) != self.sizeof(axis):
            nchn = self.sizeof(axis)
            raise ValueError(f"{nchn}-channel image needs {nchn} matrices.")
        
        corrected = []
        for i, m in enumerate(mtx):
            if isinstance(m, (int, float)) and m==1:
                corrected.append(self[f"{axis}={i+1}"])
            else:
                corrected.append(self[f"{axis}={i+1}"].affine(order=order, matrix=m))

        out = stack(corrected, axis=axis, dtype=self.dtype)
        out.temp = mtx
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def hessian_eigval(self, sigma=1, *, pxsize=None, dims=None) -> list[ImgArray]:
        """
        Calculate Hessian's eigenvalues for each image. If dims=2, every yx-image 
        is considered to be a single spatial image, and if dims=3, zyx-image.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        pxsize : scalar or array (dims,), optional
            Pixel size (to normalize matrix).
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        list of float ImgArray
            eigenvalues in smaller->larger order
        """        
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = check_nd_pxsize(pxsize, ndim)
        eigval = self.as_float().parallel(_hessian_eigval, 
                                          complement_axes(dims), 
                                          sigma, pxsize,
                                          outshape=self.shape+(ndim,))
        
        eigval = list(np.moveaxis(eigval, -1, 0))
        for i, each in enumerate(eigval):
            each._set_info(self, f"{ndim}D-Hessian-eigenvalue[{i}]")
        
        return eigval
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def hessian_eig(self, sigma=1, *, pxsize=None, dims=None) -> tuple:
        """
        Calculate Hessian's eigenvalues and eigenvectors.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        pxsize : scalar or array (dims,), optional
            Pixel size (to normalize matrix).
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        list of float ImgArray and 2D-list of float ImgArray
        """                
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = check_nd_pxsize(pxsize, ndim)
        eigval, eigvec = self.parallel_eig(_hessian_eigh, 
                                           complement_axes(dims), 
                                           sigma, pxsize)
        
        # set information of eigenvalue list
        for i in range(ndim):
            eigval[i] = eigval[i].view(self.__class__)
            eigval[i]._set_info(self, f"{ndim}D-Hessian-eigenvalue[{i}]")
        
        # set information of eigenvector list
        for i in range(ndim):
            for j in range(ndim):
                eigvec[i][j] = eigvec[i][j].view(self.__class__)
                eigvec[i][j]._set_info(self, f"{ndim}D-Hessian-eigenvector[{i}][{dims[j]}]")
                
        return eigval, eigvec
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def structure_tensor_eigval(self, sigma=1, *, pxsize=None, dims=None):
        """
        Calculate structure tensor's eigenvalues and eigenvectors.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        pxsize : scalar or array (dims,), optional
            Pixel size (to normalize matrix).
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        list of float ImgArray and 2D-list of float ImgArray
        """          
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = check_nd_pxsize(pxsize, ndim)
        eigval = self.as_float().parallel(_structure_tensor_eigval, 
                                          complement_axes(dims), 
                                          sigma, pxsize,
                                          outshape=self.shape+(ndim,))
        
        eigval = list(np.moveaxis(eigval, -1, 0))
        for i, each in enumerate(eigval):
            each._set_info(self, f"{ndim}D-Structure-Tensor-eigenvalue[{i}]")
        
        return eigval
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def structure_tensor_eig(self, sigma=1, *, pxsize=None, dims=None):
        """
        Calculate structure tensor's eigenvalues and eigenvectors.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        pxsize : scalar or array (dims,), optional
            Pixel size (to normalize matrix).
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        list of float ImgArray and 2D-list of float ImgArray
        """                
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = check_nd_pxsize(pxsize, ndim)
        eigval, eigvec = self.parallel_eig(_structure_tensor_eigh, 
                                           complement_axes(dims), 
                                           sigma, pxsize)
        
        # set information of eigenvalue list
        for i in range(ndim):
            eigval[i] = eigval[i].view(self.__class__)
            eigval[i]._set_info(self, f"{ndim}D-Structure-Tensor-eigenvalue[{i}]")
        
        # set information of eigenvector list
        for i in range(ndim):
            for j in range(ndim):
                eigvec[i][j] = eigvec[i][j].view(self.__class__)
                eigvec[i][j]._set_info(self, f"{ndim}D-Structure-Tensor-eigenvector[{i}][{dims[j]}]")
                
        return eigval, eigvec
    
    @dims_to_spatial_axes
    @same_dtype()
    @record()
    def sobel_filter(self, dims=None, update:bool=False):
        out = self.parallel(_sobel, complement_axes(dims))
        return out
    
    @dims_to_spatial_axes
    @same_dtype()
    def _running_kernel(self, radius:float, dims:int, function=None, update:bool=False) -> ImgArray:
        disk = ball_like(radius, len(dims))
        out = self.as_uint16().parallel(function, complement_axes(dims), disk)
        return out
    
    @record()
    def erosion(self, radius:float=1, dims=None, update:bool=False) -> ImgArray:
        f = _binary_erosion if self.dtype == bool else _erosion
        return self._running_kernel(radius, dims, f, update)
    
    @record()
    def dilation(self, radius:float=1, dims=None, update:bool=False) -> ImgArray:
        f = _binary_dilation if self.dtype == bool else _dilation
        return self._running_kernel(radius, dims, f, update)
    
    @record()
    def opening(self, radius:float=1, dims=None, update:bool=False) -> ImgArray:
        f = _binary_opening if self.dtype == bool else _opening
        return self._running_kernel(radius, dims, f, update)
    
    @record()
    def closing(self, radius:float=1, dims=None, update:bool=False) -> ImgArray:
        f = _binary_closing if self.dtype == bool else _closing
        return self._running_kernel(radius, dims, f, update)
    
    @record()
    def tophat(self, radius:float=50, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, dims, _tophat, update)
    
    @record()
    def mean_filter(self, radius:float=1, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, dims, _mean, update)
    
    @record()
    def median_filter(self, radius:float=1, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, dims, _median, update)
    
    @record()
    def entropy_filter(self, radius:float=1, dims=None) -> ImgArray:
        disk = ball_like(radius, len(dims))
        return self.as_uint16().parallel(_entropy, dims, disk)
    
    @record()
    def enhance_contrast(self, radius:float=1, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, dims, _enhance_contrast, update)
    
    @same_dtype()
    @record()
    def fill_hole(self, thr="otsu", update:bool=False) -> ImgArray:
        """
        Filling holes
        Reference
        ---------
        https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html

        Parameters
        ----------
        thr : scalar or str, optional
            Threshold (value or method) to apply if image is not binary, by default "otsu"
        update : bool, optional
            If update self to filtered image, by default False

        Returns
        -------
        ImgArray
            Hole-filled image.
        """        
        if self.dtype != bool:
            mask = self.threshold(thr=thr).value
        else:
            mask = self.value
            
        seed = np.copy(self.value)
        seed[1:-1, 1:-1] = self.max()
        out = skmorph.reconstruction(seed, mask, method="erosion").view(self.__class__)
        return out
    
    
    @dims_to_spatial_axes
    @same_dtype(True)
    @record()
    def gaussian_filter(self, sigma:float=1, dims=None, update:bool=False) -> ImgArray:
        """
        Run Gaussian filter (Gaussian blur).
        Parameters
        ----------
        sigma : scalar or array of scalars, optional
            Standard deviation(s) of Gaussian.
        dims : int or str, optional
            Dimension of axes.
        update : bool, optional
            If update self to filtered image.
            
        Returns
        -------
        ImgArray
            Filtered image.
        """        
        return self.parallel(_gaussian, dims, sigma)

    @dims_to_spatial_axes
    @record()
    def dog_filter(self, low_sigma:float=1, high_sigma=None, *, dims=None) -> ImgArray:
        """
        Run Difference of Gaussian filter. This function does not support `update`
        argument because intensity can be negative.
        
        Parameters
        ----------
        low_sigma : scalar or array of scalars, optional
            lower standard deviation(s) of Gaussian, by default 1
        high_sigma : scalar or array of scalars, optional
            higher standard deviation(s) of Gaussian, by default 1
        dims : int or str, optional
            Dimension of axes.
            
        Returns
        -------
        ImgArray
            Filtered image.
        """        
        if high_sigma is None:
            high_sigma = low_sigma * 1.6
        out = self.parallel(_difference_of_gaussian, complement_axes(dims),
                            low_sigma, high_sigma)
        
        return out
    
    @dims_to_spatial_axes
    @same_dtype()
    @record()
    def rolling_ball(self, radius:float=50, smoothing:bool=True, *, dims=None, update:bool=False) -> ImgArray:
        """
        Subtract Background using rolling-ball algorithm.

        Parameters
        ----------
        radius : int, optional
            Radius of rolling ball, by default 50
        smoothing : bool, optional
            If apply 3x3 averaging before creating background.
        dims : int or str, optional
            Dimension of axes.
            
        Returns
        -------
        ImgArray
            Background subtracted image.
        """        
        out = self.as_uint16().parallel(_rolling_ball, complement_axes(dims), 
                                        radius, smoothing)
        return out
    
    @dims_to_spatial_axes
    def peak_local_max(self, *, min_distance:int=1, thr:float=None, 
                       num_peaks:int=np.inf, num_peaks_per_label:int=np.inf, 
                       use_labels:bool=True, dims=None):
        """
        Find local maxima. This algorithm corresponds to ImageJ's 'Find Maxima' but
        is more flexible.

        Parameters
        ----------
        min_distance : int, optional
            Minimum distance allowed for each two peaks, by default 1
        thr : float, optional
            The absolute minimum intensity of peaks, by default None
        num_peaks : int, optional
            Maximum number of peaks **for each iteration**.
        num_peaks_per_label : int, optional
            Maximum number of peaks per label, by default np.inf
        use_labels : bool, optional
            If use self.labels when it exists, by default True
        dims : int or str, optional
            Dimension of axes.
        """        
        
        # separate spatial dimensions and others
        ndim = len(dims)
        c_axes = complement_axes(dims, all_axes=self.axes)
        shape = self.sizesof(c_axes)
        
        # if c_axes:
        out = PropArray(np.zeros(shape), name=self.name, axes=c_axes, dirpath=self.dirpath,
                        propname="local_max_indices")
        
        self.ongoing = "peak_local_max"
        for sl, img in self.iter(c_axes, israw=True):
            if use_labels and hasattr(img, "labels"):
                labels = img.labels
            else:
                labels = None
            
            # skfeat.peak_local_max overwrite something so we need to give copy of img.
            indices = skfeat.peak_local_max(np.array(img),
                                            min_distance=min_distance, 
                                            threshold_abs=thr,
                                            num_peaks=num_peaks,
                                            num_peaks_per_label=num_peaks_per_label,
                                            labels=np.array(labels))
            
            out[sl[:-ndim]] = SpatialList(ImgArray(indices[:, i], name=self.name, axes=a, 
                                                    dirpath=self.dirpath, history=self.history+["Local-Max"])
                                            for i, a in enumerate(dims))
        self.ongoing = None
        del self.ongoing
        
        return out
    
        
    
    @record()
    def fft(self) -> ImgArray:
        """
        Fast Fourier transformation.
        This function returns complex array. Inconpatible with some functions here.
        """
        freq = fft(self.value.astype("float32"))
        out = np.fft.fftshift(freq)
        return out
    
    @record()
    def ifft(self) -> ImgArray:
        freq = np.fft.fftshift(self.value)
        out = np.real(ifft(freq))
        return out
    
    @dims_to_spatial_axes
    @record()
    def threshold(self, thr="otsu", *, dims=None, **kwargs) -> ImgArray:
        """
        Parameters
        ----------
        thr: int or array or None, optional
            Threshold value, or thresholding algorithm.
        dims : int or str, optional
            Dimension of axes.
        **kwargs:
            Keyword arguments that will passed to function indicated in 'method'.

        """

        methods_ = {"isodata": skfil.threshold_isodata,
                    "li": skfil.threshold_li,
                    "local": skfil.threshold_local,
                    "mean": skfil.threshold_mean,
                    "min": skfil.threshold_minimum,
                    "minimum": skfil.threshold_minimum,
                    "multiotsu": skfil.threshold_multiotsu,
                    "niblack": skfil.threshold_niblack,
                    "otsu": skfil.threshold_otsu,
                    "sauvola": skfil.threshold_sauvola,
                    "triangle": skfil.threshold_triangle,
                    "yen": skfil.threshold_yen
                    }
        
        if isinstance(thr, str):
            method = thr.lower()
            try:
                func = methods_[method]
            except KeyError:
                s = ", ".join(list(methods_.keys()))
                raise KeyError(f"{method}\nmethod must be: {s}")
            
            out = np.zeros(self.shape, dtype=bool)
            for t, img in self.iter(complement_axes(dims), False):
                thr = func(img, **kwargs)
                out[t] = img >= thr
            out = out.view(self.__class__)

        else:
            # TODO: only PropArray
            if hasattr(thr, "__iter__"):
                # out = np.zeros(self.shape, dtype=bool)
                # for i, (t, img) in enumerate(self.iter(complement_axes(dims), False)):
                #     out[t] = img >= thr[i]
                # out = out.view(self.__class__)
                raise NotImplementedError
            else:
                out = self >= thr
        out = out.view(self.__class__)
        out.temp = thr
        return out
        
    
    def specify(self, xy:tuple[int], dxdy:tuple[int], position="corner") -> ImgArray:
        """
        Make a rectancge label.
        """
        x, y = xy
        dx, dy = dxdy
        
        if position == "corner":
            pass
        elif position == "center":
            x -= dx//2
            y -= dy//2
        else:
            raise ValueError("'position' must be either 'corner' or 'center'")
        
        if hasattr(self, "labels"):
            self.labels[y:y+dy, x:x+dx] = self.labels.max() + 1
        else:
            labels = np.zeros(self.sizesof("yx"), dtype="uint8")
            labels[y:y+dy, x:x+dx] = 1
            self.labels = labels
        
        return self

    @record()
    def crop_center(self, scale:float=0.5) -> ImgArray:
        """
        Crop out the center of an image.
        e.g. when scale=0.5, create 512x512 image from 1024x1024 image.
        """
        if scale <= 0 or 1 < scale:
            raise ValueError(f"scale must be (0, 1], but got {scale}")
        
        sizex = self.sizeof("x")
        sizey = self.sizeof("y")
        
        x0 = int(sizex / 2 * (1 - scale)) + 1
        x1 = int(sizex / 2 * (1 + scale))
        y0 = int(sizey / 2 * (1 - scale)) + 1
        y1 = int(sizey / 2 * (1 + scale))

        out = self[f"x={x0}-{x1};y={y0}-{y1}"]
        
        return out
    
    @dims_to_spatial_axes
    @record()
    def distance_map(self, *, dims=None) -> ImgArray:
        """
        Calculate distance map from binary images.

        Parameters
        ----------
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Distance map, the further the brighter

        """        
        if self.dtype != bool:
            raise TypeError("Cannot run distance_map() with non-binary image.")
        out = self.parallel(_distance_transform_edt, complement_axes(dims))
        
        return out
        
    @dims_to_spatial_axes
    @record()
    def skeletonize(self, dims=None):
        if self.dtype != bool:
            raise TypeError("Cannot run skeletonize() with non-binary image.")
        
        out = self.parallel(_skeletonize, complement_axes(dims), outdtype=bool)
        # TODO: medial_axis, skeletonize, etc...
        # https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
        return out
    
    @record(append_history=False)
    def curve_fit(self, f, p0, axes=None) -> PropArray:
        # TODO: any general method? Move to PropArray?
        params, cov = opt.curve_fit(f, np.arange(self.size), self, p0 = p0)
        pass
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def profile_line(self, src, dst, linewidth=None, *, order=None, dims=None) -> PropArray:
        c_axes = complement_axes(dims, all_axes=self.axes)
        out = PropArray(np.empty(self.sizesof(c_axes)), name=self.name, axes=c_axes, 
                        dirpath=self.dirpath, propname="line_profile")
        for sl, img in self.iter(c_axes):
            out[sl] = skmes.profile_line(img, src, dst, linewidth=linewidth, order=order)
            
        return out
    
    @dims_to_spatial_axes
    @record(False)
    def label(self, label_image=None, *, dims=None, connectivity=None) -> ImgArray:
        """
        Run skimage's label() and store the results as attribute.

        Parameters
        ----------
        label_image : array, optional
            Image to make label, by default self is used.
        connectivity : int, optional
            Passed to skimage's label(), by default None

        Returns
        -------
        labeled image
        """        
        # check the shape of label_image
        if label_image is None:
            label_image = self
            
        elif not hasattr(label_image, "axes") or label_image.axes.is_none():
            raise ValueError("Use Array with axes for label_image.")
        
        elif not axes_included(self, label_image):
            raise ImageAxesError("Not all the axes in 'label_image' are included in self: "
                                 f"{label_image.axes} and {self.axes}")
        
        elif not shape_match(self, label_image):
            raise ImageAxesError("Shape mismatch.")
        
        c_axes = complement_axes(dims)
        label_image.ongoing = "label"
        labels = label_image.parallel(_label, c_axes, connectivity, outdtype="uint32").view(np.ndarray)
        label_image.ongoing = None
        del label_image.ongoing
        
        min_nlabel = 0
        for sl, _ in label_image.iter(c_axes, False):
            labels[sl][labels[sl]>0] += min_nlabel
            min_nlabel += labels[sl].max()
        
        self.labels = labels.view(Label).optimize()
        self.labels._set_info(label_image, "Labeled")
        return self
    
    @dims_to_spatial_axes
    @need_labels
    @record(record_label=True)
    def expand_labels(self, distance:int=1, *, dims=None) -> ImgArray:
        """
        Expand areas of labels.

        Parameters
        ----------
        distance : int, optional
            The distance to expand, by default 1

        Returns
        -------
        ImgArray
            Same array but labels are updated.
        """        
        ndim = len(dims)
        labels = np.empty_like(self.labels).value
        for sl, img in self.iter(complement_axes(dims), israw=True):
            labels[sl[:-ndim]] = skseg.expand_labels(img.labels.value, distance)
        
        self.labels = labels.view(Label)
        
        return self
    
    @dims_to_spatial_axes
    @need_labels
    @record(record_label=True)
    def watershed(self, connectivity=1, input_="self", *, dims=None) -> ImgArray:
        """
        Label segmentation using watershed algorithm.

        Parameters
        ----------
        connectivity : int, optional
            Passed to skimage.segmentation.watershed.
        input_ : str, optional
            What image will be the input of watershed algorithm.
            - "labels" ... self.labels is used.
            - "self" ... self is used.
            - "distance" ... distance map of self.labels is used.
        dims : int or str, optional
            Spatial dimension.
            
        Returns
        -------
        ImgArray
            Same array but labels are updated.
        """
        
        ndim = len(dims)
        markers = self.peak_local_max(dims=dims)
        
        # Prepare the input image.
        if input_ == "labels":
            input_img = self.labels.copy()
        elif input_ == "self":
            input_img = self.copy()
        elif input_ == "distance":
            distance_img = -ndi.distance_transform_edt(self.labels.value)
            input_img = array(distance_img, dtype="float32")
        else:
            raise ValueError("'input_' must be either 'self', 'labels' or 'distance'.")
        
        input_img._view_labels(self)
        
        labels = np.zeros(input_img.shape, dtype="uint32")
        input_img.ongoing = "watershed"
        shape = self.sizesof(dims)
        n_labels = 0
        
        for sl, img in input_img.iter(complement_axes(dims), israw=True):
            # Make array from max list
            marker_input = np.zeros(shape, dtype="uint32")
            
            sl0 = markers[sl[:-ndim]]
            
            marker_input[tuple(sl0)] = np.arange(1, len(sl0[0])+1, dtype="uint32")
            labels[sl] = skseg.watershed(img.value, marker_input, mask=img.labels.value, 
                                         connectivity=connectivity)
            labels[sl][labels[sl]>0] += n_labels
            n_labels = labels[sl].max()
            
        input_img.ongoing = None
        del input_img.ongoing
        
        labels = labels.view(Label)
        self.labels = labels.optimize()
        return self
    
    @dims_to_spatial_axes
    def label_threshold(self, thr="otsu", *, dims=None, **kwargs) -> ImgArray:
        """
        Make labels with threshold().

        Parameters
        ----------
        All are passed to self.threshold()
        
        Returns
        -------
        ImgArray
            Same array but labels are updated.
        """        
        labels = self.threshold(thr=thr, dims=dims, **kwargs)
        return self.label(labels)
    
        
    @need_labels
    def regionprops(self, properties:tuple[str]=("mean_intensity", "area"), *, 
                    extra_properties=None) -> ArrayDict:
        """
        Run skimage's regionprops() function and return the results as PropArray, so
        that you can access using flexible slicing. For example, if a tcyx-image is
        analyzed with properties=("X", "Y"), then you can get X's time-course profile
        of channel 1 at label 3 by prop["X"]["p=5;c=1"].

        Parameters
        ----------
        properties : iterable, optional
            properties to analyze, see skimage.measure.regionprops.
        extra_properties : iterable of callable, optional
            extra properties to analyze, see skimage.measure.regionprops.

        Returns
        -------
            ArrayDict of PropArray
        """        
        
        if isinstance(properties, str):
            properties = (properties,)

        if "p" in self.axes:
            # this dimension will be label
            raise ValueError("axis 'p' is forbidden.")
        
        prop_axes = complement_axes(self.labels.axes, all_axes="tzcyx")
        shape = self.sizesof(prop_axes)
        
        out = ArrayDict({p: PropArray(np.zeros((self.labels.max(),) + shape, dtype="float32"),
                                      name=self.name, 
                                      axes="p" + prop_axes,
                                      dirpath=self.dirpath,
                                      propname = p)
                         for p in properties})
        
        # calculate property value for each slice
        for sl in itertools.product(*map(range, shape)):
            props = skmes.regionprops(self.labels, self.value[sl], 
                                      cache=False,
                                      extra_properties=extra_properties)
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]
        
        return out
    
    @same_dtype()
    def proj(self, axis=None, method="mean") -> ImgArray:
        """
        Z-projection.
        'method' must be in func_dict.keys() or some function like np.mean.
        This function is not compatible with record().
        """
        func_dict = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max, "median": np.median}
        if method in func_dict.keys():
            func = func_dict[method]
        elif callable(method):
            func = method
        else:
            raise TypeError(f"'method' must be one of {', '.join(list(func_dict.keys()))} or callable object.")
        
        if axis is None:
            axis = find_first_appeared(self.axes, "tzcp")
        axisint = self.axisof(axis)
        out = func(self.value, axis=axisint).view(self.__class__)
        out._set_info(self, f"proj(axis={axis}, method={method})", del_axis(self.axes, axisint))
        return out

    @record()
    def clip_outliers(self, in_range=("0%", "100%")) -> ImgArray:
        """
        Saturate low/high intensity using np.clip.mean

        Parameters
        ----------
        in_range : two scalar values, optional
            range of lower/upper limits, by default (0, 100)

        Returns
        -------
        ImgArray
            Clipped image with temporal attribute
        """        
        lowerlim, upperlim = check_clip_range(in_range, self.value)
        out = np.clip(self.value, lowerlim, upperlim)
        out = out.view(self.__class__)
        out.temp = [lowerlim, upperlim]
        return out
    
    @record()
    def rescale_intensity(self, in_range=("0%", "100%"), dtype=np.uint16) -> ImgArray:
        """
        Rescale the intensity of the image using skimage.exposure.rescale_intensity.

        Parameters
        ----------
        in_range : two scalar values, optional
            range of lower/upper limit, by default (0, 100)
        dtype : numpy dtype, optional
            output dtype, by default np.uint16

        Returns
        -------
        ImgArray
            Rescaled image with temporal attribute
        """        
        out = self.view(np.ndarray).astype("float32")
        lowerlim, upperlim = check_clip_range(in_range, self.value)
            
        out = skexp.rescale_intensity(out, in_range=(lowerlim, upperlim), out_range=dtype)
        
        out = out.view(self.__class__)
        out.temp = [lowerlim, upperlim]
        return out

        

# non-member functions.

def array(arr, dtype="uint16", *, name=None, axes=None, lut=None) -> ImgArray:
    """
    make an ImgArray object, just like np.array(x)
    """
    if isinstance(arr, str):
        raise TypeError(f"String is invalid input. Do you mean imread(path)?")
    
    arr = np.array(arr, dtype=dtype)
        
    # Automatically determine axes
    if axes is None:
        axes = ["x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][arr.ndim-1]
            
    self = ImgArray(arr, name=name, axes=axes, lut=lut)
    
    return self

def zeros(shape, dtype="uint16", *, name=None, axes=None, lut=None) -> ImgArray:
    return array(np.zeros(shape, dtype=dtype), dtype=dtype, name=name, axes=axes, lut=lut)

def empty(shape, dtype="uint16", *, name=None, axes=None, lut=None) -> ImgArray:
    return array(np.empty(shape, dtype=dtype), dtype=dtype, name=name, axes=axes, lut=lut)

def imread(path:str, dtype:str="uint16", *, axes=None, lut=None) -> ImgArray:
    """
    Load image from path.

    Parameters
    ----------
    path : str
        Path to the image.
    dtype : str, optional
        dtype of the image, by default "uint16"
    axes : str or None, optional
        If the image does not have axes metadata, this value will be used.
    lut : list of str, or None, optional
        LUT of the image.

    Returns
    -------
    ImgArray
    """    
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: {path}")
    
    fname, fext = os.path.splitext(os.path.basename(path))
    img = io.imread(path)
    dirpath = os.path.dirname(path)
    
    # read tif metadata
    if fext == ".tif":
        meta = get_meta(path)
    elif fext in (".png", ".jpg") and img.ndim == 3 and img.shape[-1] <= 4:
        meta = {"axes":"yxc", "ijmeta":{}, "history":[]}
    else:
        meta = {"axes":axes, "ijmeta":{}, "history":[]}
    
    
    
    axes = meta["axes"]
    metadata = meta["ijmeta"]
    lut = None                  # TODO: read LUT from ImageJ metadata
    if meta["history"]:
        name = meta["history"].pop(0)
        history = meta["history"]
    else:
        name = fname
        history = []
        
    
    self = ImgArray(img, name=name, axes=axes, dirpath=dirpath, 
                    history=history, metadata=metadata, lut=lut)
        
    # In case the image is in yxc-order. This sometimes happens.
    if "c" in self.axes and self.sizeof("c") > self.sizeof("x"):
        self = np.moveaxis(self, -1, -3)
        _axes = self.axes.axes
        _axes = _axes[:-3] + "cyx"
        self.axes = _axes
    
    if self.axes.is_none():
        return self
    else:
        return self.sort_axes().as_img_type(dtype) # arrange in ptzcyx-order

def imread_collection(dirname:str, axis:str="p", *, ext:str="tif", 
                      ignore_exception:bool=False, dtype="uint16") -> ImgArray:
    """
    Read images recursively from a directory, and stack them into one ImgArray.

    Parameters
    ----------
    dirname : str
        Path to the directory
    axis : str, optional
        To specify which axis will be the new one, by default "p"
    ext : str, optional
        Extension of files, by default "tif"
    ignore_exception : bool, optional
        If true, arrays with wrong shape will be ignored, by default False
    """    
    paths = glob.glob(f"{dirname}{os.sep}**{os.sep}*.{ext}", recursive=True)
    imgs = []
    shapes = []
    for path in paths:
        img = imread(path, dtype=dtype)
        imgs.append(img)
        shapes.append(img.shape)
    
    list_of_shape = list(set(shapes))
    if len(list_of_shape) > 1:
        if ignore_exception:
            ctr = collections.Counter(shapes)
            common_shape = ctr.most_common()[0][0]
            imgs = [img for img in imgs if img.shape == common_shape]
        else:
            raise ValueError("Input directory has images with different shapes: "
                            f"{', '.join(map(str, list_of_shape))}")
    
    out = stack(imgs, axis=axis)
    out.dirpath, out.name = os.path.split(dirname)
    out.history[-1] = "imread_collection"
    return out
    

def read_meta(path:str) -> dict:
    meta = get_meta(path)
    return meta

def set_cpu(n_cpu:int) -> None:
    ImgArray.n_cpu=n_cpu
    return None

def stack(imgs, axis="c", dtype="uint16"):
    """
    Create stack image from list of images.

    Parameters
    ----------
    imgs : iterable object of images.
        Images to stack. These images must have exactly the same shapes.
    axis : str, optional
        Which axis will be the new one, by default "c"
    dtype : str, optional
        Output dtype, by default "uint16"

    Returns
    -------
    ImgArray
        Image stack

    """    
    
    if isinstance(imgs, np.ndarray):
        raise TypeError("cannot stack single array.")
    
    # find where to add new axis
    if imgs[0].axes:
        new_axes = Axes(axis + str(imgs[0].axes))
        new_axes.sort()
        _axis = new_axes.find(axis)
    else:
        new_axes = None
        _axis = 0

    arrs = [img.as_img_type(dtype).value for img in imgs]

    out = np.stack(arrs, axis=0)
    out = np.moveaxis(out, 0, _axis)
    out = array(out)    
    out._set_info(imgs[0], f"Make-Stack(axis={axis})", new_axes)
    
    # connect LUT if needed.
    if axis == "c":
        luts = [img.lut[0] for img in imgs]
        out.lut = luts
    
    return out


