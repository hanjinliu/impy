from __future__ import annotations
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import collections
from numpy.core.numeric import True_
from skimage import io
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import exposure as skexp
from skimage import measure as skmes
from skimage import segmentation as skseg
from skimage import feature as skfeat
from skimage import registration as skreg
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft
from .func import *
from .deco import *
from .gauss import GaussianBackground, GaussianParticle
from .labeledarray import LabeledArray
from .label import Label
from .axes import Axes, ImageAxesError
from .specials import PropArray, MarkerArray, IndexArray
from .utilcls import *
from ._process import *


class ImgArray(LabeledArray):
    
    def freeze(self):
        """
        To avoid image analysis.
        """        
        return self.view(LabeledArray)
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def affine(self, *, dims=None, order:int=1, **kwargs) -> ImgArray:
        """
        Affine transformation
        kwargs: matrix, scale, rotation, shear, translation
        """
        if dims != 2:
            raise ValueError("dims != 2 version have yet been implemented")
        mx = sktrans.AffineTransform(**kwargs)
        out = self.parallel(affine_, complement_axes(dims, self.axes), mx, order)
        return out
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def translate(self, translation=None, *, dims=2) -> ImgArray:
        """
        Simple translation of image, i.e. (x, y) -> (x+dx, y+dy)
        """
        mx = sktrans.AffineTransform(translation=translation)
        out = self.parallel(affine_, complement_axes(dims, self.axes), mx)
        return out

    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def rescale(self, scale:float=1/16, *, dims=None, order:int=None) -> ImgArray:
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
        out = out.view(self.__class__)
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
    
    # @dims_to_spatial_axes
    # @record(append_history=False)
    # def gaussfit_particle(self, markers=None, radius=4,
    #                       p0=None, *, dims=None) -> PropArray:
        
    #     ndim = len(dims)
    #     if markers is None:
    #         markers = self.peak_local_max(dims=dims, min_distance=int(radius*1.4143), squeeze=False)
        
    #     self.specify(markers, radius, labeltype="square")
        
    #     fitting_params = PropArray(np.empty(markers.shape), name=self.name, 
    #                        dirpath=self.dirpath, propname="gaussfit_particle_fitting_params")
        
    #     fitting_result = PropArray(np.empty(markers.shape), name=self.name, 
    #                                dirpath=self.dirpath, propname="gaussfit_particle_fitting_result")
        
    #     self.ongoing = "gaussfit_particle"
    #     for sl, data in self.iter(complement_axes(dims)):
    #         sl0 = sl[:-ndim]
    #         centers = markers[sl0]
            
    #         fitting_params_ = PropArray(np.empty(centers.shape[1]), propname="fitting_params")
    #         fitting_result_ = PropArray(np.empty(centers.shape[1]), propname="fitting_result")
            
    #         gaussian = GaussianParticle(p0)
    #         r0s = centers - radius // 2
    #         r1s = centers + (radius+1) // 2
            
    #         for i, ((_, r0), (_, r1)) in enumerate(zip(r0s.iter("p"), r1s.iter("p"))):
    #             # r0 = (y0, x0)
    #             s = tuple(slice(x0, x1) for x0, x1 in zip(r0, r1))
    #             if data[s].shape != (radius, radius):
    #                 fitting_result_[i] = None
    #                 fitting_params_[i] = None
    #             else:
    #                 fitting_result_[i] = gaussian.fit(data[s])
    #                 gaussian.shift([r0[1], r0[0]])
    #                 fitting_params_[i] = gaussian.params
            
    #         fitting_result[sl0] = fitting_result_
    #         fitting_params[sl0] = fitting_params_

    #     result = ArrayDict(fitting_result=fitting_result, parameters=fitting_params)
    #     self.ongoing = None
    #     del self.ongoing
        
    #     return result
    
    
    @dims_to_spatial_axes
    def find_sm(self, sigma:float=1.5, *, percentile:float=99, num_peaks:int=np.inf, 
                squeeze:bool=True, dims=None):
        """
        Single molecule detection using difference of Gaussian method.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of puncta.
        percentile, num_peaks, squeeze, dims
            Passed to peak_local_max()

        
        Returns
        -------
        PropArray of IndexArrays, or if squeeze=True, IndexArray
            PropArray with dtype=object is returned, with IndexArrays in it. Every IndexArray has
            rp-axes, where r=0 means y-coordinate for 2D-image, and `p` is the index of points.
        """        
        
        dog_img = self.dog_filter(low_sigma=sigma, dims=dims)
        markers = dog_img.peak_local_max(min_distance=1, percentile=percentile, 
                                         num_peaks=num_peaks, squeeze=squeeze, dims=dims)
        return markers
    
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
        
        def check_matrix(ref):
            mtx = []
            for m in ref:
                if np.isscalar(m): 
                    if m == 1:
                        mtx.append(m)
                    else:
                        raise ValueError(f"Only `1` is ok, but got {m}")
                    
                elif m.shape != (3, 3) or not np.allclose(m[2,:2], 0):
                    raise ValueError(f"Wrong Affine transformation matrix:\n{m}")
                
                else:
                    mtx.append(m)
            return mtx
        
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
            mtx = check_matrix(ref)
        
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
            if np.isscalar(m) and m==1:
                corrected.append(self[f"{axis}={i}"])
            else:
                corrected.append(self[f"{axis}={i}"].affine(order=order, matrix=m))

        out = stack(corrected, axis=axis, dtype=self.dtype)
        out.temp = mtx
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def hessian_eigval(self, sigma=1, *, dims=None) -> ImgArray:
        """
        Calculate Hessian's eigenvalues for each image. If dims=2, every yx-image 
        is considered to be a single spatial image, and if dims=3, zyx-image.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        ImgArray
            Array of eigenvalues. The axis `l` denotes the index of eigenvalues.
            l=0 means the smallest eigenvalue.
        """        
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        
        eigval = self.as_float().parallel(hessian_eigval_, 
                                          complement_axes(dims, self.axes), 
                                          sigma, pxsize,
                                          outshape=self.shape+(ndim,))
        
        eigval.axes = str(self.axes) + "l"
        eigval = eigval.sort_axes()
        eigval._set_info(self, f"hessian_eigval", new_axes=eigval.axes)
        
        return eigval
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def hessian_eig(self, sigma=1, *, dims=None) -> tuple[ImgArray, ImgArray]:
        """
        Calculate Hessian's eigenvalues and eigenvectors.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        ImgArray and ImgArray
            Arrays of eigenvalues and eigenvectors. The axis `l` denotes the index of 
            eigenvalues. l=0 means the smallest eigenvalue. `r` denotes the index of
            spatial dimensions. For 3D image, r=0 means z-element of an eigenvector.
        """                
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        eigval, eigvec = self.parallel_eig(hessian_eigh_, 
                                           complement_axes(dims, self.axes), 
                                           sigma, pxsize)
        
        eigval.axes = str(self.axes) + "l"
        eigval = eigval.sort_axes()
        eigval._set_info(self, f"hessian_eigval", new_axes=eigval.axes)
        
        eigvec.axes = str(self.axes) + "rl"
        eigvec = eigvec.sort_axes()
        eigvec._set_info(self, f"hessian_eigvec", new_axes=eigvec.axes)
        
        return eigval, eigvec
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def structure_tensor_eigval(self, sigma=1, *, dims=None) -> ImgArray:
        """
        Calculate structure tensor's eigenvalues and eigenvectors.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        ImgArray
            Array of eigenvalues. The axis `l` denotes the index of eigenvalues.
            l=0 means the smallest eigenvalue.
        """          
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        eigval = self.as_float().parallel(structure_tensor_eigval_, 
                                          complement_axes(dims, self.axes), 
                                          sigma, pxsize,
                                          outshape=self.shape+(ndim,))
        
        eigval.axes = str(self.axes) + "l"
        eigval = eigval.sort_axes()
        eigval._set_info(self, f"structure_tensor_eigval", new_axes=eigval.axes)
        return eigval
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def structure_tensor_eig(self, sigma=1, *, dims=None)-> tuple[ImgArray, ImgArray]:
        """
        Calculate structure tensor's eigenvalues and eigenvectors.

        Parameters
        ----------
        sigma : scalar or array (dims,), optional
            Standard deviation of Gaussian filter applied before calculating Hessian.
        dims : int or str, optional
            Spatial dimension.

        Returns
        -------
        ImgArray and ImgArray
            Arrays of eigenvalues and eigenvectors. The axis `l` denotes the index of 
            eigenvalues. l=0 means the smallest eigenvalue. `r` denotes the index of
            spatial dimensions. For 3D image, r=0 means z-element of an eigenvector.
        """                
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        eigval, eigvec = self.parallel_eig(structure_tensor_eigh_, 
                                           complement_axes(dims, self.axes), 
                                           sigma, pxsize)
        
        eigval.axes = str(self.axes) + "l"
        eigval = eigval.sort_axes()
        eigval._set_info(self, f"structure_tensor_eigval", new_axes=eigval.axes)
        
        eigvec.axes = str(self.axes) + "rl"
        eigvec = eigvec.sort_axes()
        eigvec._set_info(self, f"structure_tensor_eigvec", new_axes=eigvec.axes)
        
        return eigval, eigvec
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def sobel_filter(self, *, dims=None, update:bool=False):
        return self.parallel(sobel_, complement_axes(dims, self.axes))
    
    @dims_to_spatial_axes
    @same_dtype()
    @record()
    def convolve(self, kernel, *, dims=None, update:bool=False):
        return self.parallel(convolve_, complement_axes(dims, self.axes), kernel)
    
    @dims_to_spatial_axes
    @same_dtype()
    def _running_kernel(self, radius:float, function=None, *, dims=None, update:bool=False) -> ImgArray:
        disk = ball_like(radius, len(dims))
        return self.parallel(function, complement_axes(dims, self.axes), disk)
    
    @record()
    def erosion(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        f = binary_erosion_ if self.dtype == bool else erosion_
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @record()
    def dilation(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        f = binary_dilation_ if self.dtype == bool else dilation_
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @record()
    def opening(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        f = binary_opening_ if self.dtype == bool else opening_
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @record()
    def closing(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        f = binary_closing_ if self.dtype == bool else closing_
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @record()
    def tophat(self, radius:float=50, *, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, tophat_, dims=dims, update=update)
    
    @record()
    def mean_filter(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, mean_, dims=dims, update=update)
    
    @record()
    def median_filter(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, median_, dims=dims, update=update)
    
    @record()
    def entropy_filter(self, radius:float=1, *, dims=None) -> ImgArray:
        disk = ball_like(radius, len(dims))
        return self.as_uint16().parallel(entropy_, dims, disk)
    
    @record()
    def enhance_contrast(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        return self._running_kernel(radius, enhance_contrast_, dims=dims, update=update)
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def fill_hole(self, thr="otsu", *, dims=None, update:bool=False) -> ImgArray:
        """
        Filling holes
        
        Reference
        ---------
        https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html

        Parameters
        ----------
        thr : scalar or str, optional
            Threshold (value or method) to apply if image is not binary.
        dims : int or str, optional
            Dimension of axes.
        update : bool, by default False
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Hole-filled image.
        """        
        if self.dtype != bool:
            mask = self.threshold(thr=thr).value
        else:
            mask = self.value
        
        return self.parallel(fill_hole_, complement_axes(dims, self.axes), mask, outdtype=self.dtype)
    
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def gaussian_filter(self, sigma:float=1, *, dims=None, update:bool=False) -> ImgArray:
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
        return self.parallel(gaussian_, complement_axes(dims, self.axes), sigma)


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
        
        return self.parallel(difference_of_gaussian_, complement_axes(dims, self.axes),
                             low_sigma, high_sigma)
        
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
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
        update : bool, optional
            If update self to filtered image.
            
        Returns
        -------
        ImgArray
            Background subtracted image.
        """        
        return self.parallel(rolling_ball_, complement_axes(dims, self.axes), 
                             radius, smoothing)
        
    
    @dims_to_spatial_axes
    def peak_local_max(self, *, min_distance:int=1, percentile:float=None, 
                       num_peaks:int=np.inf, num_peaks_per_label:int=np.inf, 
                       use_labels:bool=True, squeeze:bool=True, dims=None):
        """
        Find local maxima. This algorithm corresponds to ImageJ's 'Find Maxima' but
        is more flexible.

        Parameters
        ----------
        min_distance : int, optional
            Minimum distance allowed for each two peaks, by default 1
        percentile : float, optional
            Percentile to compute absolute threshold.
        num_peaks : int, optional
            Maximum number of peaks **for each iteration**.
        num_peaks_per_label : int, default is np.inf
            Maximum number of peaks per label.
        use_labels : bool, default is True
            If use self.labels when it exists.
        squeeze : bool, default is True
            If True and 0-dimensional array will be returned, the contents will be
            returned instead. This situation only happens when self.ndim == len(dims).
        dims : int or str, optional
            Dimension of axes.
            
        Returns
        -------
        PropArray of IndexArrays, or if squeeze=True, IndexArray
            PropArray with dtype=object is returned, with IndexArrays in it. Every IndexArray has
            rp-axes, where r=0 means y-coordinate for 2D-image, and `p` is the index of points.
        """        
        
        # separate spatial dimensions and others
        ndim = len(dims)
        c_axes = complement_axes(dims, self.axes)
        shape = self.sizesof(c_axes)
        
        if percentile is None:
            thr = None
        else:
            thr = np.percentile(self.value, percentile)
        
        # if c_axes:
        out = PropArray(np.zeros(shape), name=self.name, axes=c_axes,
                        dirpath=self.dirpath, propname="local_max_indices")
        
        self.ongoing = "peak_local_max"
        for sl, img in self.iter(c_axes, israw=True, exclude=dims):
            # skfeat.peak_local_max overwrite something so we need to give copy of img.
            if use_labels and hasattr(img, "labels"):
                labels = np.array(img.labels)
            else:
                labels = None
            
            indices = skfeat.peak_local_max(np.array(img),
                                            min_distance=min_distance, 
                                            threshold_abs=thr,
                                            num_peaks=num_peaks,
                                            num_peaks_per_label=num_peaks_per_label,
                                            labels=labels)
            
            indarr = IndexArray(indices.T, name=self.name, axes="rp", 
                                dirpath=self.dirpath)
            out[sl] = indarr
        
        self.ongoing = None
        del self.ongoing
        
        if squeeze and out.ndim == 0:
            out = out[()]
        else:
            out.set_scale(self)
            
        return out
    
        
    @dims_to_spatial_axes
    @record()
    def fft(self, *, dims=None) -> ImgArray:
        """
        Fast Fourier transformation.
        This function returns complex array. Inconpatible with some ImgArray functions.
        
        Parameters
        ----------
        
        dims : int or str, optional
            Dimension of axes.
            
        Returns
        -------
        ImgArray
            Complex array.
        """
        freq = fft(self.value.astype("float32"), shape=self.sizesof(dims), 
                   axes=[self.axisof(a) for a in dims])
        out = np.fft.fftshift(freq)
        return out
    
    @record()
    def ifft(self, *, dims=None) -> ImgArray:
        """
        Fast Inverse Fourier transformation. Complementary function with `fft()`.
        
        Parameters
        ----------
        
        dims : int or str, optional
            Dimension of axes.
            
        Returns
        -------
        ImgArray
            Real array.
        """
        
        freq = np.fft.fftshift(self.value)
        out = ifft(freq, shape=self.sizesof(dims), 
                   axes=[self.axisof(a) for a in dims])
        out = np.real(out)
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

        Returns
        -------
        ImgArray
            Boolian array.
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
            for t, img in self.iter(complement_axes(dims, self.axes), False):
                thr = func(img, **kwargs)
                out[t] = img >= thr
            

        elif np.isscalar(thr):
            out = self >= thr
        else:
            raise TypeError("'thr' must be numeric, or str specifying a thresholding method.")                
        
        return out
        
    @dims_to_spatial_axes
    def specify(self, center, radius, *, dims=None, labeltype="square") -> ImgArray:
        """
        Make rectangle or ellipse labels from points.
        
        Parameters
        ----------
        center : array like, MarkerArray or PropArray
            Coordinates of centers. 
        radius : float or array
            Radius of labels.
        dims : int or str, optional
            Dimension of axes.
        labeltype : str, by default "square"
            The shape of labels.

        Returns
        -------
        ImgArray
            Labeled image.
        """
        
        if isinstance(center, PropArray):
            melted = center.melt()
            dims = "".join(a for a in dims if a not in center.axes)
            ndim = len(dims)
            
            if np.isscalar(radius):
                radius = np.full(ndim, radius)
            radius = np.asarray(radius)
            
            shape = self.sizesof(dims)
            label_shape = center.shape + shape
            label_axes = str(center.axes) + dims
            if not hasattr(self, "labels"):
                self.labels = Label(np.zeros(label_shape, dtype="uint8"), dtype="uint8", axes=label_axes)
                self.labels.set_scale(self)

            for _, marker in melted.iter("p"):
                center = tuple(marker[-ndim:])
                label_sl = tuple(marker[:-ndim])
                sl = specify_one(center, radius, shape, labeltype)
                self.labels[label_sl][sl] = self.labels.max() + 1
                if self.labels.max() == np.iinfo(self.labels.dtype).max:
                    self.labels = self.labels.as_larger_type()
        
        else:
            center = np.asarray(center)
            if center.ndim == 1:
                center = center.reshape(-1, 1)
            center = MarkerArray(center, dtype="uint16", axes="rp")
            c = PropArray(np.zeros(()), axes="")
            c[()] = center
            center = c
            return self.specify(center, radius, dims=dims, labeltype=labeltype)     
        
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
        
        x0 = int(sizex / 2 * (1 - scale))
        x1 = int(sizex / 2 * (1 + scale)) + 1
        y0 = int(sizey / 2 * (1 - scale))
        y1 = int(sizey / 2 * (1 + scale)) + 1

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
        return self.parallel(distance_transform_edt_, complement_axes(dims, self.axes))
        
    @dims_to_spatial_axes
    @record()
    def skeletonize(self, *, dims=None) -> ImgArray:
        """
        Skeletonize images. Only works for binary images.

        Parameters
        ----------
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Skeletonized image.
        """        
        if self.dtype != bool:
            raise TypeError("Cannot run skeletonize() with non-binary image.")
        
        return self.parallel(skeletonize_, complement_axes(dims, self.axes), outdtype=bool)
    
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def profile_line(self, src, dst, linewidth=1, *, order=None, dims=None) -> PropArray:
        """
        Measure line profile iteratively for every slice of image.

        Parameters
        ----------
        src : array, shape (2,)
            Source coordinate.
        dst : array, shape (2,)
            Destination coordinate.
        linewidth : int, by default 1.
            Line width.
        order : int, optional
            Spline interpolation order.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Line scans.
        """        
        # determine length
        src = np.asarray(src, dtype=float)
        dst = np.asarray(dst, dtype=float)
        d_row, d_col = dst - src
        length = int(np.ceil(np.hypot(d_row, d_col) + 1))
        
        c_axes = complement_axes(dims, self.axes)
        out = np.empty(self.sizesof(c_axes) + (length,), dtype="float32")
        
        for sl, img in self.iter(c_axes, exclude=dims):
            out[sl] = skmes.profile_line(img, src, dst, linewidth=linewidth, 
                                         order=order, mode="reflect")
        out = out.view(self.__class__)
        out._set_info(self, "profile_line", c_axes+dims[-1])
        out.set_scale(self)
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
        dims : int or str, optional
            Dimension of axes.
        connectivity : int, optional
            Passed to skimage's label(), by default None

        Returns
        -------
            Labeled image
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
        
        c_axes = complement_axes(dims, self.axes)
        label_image.ongoing = "label"
        labels = label_image.parallel(label_, c_axes, connectivity, outdtype="uint32").view(np.ndarray)
        label_image.ongoing = None
        del label_image.ongoing
        
        min_nlabel = 0
        for sl, _ in label_image.iter(c_axes, False):
            labels[sl][labels[sl]>0] += min_nlabel
            min_nlabel += labels[sl].max()
        
        self.labels = labels.view(Label).optimize()
        self.labels._set_info(label_image, "Labeled")
        self.labels.set_scale(self)
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
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Same array but labels are updated.
        """        
        
        labels = np.empty_like(self.labels).value
        for sl, img in self.iter(complement_axes(dims, self.axes), israw=True, exclude=dims):
            labels[sl] = skseg.expand_labels(img.labels.value, distance)
        
        self.labels.value[:] = labels
        
        return self
    
    @dims_to_spatial_axes
    @need_labels
    @record(record_label=True)
    def watershed(self, markers:PropArray=None, connectivity=1, input_="distance", *, dims=None) -> ImgArray:
        """
        Label segmentation using watershed algorithm.

        Parameters
        ----------
        markers : PropArray, optional
            Returned by such as `peak_local_max()`. Array of coordinates of peaks.
        connectivity : int, optional
            Passed to skimage.segmentation.watershed.
        input_ : str, optional
            What image will be the input of watershed algorithm.            
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
        # Prepare the input image.
        if input_ == "self":
            input_img = self.copy()
        elif input_ == "distance":
            input_img = self.__class__(self.labels>0, axes=self.axes).distance_map(dims=dims)
        else:
            raise ValueError("'input_' must be either 'self' or 'distance'.")
        
        if markers is None:
            markers = input_img.peak_local_max(dims=dims, squeeze=False)
        elif isinstance(markers, IndexArray):
            m = PropArray(np.zeros(()), axes="")
            m[()] = markers
            markers = m
                
        input_img._view_labels(self)
        if input_img.dtype == bool:
            input_img = input_img.astype("uint8")
        
        labels = np.zeros(input_img.shape, dtype="uint32")
        input_img.ongoing = "watershed"
        shape = self.sizesof(dims)
        n_labels = 0
        
        for sl, img in input_img.iter(complement_axes(dims, self.axes), israw=True):
            # Make array from max list
            marker_input = np.zeros(shape, dtype="uint32")
            
            sl0 = markers[sl[:-ndim]]
            
            marker_input[tuple(sl0)] = np.arange(1, len(sl0[0])+1, dtype="uint32")
            labels[sl] = skseg.watershed(-img.value, marker_input, mask=img.labels.value, 
                                         connectivity=connectivity)
            labels[sl][labels[sl]>0] += n_labels
            n_labels = labels[sl].max()
            
        input_img.ongoing = None
        del input_img.ongoing
        
        labels = labels.view(Label)
        self.labels = labels.optimize()
        self.labels.set_scale(self)
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
    def regionprops(self, properties:tuple[str,...]=("mean_intensity",), *, 
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
            raise ValueError("axis 'p' is forbidden, in regionprop().")
        
        prop_axes = complement_axes(self.labels.axes, self.axes)
        shape = self.sizesof(prop_axes)
        
        out = ArrayDict({p: PropArray(np.zeros((self.labels.max(),) + shape, dtype="float32"),
                                      name=self.name, 
                                      axes="p" + prop_axes,
                                      dirpath=self.dirpath,
                                      propname = p)
                         for p in properties})
        
        # calculate property value for each slice
        timer = Timer()
        print("regionprops ...", end="")
        for sl in itertools.product(*map(range, shape)):
            props = skmes.regionprops(self.labels, self.value[sl], 
                                      cache=False,
                                      extra_properties=extra_properties)
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]
        timer.toc()
        print(f"\rregionprops completed ({timer})")
        for parr in out.values():
            parr.set_scale(self)
        return out
    
    @same_dtype()
    @record(append_history=False)
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
            axis = find_first_appeared(self.axes, exclude="yx")
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
    
    @record(append_history=False)
    def track_drift(self, axis="t", show_drift=True, **kwargs):
        """
        Calculate xy-directional 
        """
        if self.ndim != 3:
            raise TypeError(f"input must be three dimensional, but got {self.shape}")

        # slow drift needs large upsampling numbers
        corr_kwargs = {"upsample_factor": 10}
        corr_kwargs.update(kwargs)
        
        # self.ongoing = "drift tracking"
        result = [[0.0, 0.0]]
        last_img = None
        for _, img in self.iter(axis):
            if last_img is not None:
                shift = skreg.phase_cross_correlation(last_img, img, return_error=False, **corr_kwargs)
                shift_total = shift + result[-1]    # list + ndarray -> ndarray
                result.append(shift_total)
                last_img = img
            else:
                last_img = img
        
        result = MarkerArray(np.array(result).T, name="drift", axes="rt")
        
        show_drift and plot_drift(result)
        
        return result
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def drift_correction(self, shift=None, ref=None, *, order=1, 
                         along="t", dims=None, update:bool=False):
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
        
        if len(dims) == 3:
            raise NotImplementedError("3-dimensional correction is not implemented. yet")
        
        if shift is None:
            # determine 'ref'
            if ref is None:
                ref = self
            elif not isinstance(ref, self.__class__):
                raise TypeError(f"'ref' must be ImgArray object, but got {type(ref)}")
            elif ref.axes != along + dims:
                raise ValueError(f"Cannot track drift using {ref.axes} image")

            shift = ref.track_drift(axis=along)
            self.ongoing = "drift_correction"
        
        elif isinstance(shift, MarkerArray):
            if shift.ndim != 2:
                raise ValueError("Wrong dimensions of 'shift'.")

            if shift.axes and shift.axes == "tr":
                shift = shift.transpose(1, 0) # rt-order
            elif shift.axes and shift.axes == "rt":
                pass
            else:
                shift.axes = "tr" if shift.shape[0] >= shift.shape[1] else "rt"
                
            nr, nt = shift.sizesof("rt")
            if nr != len(dims) or nt != self.sizeof("t"):
                raise ValueError("Wrong shape of 'shift'.")
        
        else:
            shift = MarkerArray(shift)
            return self.drift_correction(shift, ref, order=order, along=along, 
                                         dims=dims,update=update)

        shift = np.flipud(shift)
        out = np.empty(self.shape)
        for sl, img in self.iter(complement_axes(dims, self.axes)):
            trans = -shift[(slice(None),)+(sl[0],)]
            mx = sktrans.AffineTransform(translation=trans)
            out[sl] = sktrans.warp(img.astype("float32"), mx, order=order)
        
        out = out.view(self.__class__)
        return out

    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def lucy(self, psf, niter:int=50, *, dims=None, update:bool=False):
        """
        Deconvolution of N-dimensional image obtained from confocal microscopy, 
        using Richardson-Lucy's algorithm.
        
        Parameters
        ----------
        psf : np.ndarray
            Point spread function.

        niters : int
            Number of iteration.
        
        dtype : str
            Output dtype
        """
        
        psf = np.asarray(psf, dtype="float32")
        psf /= np.max(psf)
        
        # start deconvolution
        return self.parallel(richardson_lucy_, complement_axes(dims), psf, niter)


# non-member functions.

def array(arr, dtype=None, *, name=None, axes=None) -> ImgArray:
    """
    make an ImgArray object, just like np.array(x)
    """
    if isinstance(arr, str):
        raise TypeError(f"String is invalid input. Do you mean imread(path)?")
    if isinstance(arr, np.ndarray) and dtype is None:
        if arr.dtype in ("uint8", "uint16", "float32"):
            dtype = arr.dtype
        elif arr.dtype.kind == "f":
            dtype = "float32"
        else:
            dtype = arr.dtype
    
    arr = np.array(arr, dtype=dtype)
        
    # Automatically determine axes
    if axes is None:
        axes = ["x", "yx", "tyx", "tzyx", "tzcyx", "ptzcyx"][arr.ndim-1]
            
    self = ImgArray(arr, name=name, axes=axes)
    
    return self

def zeros(shape, dtype="uint16", *, name=None, axes=None) -> ImgArray:
    return array(np.zeros(shape, dtype=dtype), dtype=dtype, name=name, axes=axes)

def zeros_like(img:ImgArray, name:str=None) -> ImgArray:
    if not isinstance(img, ImgArray):
        raise TypeError("'zeros_like' in impy can only take ImgArray as an input")
    
    return zeros(img.shape, dtype=img.dtype, name=name, axes=img.axes)

def empty(shape, dtype="uint16", *, name=None, axes=None) -> ImgArray:
    return array(np.empty(shape, dtype=dtype), dtype=dtype, name=name, axes=axes)

def empty_like(img:ImgArray, name:str=None) -> ImgArray:
    if not isinstance(img, ImgArray):
        raise TypeError("'empty_like' in impy can only take ImgArray as an input")
    
    return empty(img.shape, dtype=img.dtype, name=img.name, axes=img.axes)

def imread(path:str, dtype:str="uint16", *, axes=None) -> ImgArray:
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
    if meta["history"]:
        name = meta["history"].pop(0)
        history = meta["history"]
    else:
        name = fname
        history = []
        
    
    self = ImgArray(img, name=name, axes=axes, dirpath=dirpath, 
                    history=history, metadata=metadata)
        
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

def stack(imgs, axis="c", dtype=None):
    """
    Create stack image from list of images.

    Parameters
    ----------
    imgs : iterable object of images.
        Images to stack. These images must have exactly the same shapes.
    axis : str, optional
        Which axis will be the new one, by default "c"
    dtype : str, optional
        Output dtype.

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

    if dtype is None:
        dtype = imgs[0].dtype

    arrs = [img.as_img_type(dtype).value for img in imgs]

    out = np.stack(arrs, axis=0)
    out = np.moveaxis(out, 0, _axis)
    out = out.view(ImgArray)
    out._set_info(imgs[0], f"Make-Stack(axis={axis})", new_axes)
    
    return out


