from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.fft import fftn as fft, ifftn as ifft, rfftn as rfft, irfftn as irfft
from functools import partial
from .._types import *
from ._skimage import *
from . import _filters, _linalg, _deconv, _misc
from ..func import *
from ..deco import *
from .labeledarray import LabeledArray
from .label import Label
from .phasearray import PhaseArray
from .specials import PropArray
from ..utilcls import *
from ..frame import *

# TODO: check https://github.com/scikit-image/scikit-image/issues/3846

class ImgArray(LabeledArray):
    @same_dtype(asfloat=True)
    def __add__(self, value):
        return super().__add__(value)
    
    @same_dtype(asfloat=True)
    def __iadd__(self, value):
        return super().__iadd__(value)
    
    @same_dtype(asfloat=True)
    def __sub__(self, value):
        return super().__sub__(value)
    
    @same_dtype(asfloat=True)
    def __isub__(self, value):
        return super().__isub__(value)
    
    @same_dtype(asfloat=True)
    def __mul__(self, value):
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot multiply negative value.")
        return super().__mul__(value)
    
    @same_dtype(asfloat=True)
    def __imul__(self, value):
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot multiply negative value.")
        return super().__imul__(value)
    
    def __truediv__(self, value):
        self = self.astype(np.float32)
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
            value[value==0] = np.inf
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot devide negative value.")
        return super().__truediv__(value)
    
    def __itruediv__(self, value):
        self = self.astype(np.float32)
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
            value[value==0] = np.inf
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot devide negative value.")
        return super().__itruediv__(value)
        
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def affine(self, matrix=None, scale=None, rotation=None, shear=None, translation=None, *,
               dims=None, order:int=1) -> ImgArray:
        """
        Convert image by Affine transformation. 2D Affine transformation is written as:
        [x']   [A00 A01 A02]   [x]
        [y'] = [A10 A11 A12] * [y]
        [1 ]   [  0   0   1]   [1]

        Parameters
        ----------
        matrix, scale, rotation, shear, translation
            See `skimage.transform.AffineTransform`.
        dims : int or str, optional
            Spatial dimensions.
        order : int, default is 1.
            Interpolation order after transformation.
            
        Returns
        -------
        ImgArray
            Transformed image.
        """
        mx = sktrans.AffineTransform(matrix=matrix, scale=scale, rotation=rotation, shear=shear,
                                     translation=translation)
        return self.apply_dask(sktrans.warp,
                               c_axes=complement_axes(dims, self.axes),
                               kwargs=dict(inverse_map=mx, order=order)
                               )
    
    @record()
    @same_dtype(True)
    def rotate(self, degree:float, center="center", *, dims="yx", order:int=1) -> ImgArray:
        # TODO: scale sensitive rotation
        if center == "center":
            center = np.array(self.sizesof(dims[::-1]))/2. - 0.5
        else:
            center = np.asarray(center)
            
        translation_0 = sktrans.SimilarityTransform(translation=center)
        rotation = sktrans.SimilarityTransform(rotation=np.deg2rad(degree))
        translation_1 = sktrans.SimilarityTransform(translation=-center)
        mx = translation_1 + rotation + translation_0
        mx.params[2] = (0, 0, 1)
        return self.apply_dask(sktrans.warp,
                               c_axes=complement_axes(dims, self.axes),
                               kwargs=dict(inverse_map=mx, order=order, clip=False)
                               )
                        

    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def translate(self, translation=None, *, dims=None, order:int=1) -> ImgArray:
        """
        Translation of an image. for skimage < 0.19, only 2D translation is implemented.

        Parameters
        ----------
        translation : array-like, optional
            Inverse map of translation. This is xyz-order.
        dims : int or str, optional
            Spatial dimensions.
        order : int, default is 1.
            Interpolation order after transformation.

        Returns
        -------
        ImgArray
            Translated image.
        """        
        ndim = len(dims)
        if translation is None:
            translation = np.zeros(ndim)
        mtx = np.eye(ndim + 1)
        mtx[0:ndim, ndim] = translation
        mx = sktrans.AffineTransform(matrix=mtx)
        return self.apply_dask(sktrans.warp,
                               c_axes=complement_axes(dims, self.axes),
                               kwargs=dict(inverse_map=mx, order=order)
                               )

    @dims_to_spatial_axes
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
        with Progress("rescale"):
            scale_ = [scale if a in dims else 1 for a in self.axes]
            out = sktrans.rescale(self.value, scale_, order=order, anti_aliasing=False)
            out = out.view(self.__class__)
            out._set_info(self, f"rescale(scale={scale})")
            out.axes = str(self.axes) # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out
    
    @dims_to_spatial_axes
    @same_dtype()
    def binning(self, binsize:int=2, method="sum", *, check_edges=True, dims=None) -> ImgArray:
        """
        Binning of images. This function is similar to `rescale` but is strictly binned by N x N blocks.
        Also, any numpy functions that accept "axis" argument are supported for reduce functions.

        Parameters
        ----------
        binsize : int, default is 2
            Bin size, such as 2x2.
        method : str or callable, default is numpy.sum
            Reduce function applied to each bin.
        check_edges : bool, default is True
            If True, only divisible `binsize` is accepted. If False, image is cropped at the end to
            match `binsize`.
        dims : str or int, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Binned image
        """ 
        if isinstance(method, str):
            binfunc = getattr(np, method)
        elif callable(method):
            binfunc = method
        else:
            raise TypeError("`method` must be a numpy function or callable object.")
        
        if binsize == 1:
            return self
        with Progress("binning"):
            img_to_reshape, shape, scale_ = _misc.adjust_bin(self.value, binsize, check_edges, dims, self.axes)
            
            reshaped_img = img_to_reshape.reshape(shape)
            axes_to_reduce = tuple(i*2+1 for i in range(self.ndim))
            out = binfunc(reshaped_img, axis=axes_to_reduce)
            out = out.view(self.__class__)
            out._set_info(self, f"binning(binsize={binsize})")
            out.axes = str(self.axes) # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out
    
    @dims_to_spatial_axes
    def radial_profile(self, nbin:int=32, center:Iterable[float]=None, r_max:float=None, *, 
                       method:str="mean", dims=None) -> PropArray:
        """
        Calculate radial profile of images. Scale along each axis will be considered, i.e., rather 
        ellipsoidal profile will be calculated instead if scales are different between axes.

        Parameters
        ----------
        nbin : int, default is 32
            Number of bins.
        center : iterable of float, optional
            The coordinate of center of radial profile. By default, the center of image is used.
        r_max : float, optional
            Maximum radius to make profile. Region 0 <= r < r_max will be split into `nbin` rings
            (or shells). **Scale must be considered** because scales of each axis may vary.
        method : str, default is "mean"
            Reduce function. Basic statistics functions are supported in `scipy.ndimage` but their
            names are not consistent with those in `numpy`. Use `numpy`'s names here.
        dims : str or int, optional
            Spatial dimensions.

        Returns
        -------
        PropArray
            Radial profile stored in x-axis by default. If input image has tzcyx-axes, then an array 
            with tcx-axes will be returned.
        """        
        func = {"mean": ndi.mean,
                "sum": ndi.sum_labels,
                "median": ndi.median,
                "max": ndi.maximum,
                "min": ndi.minimum,
                "std": ndi.standard_deviation,
                "var": ndi.variance}[method]
        
        spatial_shape = self.sizesof(dims)
        inds = np.indices(spatial_shape)
        
        # check center
        if center is None:
            center = [s/2 for s in spatial_shape]
        elif len(center) != len(dims):
            raise ValueError(f"Length of `center` must match input dimensionality '{dims}'.")
        
        r = np.sqrt(sum(((x - c)/self.scale[a])**2 for x, c, a in zip(inds, center, dims)))
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
        
        c_axes = complement_axes(dims, self.axes)
        
        out = PropArray(np.empty(self.sizesof(c_axes)+(labels.max(),)), dtype=np.float32, axes=c_axes+dims[-1], 
                        dirpath=self.dirpath, metadata=self.metadata, propname="radial_profile")
        radial_func = partial(func, labels=labels, index=np.arange(1, labels.max()+1))
        for sl, img in self.iter(c_axes, exclude=dims):
            out[sl] = radial_func(img)
        return out
    
    @record()
    def gaussfit(self, scale:float=1/16, p0:list=None, show_result:bool=True, 
                 method:str="Powell") -> ImgArray:
        """
        Fit the image to 2-D Gaussian.

        Parameters
        ----------
        scale : float, default is 1/16.
            Scale of rough image (to speed up fitting).
        p0 : list or None, optional
            Initial parameters.
        show_result : bool, default is True
            If True, plot the fitting result on the cross sections.
        method : str, optional
            Fitting method. See `scipy.optimize.minimize`.

        Returns
        -------
        ImgArray
            Fit image.
        """
        if self.ndim != 2:
            raise TypeError(f"input must be two dimensional, but got {self.shape}")
        
        rough = self.rescale(scale).value.astype(np.float32)
        gaussian = GaussianBackground(p0)
        result = gaussian.fit(rough, method=method)
        gaussian.rescale(1/scale)
        fit = gaussian.generate(self.shape).view(self.__class__)
        fit.temp = dict(params=gaussian.params, result=result)
        
        # show fitting result
        show_result and plot_gaussfit_result(self, fit)
        return fit
    
    @record()
    @same_dtype(asfloat=True)
    def gauss_correction(self, ref:ImgArray=None, scale:float=1/16, median_radius:float=15):
        """
        Correct unevenly distributed excitation light using Gaussian fitting. This method subtracts
        background intensity at the same time. If input image is uint, then output value under 0 will
        replaced with 0. If you want to quantify background, it is necessary to first convert input
        image to float image.

        Parameters
        ----------
        ref : ImgArray, default is `self`.
            Reference image to estimate background.
        scale : float, default is 1/16.
            Scale of rough image (to speed up fitting).
        median_radius : float, default is 15.
            Radius of median prefilter's kernel. If smaller than 1, prefiltering will be skipped.

        Returns
        -------
        ImgArray
            Corrected and background subtracted image.
        
        Example
        -------
        (1) When input image has "ptcyx"-axes, and you want to estimate the background intensity
        for each channel by averaging all the positions and times.
        >>> img_cor = img.gauss_correction(ref=img.proj("pt"))
        
        (2) When input image has "ptcyx"-axes, and you want to estimate the background intensity
        for each channel and time point by averaging all the positions.
        >>> img_cor = img.gauss_correction(ref=img.proj("p"))
        """
        if ref is None:
            ref = self
        elif not isinstance(ref, self.__class__):
            raise TypeError(f"`ref` must be None or ImgArray, but got {type(ref)}")
        
        self_loop_axes = complement_axes(ref.axes, self.axes)
        ref_loop_axes = complement_axes("yx", ref.axes)
        out = np.empty(self.shape, dtype=np.float32)
        for sl0, ref_ in ref.iter(ref_loop_axes, israw=True):
            if median_radius >= 1:
                ref_ = ref_.median_filter(radius=median_radius)
            fit = ref_.gaussfit(scale=scale, show_result=False)
            a = fit.max()
            for sl, img in self.iter(self_loop_axes, israw=True):
                out[sl][sl0] = (img[sl0] / fit * a - a).value
        
        return out.view(self.__class__)
    
    @record()
    def affine_correction(self, matrices=None, *, bins:int=256, order:int=1, prefilter:bool=True, 
                          along:str="c") -> ImgArray:
        """
        Correct chromatic aberration using Affine transformation. Input matrix is determined by maximizing
        normalized mutual information.
        
        Parameters
        ----------
        matrices : array or iterable of arrays, optional
            Affine matrices.
        bins : int, default is 256
            Number of bins that is generated on calculating mutual information.
        order : int, optional
            Interporation order, by default 3
        prefilter : bool, default is True.
            If median filter is applied to all images before fitting. This does not
            change original images.
        along : str, default is "c"
            Along which axis correction will be performed.
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
            elif along not in self.axes:
                raise ValueError("Image does not have channel axis.")
            elif self.sizeof(along) < 2:
                raise ValueError("Image must have two channels or more.")
        
        check_c_axis(self)
        
        if isinstance(matrices, np.ndarray):
            # ref is single Affine transformation matrix or a reference image stack.
            matrices = [1, matrices]
                
        elif isinstance(matrices, (list, tuple)):
            # ref is a list of Affine transformation matrix
            matrices = _misc.check_matrix(matrices)
            
        # Determine matrices by fitting
        # if Affine matrix is not given
        if matrices is None:
            if prefilter:
                imgs = self.median_filter(radius=1).split(along)
            else:
                imgs = self.split(along)
            matrices = [1] + [_misc.affinefit(img, imgs[0], bins, order) for img in imgs[1:]]
        
        # check Affine matrix shape
        if len(matrices) != self.sizeof(along):
            nchn = self.sizeof(along)
            raise ValueError(f"{nchn}-channel image needs {nchn} matrices.")
        
        # Affine transformation
        corrected = []
        for i, m in enumerate(matrices):
            if np.isscalar(m) and m==1:
                corrected.append(self[f"{along}={i}"].value)
            else:
                corrected.append(self[f"{along}={i}"].affine(order=order, matrix=m).value)

        out = np.stack(corrected, axis=self.axisof(along)).astype(self.dtype)
        out = out.view(self.__class__)
        out.temp = matrices
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def hessian_eigval(self, sigma:nDFloat=1, *, dims=None) -> ImgArray:
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
        
        Example
        -------
        Extract filament
        >>> eig = -img.hessian_eigval()["l=0"]
        >>> eig[eig<0] = 0
        """        
        ndim = len(dims)
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        
        eigval = self.as_float().apply_dask(_linalg.hessian_eigval, 
                                            c_axes=complement_axes(dims, self.axes), 
                                            new_axis=-1,
                                            args=(sigma, pxsize)
                                            )
        
        eigval.axes = str(self.axes) + "l"
        eigval = eigval.sort_axes()
        eigval._set_info(self, f"hessian_eigval", new_axes=eigval.axes)
        
        return eigval
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def hessian_eig(self, sigma:nDFloat=1, *, dims=None) -> tuple[ImgArray, ImgArray]:
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
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        
        eigs = self.as_float().apply_dask(_linalg.hessian_eigh, 
                                          c_axes=complement_axes(dims, self.axes),
                                          new_axis=[-2, -1],
                                          args=(sigma, pxsize)
                                          )
        
        eigval, eigvec = _linalg.eigs_post_process(eigs, self.axes)
        eigval._set_info(self, f"hessian_eigval", new_axes=eigval.axes)
        eigvec._set_info(self, f"hessian_eigvec", new_axes=eigvec.axes)
        return eigval, eigvec
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def structure_tensor_eigval(self, sigma:nDFloat=1, *, dims=None) -> ImgArray:
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
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        
        eigval = self.as_float().apply_dask(_linalg.structure_tensor_eigval, 
                                            c_axes=complement_axes(dims, self.axes), 
                                            new_axis=-1,
                                            args=(sigma, pxsize),
                                            )
        
        eigval.axes = str(self.axes) + "l"
        eigval = eigval.sort_axes()
        eigval._set_info(self, f"structure_tensor_eigval", new_axes=eigval.axes)
        return eigval
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def structure_tensor_eig(self, sigma:nDFloat=1, *, dims=None)-> tuple[ImgArray, ImgArray]:
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
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        
        eigs = self.as_float().apply_dask(_linalg.structure_tensor_eigh,
                                          c_axes=complement_axes(dims, self.axes),
                                          new_axis=[-2, -1],
                                          args=(sigma, pxsize)
                                          )
        
        eigval, eigvec = _linalg.eigs_post_process(eigs, self.axes)
        eigval._set_info(self, f"structure_tensor_eigval", new_axes=eigval.axes)
        eigvec._set_info(self, f"structure_tensor_eigvec", new_axes=eigvec.axes)
        
        return eigval, eigvec
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def edge_filter(self, method:str="sobel", *, dims=None, update:bool=False) -> ImgArray:
        """
        Sobel filter. This filter is useful for edge detection.

        Parameters
        ----------
        method : str, {"sobel", "farid", "scharr", "prewitt"}, default is "sobel"
            Edge operator name.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        # Get operator
        method_dict = {"sobel": skfil.sobel,
                       "farid": skfil.farid,
                       "scharr": skfil.scharr,
                       "prewitt": skfil.prewitt}
        try:
            f = method_dict[method]
        except KeyError:
            raise ValueError("`method` must be 'sobel', 'farid' 'scharr', or 'prewitt'.")
        
        return self.apply_dask(f, c_axes=complement_axes(dims, self.axes))
    
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @record()
    def lowpass_filter(self, cutoff:nDFloat=0.2, order:float=2, *, dims=None, update:bool=False) -> ImgArray:
        """
        Butterworth low-pass filter.

        Parameters
        ----------
        cutoff : float or array-like, default is 0.2
            Cutoff frequency.
        order : float, default is 2
            Steepness of cutoff.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image
        """         
        from ._skimage import _get_ND_butterworth_filter
        cutoff = check_nd(cutoff, len(dims))
        spatial_shape = self.sizesof(dims)
        spatial_axes = [self.axisof(a) for a in dims]
        weight = _get_ND_butterworth_filter(spatial_shape, cutoff, order, False, True)
        out = irfft(weight*rfft(self.value, axes=spatial_axes), s=spatial_shape, axes=spatial_axes)
        return out
    
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @record()
    def highpass_filter(self, cutoff:nDFloat=0.2, order:float=2, *, dims=None, update:bool=False) -> ImgArray:
        """
        Butterworth high-pass filter.

        Parameters
        ----------
        cutoff : float or array-like, default is 0.2
            Cutoff frequency.
        order : float, default is 2
            Steepness of cutoff.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image
        """         
        from ._skimage import _get_ND_butterworth_filter
        cutoff = check_nd(cutoff, len(dims))
        spatial_shape = self.sizesof(dims)
        spatial_axes = [self.axisof(a) for a in dims]
        weight = _get_ND_butterworth_filter(spatial_shape, cutoff, order, True, True)
        out = irfft(weight*rfft(self.value, axes=spatial_axes), s=spatial_shape, axes=spatial_axes)
        return out
    
    
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @record()
    def convolve(self, kernel, *, mode:str="reflect", cval:float=0, dims=None, 
                 update:bool=False) -> ImgArray:
        """
        General linear convolution by running kernel filtering.

        Parameters
        ----------
        kernel : array-like
            Convolution kernel.
        mode : str, default is "reflect".
            Padding mode. See `scipy.ndimage.convolve`.
        cval : int, default is 0
            Constant value to fill outside the image if mode == "constant".
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self after convolution.

        Returns
        -------
        ImgArray
            Convolved image.
        """        
        return self.apply_dask(ndi.convolve, 
                               c_axes=complement_axes(dims, self.axes), 
                               dtype=self.dtype,
                               args=(kernel,),
                               kwargs=dict(mode=mode, cval=cval)
                               )
    
    @dims_to_spatial_axes
    @same_dtype()
    def _running_kernel(self, radius:float, function=None, *, dims=None, update:bool=False) -> ImgArray:
        disk = ball_like(radius, len(dims))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.apply_dask(function, 
                                  c_axes=complement_axes(dims, self.axes), 
                                  dtype=self.dtype,
                                  args=(disk,)
                                  )
        return out
    
    @record()
    def erosion(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Morphological erosion. If input is binary image, the running function will automatically switched to
        `binary_erosion` to speed up calculation.

        Parameters
        ----------
        radius : float, default is 1.
            Radius of kernel.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        f = skimage.morphology.binary_erosion if self.dtype == bool else skimage.morphology.erosion
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @record()
    def dilation(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Morphological dilation. If input is binary image, the running function will automatically switched to
        `binary_dilation` to speed up calculation.

        Parameters
        ----------
        radius : float, default is 1.
            Radius of kernel.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        f = skimage.morphology.binary_dilation if self.dtype == bool else skimage.morphology.dilation
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @record()
    def opening(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Morphological opening. If input is binary image, the running function will automatically switched to
        `binary_opening` to speed up calculation.

        Parameters
        ----------
        radius : float, default is 1.
            Radius of kernel.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        f = skimage.morphology.binary_opening if self.dtype == bool else skimage.morphology.opening
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @record()
    def closing(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Morphological closing. If input is binary image, the running function will automatically switched to
        `binary_closing` to speed up calculation.

        Parameters
        ----------
        radius : float, default is 1.
            Radius of kernel.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        f = skimage.morphology.binary_closing if self.dtype == bool else skimage.morphology.closing
        return self._running_kernel(radius, f, dims=dims, update=update)
    
    @dims_to_spatial_axes
    @record()
    def tophat(self, radius:float=50, *, dims=None, update:bool=False) -> ImgArray:
        """
        Tophat morphological image processing. This is useful for background subtraction.

        Parameters
        ----------
        radius : float, default is 50.
            Radius of kernel.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        
        disk = ball_like(radius, len(dims))
        return self.apply_dask(ndi.white_tophat, 
                               c_axes=complement_axes(dims, self.axes), 
                               dtype=self.dtype,
                               kwargs=dict(footprint=disk)
                               )
    
    @dims_to_spatial_axes
    @record()
    def mean_filter(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Mean filter. Kernel is filled with same values.

        Parameters
        ----------
        radius : float, default is 1
            Radius of kernel.
        dims : str, optional
            Spatial dimensions.
        update : bool, optional
            If update self after filtering.
            
        Returns
        -------
        ImgArray
            Filtered image
        """        
        return self._running_kernel(radius, _filters.mean_filter, dims=dims, update=update)
    
    @dims_to_spatial_axes
    @record()
    def std_filter(self, radius:float=1, *, dims=None) -> ImgArray:
        """
        Standard deviation filter.

        Parameters
        ----------
        radius : float, default is 1
            Radius of kernel.
        dims : str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Filtered image
        """        
        disk = ball_like(radius, len(dims))
        return self.as_float().apply_dask(_filters.std_filter, 
                                          c_axes=complement_axes(dims, self.axes), 
                                          args=(disk,)
                                          )
    
    @dims_to_spatial_axes
    @record()
    def coef_filter(self, radius:float=1, *, dims=None) -> ImgArray:
        """
        Coefficient of variance filter. For kernel area X, std(X)/mean(X) are calculated.
        This filter is useful for feature extraction from images with uneven background intensity.

        Parameters
        ----------
        radius : float, default is 1
            Radius of kernel.
        dims : str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Filtered image
        """        
        disk = ball_like(radius, len(dims))
        return self.as_float().apply_dask(_filters.coef_filter, 
                                          c_axes=complement_axes(dims, self.axes), 
                                          args=(disk,)
                                          )
    
    @dims_to_spatial_axes
    @record()
    def median_filter(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Running median filter. This filter is useful for deleting outliers generated by noise.

        Parameters
        ----------
        radius : float, default is 1.
            Radius of kernel.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self after filtering.

        Returns
        -------
        ImgArray
            Filtered image.
        """     
        disk = ball_like(radius, len(dims))
        return self.apply_dask(ndi.median_filter, 
                               c_axes=complement_axes(dims, self.axes), 
                               dtype=self.dtype,
                               kwargs=dict(footprint=disk)
                               )
    
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def diameter_opening(self, diameter:int=8, *, connectivity:int=1, dims=None, 
                         update:bool=False) -> ImgArray:
        return self.apply_dask(skimage.morphology.diameter_opening, 
                               c_axes=complement_axes(dims, self.axes), 
                               kwargs=dict(diameter_threshold=diameter, connectivity=connectivity)
                               )
        
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def diameter_closing(self, diameter:int=8, *, connectivity:int=1, dims=None,
                         update:bool=False) -> ImgArray:
        return self.apply_dask(skimage.morphology.diameter_closing, 
                               c_axes=complement_axes(dims, self.axes), 
                               kwargs=dict(diameter_threshold=diameter, connectivity=connectivity)
                               )
    
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def area_opening(self, area:int=64, *, connectivity:int=1, dims=None, 
                     update:bool=False) -> ImgArray:
        if self.dtype == bool:
            return self.apply_dask(skimage.morphology.remove_small_objects,
                                   c_axes=complement_axes(dims, self.axes), 
                                   kwargs=dict(min_size=area, connectivity=connectivity)
                                   )
        else:
            return self.apply_dask(skimage.morphology.area_opening, 
                                   c_axes=complement_axes(dims, self.axes), 
                                   kwargs=dict(area_threshold=area, connectivity=connectivity)
                                   )
        
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def area_closing(self, area:int=64, *, connectivity:int=1, dims=None, 
                     update:bool=False) -> ImgArray:
        if self.dtype == bool:
            return self.apply_dask(skimage.morphology.remove_small_holes,
                                   c_axes=complement_axes(dims, self.axes), 
                                   kwargs=dict(min_size=area, connectivity=connectivity)
                                   )
        else:
            return self.apply_dask(skimage.morphology.area_closing, 
                                   c_axes=complement_axes(dims, self.axes), 
                                   kwargs=dict(area_threshold=area, connectivity=connectivity)
                                   )
    
    @dims_to_spatial_axes
    @record()
    def entropy_filter(self, radius:nDFloat=5, *, dims=None) -> ImgArray:
        """
        Running entropy filter. This filter is useful for detecting change in background distribution.

        Parameters
        ----------
        radius : float, default is 5
            Kernel radius of the filter.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        disk = ball_like(radius, len(dims))
        
        self = self.as_uint8()
        return self.apply_dask(skfil.rank.entropy, 
                               c_axes=complement_axes(dims, self.axes),
                               kwargs=dict(selem=disk)
                               ).as_float()
    
    @dims_to_spatial_axes
    @record()
    def enhance_contrast(self, radius:nDFloat=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Enhance contrast filter.

        Parameters
        ----------
        radius : int, optional
            Kernel radius of the filter.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Contrast enhanced image.
        """        
        return self._running_kernel(radius, skfil.rank.enhance_contrast, dims=dims, update=update)
    
    @dims_to_spatial_axes
    @record()
    def laplacian_filter(self, radius:int=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Edge detection using Laplacian filter. Kernel is made by skimage's function.

        Parameters
        ----------
        radius : int, default is 1
            Radius of kernel. Shape of kernel will be (2*radius+1, 2*radius+1).
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        ndim = len(dims)
        _, laplace_op = skres.uft.laplacian(ndim, (2*radius+1,) * ndim)
        return self.apply_dask(ndi.convolve, 
                               c_axes=complement_axes(dims, self.axes), 
                               dtype=self.dtype,
                               args=(laplace_op,),
                               kwargs=dict(mode="reflect")
                               )
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def kalman_filter(self, gain:float=0.8, noise_var:float=0.05, *, along:str="t", dims=None, 
                      update:bool=False) -> ImgArray:
        """
        Kalman filter for image smoothing. This function is same as "Kalman Stack Filter" in ImageJ but support
        batch processing. This filter is useful for preprocessing of particle tracking.

        Parameters
        ----------
        gain : float, default is 0.8
            Filter gain.
        noise_var : float, default is 0.05
            Initial estimate of noise variance.
        along : str, default is "t"
            Which axis will be the time axis.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Filtered image
        """        
        t_axis = self.axisof(along)
        min_a = min(self.axisof(a) for a in dims)
        if t_axis > min_a:
            self = np.swapaxes(self, t_axis, min_a)
        out = self.apply_dask(_filters.kalman_filter, 
                              c_axes=complement_axes(along + dims, self.axes), 
                              args=(gain, noise_var)
                              )
        if t_axis > min_a:
            out = np.swapaxes(out, min_a, t_axis)
                
        return out
    
    @record(append_history=False)
    def focus_map(self, radius:int=1, *, dims="yx") -> PropArray:
        """
        Compute focus map using variance of Laplacian method. yx-plane with higher variance is likely a
        focal plane because sharper image causes higher value of Laplacian on the edges.

        Parameters
        ----------
        radius : int, default is 1
            Radius of Laplacian filter's kernel.

        Returns
        -------
        PropArray
            Array of variance of Laplacian
        
        Example
        -------
        Get the focus plane from a 3D image.
        >>> score = img.focus_map()
        >>> score.plot()               # plot the variation of laplacian focus
        >>> z_focus = np.argmax(score) # determine the focus plane
        >>> img[z_focus]               # get the focus plane
        """        
        c_axes = complement_axes(dims, self.axes)
        laplace_img = self.as_float().laplacian_filter(radius, dims=dims)
        out = PropArray(np.empty(self.sizesof(c_axes)), dtype=np.float32, name=self.name, 
                        axes=c_axes, propname="variance_of_laplacian")
        
        for sl, img in laplace_img.iter(c_axes, exclude=dims):
            out[sl] = np.var(img)
        return out
    
    @record()
    @same_dtype(asfloat=True)
    def unmix(self, matrix, bg=None, *, along:str="c", update:bool=False) -> ImgArray:
        """
        Unmix fluorescence leakage between channels in a linear way. For example, a blue/green image,
        fluorescent leakage can be written as following equation:
            { B_obs =     B_real + a * G_real
            { G_obs = b * B_real +     G_real
        where "obs" means observed intensities, "real" means the real intensity. In this linear case, 
        leakage matrix:
            M = [ 1, a]  Vobs = M * Vreal
                [ b, 1], 
        must be predefined. If M is given, then real intensities can be restored by:
            Vreal = M^-1 * Vobs
        
        Parameters
        ----------
        matrix : array-like
            Leakage matrix. The (i, j) component of the matrix is the leakage from i-th channel to
            j-th channel.
        bg : array-like, optional
            Vector of background intensities for each channel. If not given, it is assumed to be the
            minimum value of each channel.
        along : str, default is "c"
            The axis of channel.
        update : bool, default is False
            If update self to unmixed image.

        Returns
        -------
        ImgArray
            Unmixed image.
        
        Example
        -------
        Complement the channel-0 to channel-1 leakage.
        >>> mtx = [[1.0, 0.4],
        >>>        [0.0, 1.0]]
        >>> bg = [1500, 1200]
        >>> unmixed_img = img.unmix(mtx, bg)
        """        
        n_chn = self.sizeof(along)
        c_ax = self.axisof(along)
        
        # check matrix and bg
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (n_chn, n_chn):
            raise ValueError(f"`map_matrix` must have shape {(n_chn, n_chn)}")
        if bg is None:
            bg = np.array([self.value[i].min() for i in range(n_chn)])
        bg = np.asarray(bg, dtype=np.float32).ravel()
        if bg.size != n_chn:
            raise ValueError(f"`bg` must have length {n_chn}")
        
        # move channel axis to the last
        input_ = np.moveaxis(np.asarray(self, dtype=np.float32), c_ax, -1)
        # multiply inverse matrix
        out = (input_ - bg) @ np.linalg.inv(matrix) + bg
        # restore the axes order
        out = np.moveaxis(out, -1, c_ax)
        
        return out.view(self.__class__)
        
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def fill_hole(self, thr:float|str="otsu", *, dims=None, update:bool=False) -> ImgArray:
        """
        Filling holes.
        
        Reference
        ---------
        https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html

        Parameters
        ----------
        thr : scalar or str, optional
            Threshold (value or method) to apply if image is not binary.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
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
                    
        return self.apply_dask(_filters.fill_hole, 
                               c_axes=complement_axes(dims, self.axes), 
                               args=(mask,),
                               dtype=self.dtype
                               )
    

    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def gaussian_filter(self, sigma:nDFloat=1, *, dims=None, update:bool=False) -> ImgArray:
        """
        Run Gaussian filter (Gaussian blur).
        
        Parameters
        ----------
        sigma : scalar or array of scalars, optional
            Standard deviation(s) of Gaussian.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to filtered image.
            
        Returns
        -------
        ImgArray
            Filtered image.
        """
        return self.apply_dask(ndi.gaussian_filter, 
                               c_axes=complement_axes(dims, self.axes), 
                               args=(sigma,), 
                               dtype=np.float32
                               )


    @dims_to_spatial_axes
    @record()
    def dog_filter(self, low_sigma:nDFloat=1, high_sigma:nDFloat=None, *, dims=None) -> ImgArray:
        """
        Run Difference of Gaussian filter. This function does not support `update`
        argument because intensity can be negative.
        
        Parameters
        ----------
        low_sigma : scalar or array of scalars, default is 1.
            lower standard deviation(s) of Gaussian.
        high_sigma : scalar or array of scalars, default is x1.6 of low_sigma.
            higher standard deviation(s) of Gaussian.
        dims : int or str, optional
            Spatial dimensions.
            
        Returns
        -------
        ImgArray
            Filtered image.
        """        
        
        low_sigma = np.array(check_nd(low_sigma, len(dims)))
        high_sigma = low_sigma * 1.6 if high_sigma is None else high_sigma
        
        return self.as_float().apply_dask(_filters.dog_filter, 
                                          c_axes=complement_axes(dims, self.axes),
                                          args=(low_sigma, high_sigma)
                                          )
    
    @dims_to_spatial_axes
    @record()
    def doh_filter(self, sigma:nDFloat=1, *, dims=None) -> ImgArray:
        """
        Determinant of Hessian filter. This function does not support `update`
        argument because output has total different scale of intensity. Because in
        most cases we want to find only bright dots, eigenvalues larger than 0 is
        ignored before computing determinant.

        Parameters
        ----------
        sigma : scalar or array of scalars, default is 1.
            Standard deviation(s) of Gaussian filter.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Filtered image.
        """    
        sigma = check_nd(sigma, len(dims))
        pxsize = np.array([self.scale[a] for a in dims])
        return self.as_float().apply_dask(_filters.doh_filter, 
                                          c_axes=complement_axes(dims, self.axes), 
                                          args=(sigma, pxsize)
                                          )
    
    @dims_to_spatial_axes
    @record()
    def log_filter(self, sigma:nDFloat=1, *, dims=None) -> ImgArray:
        """
        Laplacian of Gaussian filter.

        Parameters
        ----------
        sigma : scalar or array of scalars, default is 1.
            Standard deviation(s) of Gaussian filter.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        return -self.as_float().apply_dask(ndi.gaussian_laplace,
                                           c_axes=complement_axes(dims, self.axes), 
                                           args=(sigma,)
                                           )
    
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def rolling_ball(self, radius:float=50, prefilter:str="mean", *, return_bg:bool=False,
                     dims=None, update:bool=False) -> ImgArray:
        """
        Subtract Background using rolling-ball algorithm.

        Parameters
        ----------
        radius : int, default is 50.
            Radius of rolling ball.
        prefilter : str, {"mean", "median", "none"}
            If apply 3x3 averaging before creating background.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to filtered image.
            
        Returns
        -------
        ImgArray
            Background subtracted image.
        """        
        method = ("mean", "median", "none")
        c_axes = complement_axes(dims, self.axes)
        if prefilter == "mean":
            filt = self.apply_dask(_filters.mean_filter, 
                                   c_axes=c_axes, 
                                   kwargs=dict(selem=np.ones((3,)*len(dims)))
                                   )
        elif prefilter == "median":
            filt = self.apply_dask(ndi.median_filter, 
                                   c_axes=c_axes, 
                                   kwargs=dict(footprint=np.ones((3,)*len(dims)))
                                   )
        elif prefilter == "none":
            filt = self
        else:
            raise ValueError(f"`prefilter` must be {', '.join(method)}.")
        filt.axes = self.axes
        back = filt.apply_dask(skres.rolling_ball, 
                               c_axes=c_axes, 
                               kwargs=dict(radius=radius))
        if not return_bg:
            out = filt.value - back
            return out
        else:
            return back
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def rof_filter(self, lmd:float=0.05, tol:float=1e-4, max_iter:int=50, *, dims=None, 
                   update:bool=False) -> ImgArray:
        """
        Rudin-Osher-Fatemi's total variation denoising.

        Parameters
        ----------
        lmd : float, default is 0.05
            Constant value in total variation.
        tol : float, default is 1e-4
            Iteration stops when gain is under this value.
        max_iter : int, default is 50
            Maximum number of iterations.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Filtered image
        """        
        return self.apply_dask(skres._denoise._denoise_tv_chambolle_nd, 
                               c_axes=complement_axes(dims, self.axes),
                               kwargs=dict(weight=lmd, eps=tol, n_iter_max=max_iter)
                               )
        
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def wavelet_denoising(self, noise_sigma:float=None, *, wavelet:str="db1", mode:str="soft", 
                          wavelet_levels:int=None, method:str="BayesShrink", max_shifts:int|tuple=0,
                          shift_steps:int|tuple=1, dims=None) -> ImgArray:
        """
        Wavelet denoising. Because it is not shift invariant, `cycle_spin` is called inside the 
        function.

        Parameters
        ----------
        noise_sigma : float, optional
            Standard deviation of noise, if known.
        wavelet : str, default is "db1"
            Any options of `pywt.wavelist`.
        mode : {"soft", "hard"}, default is "soft"
            Type of denoising.
        wavelet_levels : int, optional
            The number of wavelet decomposition levels to use.
        method : {"BayesShrink", "VisuShrink"}, default is "BayesShrink"
            Thresholding method to be used
        max_shifts : int or tuple, default is 0
            Shifts in range(0, max_shifts+1) will be used.
        shift_steps : int or tuple, default is 1
            Step size of shifts.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Denoised image.
        """        
        func_kw=dict(sigma=noise_sigma, 
                     wavelet=wavelet, 
                     mode=mode, 
                     wavelet_levels=wavelet_levels,
                     method=method)
        return self.apply_dask(skres.cycle_spin, 
                               c_axes=complement_axes(dims, self.axes), 
                               args=(skres.denoise_wavelet,),
                               kwargs=dict(func_kw=func_kw, max_shifts=max_shifts, shift_steps=shift_steps)
                               )
    
    @record(append_history=False)
    def split_pixel_unit(self, center:tuple[float, float]=(0, 0), *, order:int=1,
                           angle_order:list[int]=None, newaxis="<") -> ImgArray:
        """
        Split a (2*N, 2*M)-image into four (N, M)-images for each other pixels. Generally, image 
        acquisition with a polarization camera will output (2*N, 2*M)-image with N x M pixel units:
        
        [0] [1] [0] [1] [0] [1] ...
        [3] [2] [3] [2] [3] [2]
        [0] [1] [0] [1] [0] [1]
        [3] [2] [3] [2] [3] [2] ...
         :                   :
        
        This function generates images only consist of positions of [0], [1], [2] or [3]. Strictly,
        each image is acquired from different position (the pixel (i,j) in [0]-image and the pixel
        (i,j) in [1]-image are acquired from different positions). This function also complement for
        this difference by Affine transformation and spline interpolation.
         
        Parameters
        ----------
        center : tuple (a, b), where 0 <= a <= 1 and 0 <= b <= 1, default is (0, 0)
            Coordinate that will be considered as the center. For example, center=(0, 0) means the most
            upper left pixel, and center=(0.5, 0.5) means the middle point of a pixel unit.
            [0] [1]    (0, 0) (0, 1)
            [2] [3] -> (1, 0) (1, 1)
        order : int, default is 4.
            Spline interpolation order. For detail see `skimage.transform.warp`. To speed up you can
            pass smaller integer in the expense of accuracy.
        angle_order : list of int, default is [2, 1, 0, 3]
            Specify which pixels correspond to which polarization angles. 0, 1, 2 and 3 corresponds to
            polarization of 0, 45, 90 and 135 degree respectively. This list will be directly passed to
            np.ndarray like `arr[angle_order]` to sort it. For example, if a pixel unit receives 
            polarized light like below:
            [0] [1]    [ 90] [ 45]    [|] [/]
            [2] [3] -> [135] [  0] or [\] [-]
            then `angle_order` should be [2, 1, 0, 3].
            
        Returns
        -------
        ImgArray
            Axis "<" is added in the first dimension.　For example, If input is "tyx"-axes, then output
            will be "<tyx"-axes.
        
        Example
        -------
        Extract polarization in 0-, 45-, 90- and 135-degree directions from an image that is acquired
        from a polarization camera, and calculate total intensity of light by averaging.
        
        >>> img_pol = img.split_pixel_unit()
        >>> img_total = img_pol.proj(axis="<")
        """        
        yc, xc = center
        if angle_order is None:
            angle_order = [2, 1, 0, 3]
        imgs = []
        for y, x in [(0,0), (0,1), (1,1), (1,0)]:
            dr = [(xc-x)/2, (yc-y)/2]
            imgs.append(self[f"y={y}::2;x={x}::2"].translate(translation=dr, order=order).value)
        imgs = np.stack(imgs, axis=0)
        imgs = imgs[angle_order]
        imgs = imgs.view(self.__class__)
        imgs._set_info(self, "split_pixel_unit", newaxis + str(self.axes))
        imgs.set_scale(y=self.scale["y"]*2, x=self.scale["x"]*2)
        return imgs
        
    def stokes(self, *, along:str="<") -> ArrayDict:
        """
        Generate stocks images from an image stack with polarized images. Currently, Degree of Linear 
        Polarization (DoLP) and Angle of Polarization (AoP) will be calculated. Those irregular values
        (np.nan, np.inf) will be replaced with 0. Be sure that to calculate DoPL correctly background
        subtraction must be applied beforehand because stokes parameter `s0` is affected by absolute
        intensities.

        Parameters
        ----------
        along : str, default is "<"
            To define which axis is polarization angle axis. Along this axis the angle of polarizer must be
            in order of 0, 45, 90, 135 degree.

        Returns
        -------
        ArrayDict
            Dictionaly with keys "dolp" and "aop", which correspond to DoPL and AoP respectively.
        
        Example
        -------
        Calculate AoP image from the raw image and display them.
        >>> img_pol = img.split_polarization()
        >>> dpol = img_pol.stokes()
        >>> ip.gui.add(img_pol.proj)
        >>> ip.gui.add(dpol.aop.rad2deg())
        
        References
        ----------
        - https://mavic.ne.jp/indutrialcamera-polarization-inline/
        - Yang, J., Qiu, S., Jin, W., Wang, X., & Xue, F. (2020). Polarization imaging model considering the 
          non-ideality of polarizers. Applied optics, 59(2), 306-314.
        - Feng, B., Guo, R., Zhang, F., Zhao, F., & Dong, Y. (2021). Calculation and hue mapping of AoP in 
          polarization imaging. May. https://doi.org/10.1117/12.2523643
        """
        new_axes = complement_axes(along, self.axes)
        img0, img45, img90, img135 = [a.as_float().value for a in self.split(along)]
        # Stokes parameters
        s0 = (img0 + img45 + img90 + img135)/2
        s1 = img0 - img90
        s2 = img45 - img135
        
        # Degree of Linear Polarization (DoLP)
        # DoLP is defined as:
        # DoLP = sqrt(s1^2 + s2^2)/s0
        s0[s0==0] = np.inf
        dolp = np.sqrt(s1**2 + s2**2)/s0
        dolp = dolp.view(self.__class__)
        dolp._set_info(self, "dolp", new_axes=new_axes)
        dolp.set_scale(self)
        
        # Angle of Polarization (AoP)
        # AoP is usually calculated as psi = 1/2argtan(s1/s2), but this is wrong because left side
        # has range of [0, pi) while right side has range of [-pi/4, pi/4). The correct formulation is:
        #       { 1/2argtan(s2/s1)          (s1>0 and s2>0)
        # AoP = { 1/2argtan(s2/s1) + pi/2   (s1<0)
        #       { 1/2argtan(s2/s1) + pi     (s1>0 and s2<0)
        # But here, np.arctan2 can detect the signs of inputs s1 and s2, so that it returns correct values.
        aop = np.arctan2(s2, s1)/2
        aop = aop.view(PhaseArray)
        aop._set_info(self, "aop", new_axes=new_axes)
        aop.unit = "rad"
        aop.border = (-np.pi/2, np.pi/2)
        aop.fix_border()
        aop.set_scale(self)
        
        out = ArrayDict(dolp=dolp, aop=aop)
        return out
        
    @dims_to_spatial_axes
    @record(append_history=False)
    def peak_local_max(self, *, min_distance:int=1, percentile:float=None, 
                       topn:int=np.inf, topn_per_label:int=np.inf, exclude_border:bool=True,
                       use_labels:bool=True, dims=None) -> MarkerFrame:
        """
        Find local maxima. This algorithm corresponds to ImageJ's 'Find Maxima' but
        is more flexible.

        Parameters
        ----------
        min_distance : int, default is 1
            Minimum distance allowed for each two peaks. This parameter is slightly
            different from that in `skimage.feature.peak_local_max` because here float
            input is allowed and every time footprint is calculated.
        percentile : float, optional
            Percentile to compute absolute threshold.
        topn : int, optional
            Maximum number of peaks **for each iteration**.
        topn_per_label : int, default is np.inf
            Maximum number of peaks per label.
        use_labels : bool, default is True
            If use self.labels when it exists.
        dims : int or str, optional
            Spatial dimensions.
            
        Returns
        -------
        MarkerFrame
            DataFrame with columns same as axes of self. For example, if self.axes is "tcyx" then
            return value has "t", "c", "y" and "x" columns, and sub-frame at t=0, c=0 contains all
            the coordinates of peaks in the slice at t=0, c=0.
        """        
        
        # separate spatial dimensions and others
        ndim = len(dims)
        dims_list = list(dims)
        c_axes = complement_axes(dims, self.axes)
        c_axes_list = list(c_axes)
        
        if isinstance(exclude_border, bool):
            exclude_border = int(min_distance) if exclude_border else False
        
        thr = None if percentile is None else np.percentile(self.value, percentile)
        df_all = []
        for sl, img in self.iter(c_axes, israw=True, exclude=dims):
            # skfeat.peak_local_max overwrite something so we need to give copy of img.
            if use_labels and hasattr(img, "labels"):
                labels = np.array(img.labels)
            else:
                labels = None
            
            indices = skfeat.peak_local_max(np.array(img),
                                            footprint=ball_like(min_distance, ndim),
                                            threshold_abs=thr,
                                            num_peaks=topn,
                                            num_peaks_per_label=topn_per_label,
                                            labels=labels,
                                            exclude_border=exclude_border)
            indices = pd.DataFrame(indices, columns=dims_list)
            indices[c_axes_list] = sl
            df_all.append(indices)
            
        df_all = pd.concat(df_all, axis=0)
        df_all = MarkerFrame(df_all, columns=self.axes, dtype=np.uint16)
        df_all.set_scale(self)
        return df_all
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def corner_peaks(self, *, min_distance:int=1, percentile:float=None, 
                     topn:int=np.inf, topn_per_label:int=np.inf, exclude_border:bool=True,
                     use_labels:bool=True, dims=None) -> MarkerFrame:
        """
        Find local corner maxima. Slightly different from peak_local_max.

        Parameters
        ----------
        min_distance : int, default is 1
            Minimum distance allowed for each two peaks. This parameter is slightly
            different from that in `skimage.feature.peak_local_max` because here float
            input is allowed and every time footprint is calculated.
        percentile : float, optional
            Percentile to compute absolute threshold.
        topn : int, optional
            Maximum number of peaks **for each iteration**.
        topn_per_label : int, default is np.inf
            Maximum number of peaks per label.
        use_labels : bool, default is True
            If use self.labels when it exists.
        dims : int or str, optional
            Spatial dimensions.
            
        Returns
        -------
        MarkerFrame
            DataFrame with columns same as axes of self. For example, if self.axes is "tcyx" then
            return value has "t", "c", "y" and "x" columns, and sub-frame at t=0, c=0 contains all
            the coordinates of corners in the slice at t=0, c=0.
        """        
        
        # separate spatial dimensions and others
        ndim = len(dims)
        dims_list = list(dims)
        c_axes = complement_axes(dims, self.axes)
        c_axes_list = list(c_axes)
        
        if isinstance(exclude_border, bool):
            exclude_border = int(min_distance) if exclude_border else False
        
        thr = None if percentile is None else np.percentile(self.value, percentile)
        
        df_all = []
        for sl, img in self.iter(c_axes, israw=True, exclude=dims):
            # skfeat.corner_peaks overwrite something so we need to give copy of img.
            if use_labels and hasattr(img, "labels"):
                labels = np.array(img.labels)
            else:
                labels = None
            
            indices = skfeat.corner_peaks(np.array(img),
                                          footprint=ball_like(min_distance, ndim),
                                          threshold_abs=thr,
                                          num_peaks=topn,
                                          num_peaks_per_label=topn_per_label,
                                          labels=labels,
                                          exclude_border=exclude_border)
            indices = pd.DataFrame(indices, columns=dims_list)
            indices[c_axes_list] = sl
            df_all.append(indices)
            
        df_all = pd.concat(df_all, axis=0)
        df_all = MarkerFrame(df_all, columns=self.axes, dtype="uint16")
        df_all.set_scale(self)
        return df_all
    
    @dims_to_spatial_axes
    @record()
    def corner_harris(self, sigma:nDFloat=1, k:float=0.05, *, dims=None) -> ImgArray:
        """
        Calculate Harris response image.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of Gaussian prefilter.
        k : float, optional
            Sensitivity factor to separate corners from edges, typically in range [0, 0.2].
            Small values of k result in detection of sharp corners.
        dims : str or int, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Harris response
        """        
        return self.apply_dask(skfeat.corner_harris, 
                               c_axes=complement_axes(dims, self.axes), 
                               kwargs=dict(k=k, sigma=sigma)
                               )
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def find_corners(self, sigma:nDFloat=1, k:float=0.05, *, dims=None) -> ImgArray:
        """
        Corner detection using Harris response.

        Parameters
        ----------
        sigma : float or array-like, optional
            Standard deviation of Gaussian prefilter.
        k : float, optional
            Sensitivity factor to separate corners from edges, typically in range [0, 0.2].
            Small values of k result in detection of sharp corners.
        dims : str or int, optional
            Spatial dimensions.

        Returns
        -------
        MarkerFrame
            Coordinates of corners. For details see `corner_peaks` method.
        """        
        res = self.gaussian_filter(sigma=1).corner_harris(sigma=sigma, k=k, dims=dims)
        out = res.corner_peaks(min_distance=3, percentile=97, dims=dims)
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def voronoi(self, coords:Coords, *, inf:nDInt=None, dims="yx") -> ImgArray:
        """
        Voronoi segmentation of an image. Image region labeled with $i$ means that all
        the points in the region are closer to the $i$-th point than any other points.

        Parameters
        ----------
        coords : MarkerFrame or (N, 2) array-like
            Coordinates of points.
        inf : int, array of int, optional
            Distance to infinity points. If not provided, infinity points are placed at
            100 times further positions relative to the image shape.
        dims : int or str, default is "yx"
            Spatial dimensions

        Returns
        -------
        ImgArray
            Image labeled with segmentation.
        """        
        from scipy.spatial import Voronoi
        coords = _check_coordinates(coords, self, dims=self.axes)
        
        ny, nx = self.sizesof(dims)
        
        if inf is None:
            infy = ny * 100
            infx = nx * 100
        elif isinstance(inf, int):
            infy = infx = inf
        else:
            infy, infx = inf
        
        infpoints = [[-infy, -infx], [-infy, nx+infx], [ny+infy, -infx], [ny+infy, nx+infx]]
        
        labels = largest_zeros(self.shape)
        n_label = 1
        for sl, crds in coords.iter(complement_axes(dims, self.axes)):
            vor = Voronoi(crds.values.tolist() + infpoints)
            for r in vor.regions:
                if all(r0 > 0 for r0 in r):
                    poly = vor.vertices[r]
                    grids = skmes.grid_points_in_poly(self.sizesof(dims), poly)
                    labels[sl][grids] = n_label
                    n_label += 1
        self.labels = Label(labels, name=self.name, axes=self.axes, dirpath=self.dirpath).optimize()
        self.labels.set_scale(self)

        return self
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def flood(self, seeds:Coords, *, connectivity:int=1, tolerance:float=None, dims=None):
        """
        Flood filling with a list of seed points. By repeating skimage's `flood` function,
        this method can perform segmentation of an image.

        Parameters
        ----------
        seeds : MarkerFrame or (N, D) array-like
            Seed points to start flood filling.
        connectivity : int, default is 1
            Defines connectivity structure.
        tolerance : float, optional
            Intensity deviation within this value will be filled.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Labeled image.
        """        
        seeds = _check_coordinates(seeds, self, dims=self.axes)
        labels = largest_zeros(self.shape)
        n_label_next = 1
        for sl, crds in seeds.iter(complement_axes(dims, self.axes)):
            for crd in crds.values:
                crd = tuple(crd)
                if labels[sl][crd] > 0:
                    n_label = labels[sl][crd]
                else:
                    n_label = n_label_next
                    n_label_next += 1
                fill_area = skimage.morphology.flood(self.value[sl], crd, connectivity=connectivity, 
                                                     tolerance=tolerance)
                labels[sl][fill_area] = n_label
        
        self.labels = Label(labels, name=self.name, axes=self.axes, dirpath=self.dirpath).optimize()
        self.labels.set_scale(self)
        return self
    
    @dims_to_spatial_axes
    def refine_sm(self, coords:Coords=None, radius:float=4, *, percentile:float=95, n_iter:int=10, 
                  sigma:float=1.5, dims=None):
        """
        Refine coordinates of peaks and calculate positional errors using `trackpy`'s functions. Mean
        and noise level are determined using original method.

        Parameters
        ----------
        coords : MarkerFrame or (N, D) array, optional
            Coordinates of peaks. If None, this will be determined by `find_sm`.
        radius : float, default is 4.
            Range to mask single molecules.
        percentile : int, default is 95
            Passed to peak_local_max()
        n_iter : int, default is 10
            Number of iteration of refinement.
        sigma : float, default is 1.5
            Expected standard deviation of particles.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        FrameDict
            Coordinates in MarkerFrame and refinement results in pd.DataFrame.
        """        
        import trackpy as tp
        if coords is None:
            coords = self.find_sm(sigma=sigma, dims=dims, percentile=percentile, exclude_border=radius)
        else:
            coords = _check_coordinates(coords, self, dims=self.axes)
        
        if hasattr(self, "labels"):
            labels_now = self.labels.copy()
        else:
            labels_now = None
        self.labels = None
        del self.labels
        self.specify(coords, radius, labeltype="circle")
        
        # set parameters
        radius = check_nd(radius, len(dims))
        sigma = tuple(map(int, check_nd(sigma, len(dims))))
        sigma = tuple([int(x) for x in sigma])
        
        df_all = []
        c_axes = complement_axes(dims, self.axes)
        c_axes_list = list(c_axes)
        dims_list = list(dims)
        with Progress("refine_sm"):
            for sl, crds in coords.iter(c_axes):
                img = self[sl]
                refined_coords = tp.refine.refine_com(img.value, img.value, radius, crds,
                                                      max_iterations=n_iter, pos_columns=dims_list)
                bg = img.value[img.labels==0]
                black_level = np.mean(bg)
                noise = np.std(bg)
                area = np.sum(ball_like_odd(radius[0], len(dims)))
                mass = refined_coords["raw_mass"].values - area * black_level
                ep = tp.uncertainty._static_error(mass, noise, radius, sigma)
                
                if ep.ndim == 1:
                    refined_coords["ep"] = ep
                else:
                    ep = pd.DataFrame(ep, columns=["ep_" + cc for cc in dims_list])
                    refined_coords = pd.concat([refined_coords, ep], axis=1)
                
                refined_coords[c_axes_list] = [s for s, a in zip(sl, coords.col_axes) if a not in dims]
                df_all.append(refined_coords)
            df_all = pd.concat(df_all, axis=0)
                
        mf = MarkerFrame(df_all.reindex(columns=list(self.axes)), 
                         columns=str(self.axes), dtype=np.float32).as_standard_type()
        mf.set_scale(self.scale)
        df = df_all[df_all.columns[df_all.columns.isin([a for a in df_all.columns if a not in dims])]]
        del self.labels
        if labels_now is not None:
            self.labels = labels_now
        return FrameDict(coords=mf, results=df)
        
    
    @dims_to_spatial_axes
    def find_sm(self, sigma:nDFloat=1.5, *, method:str="dog", cutoff:float=None, percentile:float=95, 
                topn:int=np.inf, exclude_border=True, dims=None) -> MarkerFrame:
        """
        Single molecule detection using difference of Gaussian, determinant of Hessian, Laplacian of 
        Gaussian or normalized cross correlation method.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of puncta.
        method : str {"dog", "doh", "log", "ncc"}, default is "dog"
            Which filter is used prior to finding local maxima. If "ncc", a Gaussian particle is used as
            the template image.
        cutoff : float, optional
            Cutoff value of filtered image generated by `method`.
        percentile, topn, exclude_border, dims
            Passed to peak_local_max()

        Returns
        -------
        MarkerFrame
            Peaks in uint16 type.
        
        Example
        -------
        Track single molecules and view the tracks with napari.
        >>> coords = img.find_sm()
        >>> lnk = coords.link(3, min_dwell=10)
        >>> ip.gui.add(img)
        >>> ip.gui.add(lnk)
        """        
        method = method.lower()
        if method in ("dog", "doh", "log"):
            if cutoff is None:
                cutoff = 0.0
            fil_img = getattr(self, method+"_filter")(sigma, dims=dims)
            fil_img[fil_img<cutoff] = cutoff
        elif method == "ncc":
            if cutoff is None:
                cutoff = 0.5
            sigma = np.array(check_nd(sigma, len(dims)))
            shape = tuple((sigma*4).astype(np.int))
            g = gauss.GaussianParticle([(np.array(shape)-1)/2, sigma, 1.0, 0.0])
            template = g.generate(shape)
            fil_img = self.ncc(template)
            fil_img[fil_img<cutoff] = cutoff
        else:
            raise ValueError("`method` must be 'dog', 'doh', 'log' or 'ncc'.")
        if np.isscalar(sigma):
            min_d = sigma*2
        else:
            min_d = max(sigma)*2
        coords = fil_img.peak_local_max(min_distance=min_d, percentile=percentile, 
                                        topn=topn, dims=dims, exclude_border=exclude_border)
        return coords
    
        
    @dims_to_spatial_axes
    def centroid_sm(self, coords:Coords=None, radius:nDInt=4, sigma:nDFloat=1.5, filt:Callable=None,
                    percentile:float=95, *, dims=None) -> MarkerFrame:
        """
        Calculate positions of particles in subpixel precision using centroid.

        Parameters
        ----------
        coords : MarkerFrame or (N, 2) array, optional
            Coordinates of peaks. If None, this will be determined by find_sm.
        radius : int, default is 4.
            Range to calculate centroids. Rectangular image with size 2r+1 x 2r+1 will be send 
            to calculate moments.
        sigma : float, default is 1.5
            Expected standard deviation of particles.
        filt : callable, optional
            For every slice `sl`, label is added only when filt(`input`) == True is satisfied.
        percentile, dims
            Passed to peak_local_max()
        dims : int or str, optional
            Spatial dimensions.
        
        Returns
        -------
        MarkerFrame
            Coordinates of peaks.
        """     
        if coords is None:
            coords = self.find_sm(sigma=sigma, dims=dims, percentile=percentile)
        else:
            coords = _check_coordinates(coords, self)
            
        ndim = len(dims)
        filt = check_filter_func(filt)
        radius = np.array(check_nd(radius, ndim))
        shape = self.sizesof(dims)
        with Progress("centroid_sm"):
            out = []
            columns = list(dims)
            c_axes = complement_axes(dims, coords._axes)
            for sl, crds in coords.iter(c_axes):
                centroids = []
                for center in crds.values:
                    bbox = _specify_one(center, radius, shape)
                    input_img = self.value[sl][bbox]
                    if input_img.size == 0 or not filt(input_img):
                        continue
                    
                    shift = center - radius
                    centroid = _calc_centroid(input_img, ndim) + shift
                    
                    centroids.append(centroid.tolist())
                df = pd.DataFrame(centroids, columns=columns)
                df[list(c_axes)] = sl[:-ndim]
                out.append(df)
            if len(out) == 0:
                raise ValueError("No molecule found.")
            out = pd.concat(out, axis=0)
            
            out = MarkerFrame(out.reindex(columns=list(coords._axes)),
                              columns=str(coords._axes), dtype=np.float32).as_standard_type()
            out.set_scale(coords.scale)

        return out
    
    @dims_to_spatial_axes
    def gauss_sm(self, coords:Coords=None, radius:nDInt=4, sigma:nDFloat=1.5, filt:Callable=None,
                 percentile:float=95, *, return_all:bool=False, dims=None) -> MarkerFrame|FrameDict:
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
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        MarkerFrame, if return_all == False
            Gaussian centers.
        FrameDict with keys {means, sigmas, errors}, if return_all == True
            Dictionary that contains means, standard deviations and fitting errors.
        """        
        # TODO: Whether error is correctly calculated has not been checked yet. For loop should be 
        # like centroid_sm because currently does not work for zcyx-image
        
        from scipy.linalg import pinv as pseudo_inverse
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
        with Progress("gauss_sm"):
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
            out = FrameDict(means = MarkerFrame(means, **kw).as_standard_type(),
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
    
    @record()
    def edge_grad(self, sigma:nDFloat=1.0, method:str="scharr", *, deg:bool=False, dims="yx") -> PhaseArray:
        """
        Calculate gradient direction using horizontal and vertical edge operation. Gradient direction
        is the direction with maximum gradient, i.e., intensity increase is largest. 

        Parameters
        ----------
        sigma : float, default is 1.0
            Standard deviation of Gaussian prefilter. If <= 0 then no prefilter is applied.
        method : str, {"sobel", "farid", "scharr", "prewitt"}, default is "scharr"
            Edge operator name.
        deg : bool, default is True
            If True, degree rather than radian is returned.
        dims : str, default is "yx"
            Spatial dimensions.

        Returns
        -------
        PhaseArray
            Phase image with range [-180, 180) if deg==True, otherwise [-pi, pi).
            
        Examples
        --------
        (1) Profile filament orientation distribution using histogram of edge gradient.
        >>> grad = img.edge_grad(deg=True)
        >>> plt.hist(grad.ravel(), bins=100)
        """        
        # Get operator
        method_dict = {"sobel": (skfil.sobel_h, skfil.sobel_v),
                       "farid": (skfil.farid_h, skfil.farid_v),
                       "scharr": (skfil.scharr_h, skfil.scharr_v),
                       "prewitt": (skfil.prewitt_h, skfil.prewitt_v)}
        try:
            op_h, op_v = method_dict[method]
        except KeyError:
            raise ValueError("`method` must be 'sobel', 'farid' 'scharr', or 'prewitt'.")
        
        # Start
        c_axes = complement_axes(dims, self.axes)
        if sigma > 0:
            self = self.gaussian_filter(sigma, dims=dims)
        grad_h = self.apply_dask(op_h, c_axes=c_axes)
        grad_v = self.apply_dask(op_v, c_axes=c_axes)
        grad = np.arctan2(-grad_h, grad_v)
        
        grad = PhaseArray(grad, border=(-np.pi, np.pi))
        grad.fix_border()
        deg and grad.rad2deg()
        return grad
    
    @record()
    def hessian_angle(self, sigma:nDFloat=1., *, deg:bool=False, dims="yx") -> PhaseArray:
        """
        Calculate filament angles using Hessian's eigenvectors.

        Parameters
        ----------
        sigma : float, default is 1
            Standard deviation of Gaussian filter applied before running Hessian.
        deg : bool, default is False
            If True, degree rather than radian is returned.
        dims : str, default is "yx"
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Phase image with range [-90, 90] if deg==True, otherwise [-pi/2, pi/2].
        """        
        eigval, eigvec = self.hessian_eig(sigma=sigma, dims=dims)
        arg = -np.arctan2(eigvec["r=0;l=1"], eigvec["r=1;l=1"])
        
        arg = PhaseArray(arg, border=(-np.pi/2, np.pi/2))
        arg.fix_border()
        deg and arg.rad2deg()
        return arg
    
    @record()
    def gabor_angle(self, n_sample:int=180, lmd:float=5, sigma:float=2.5, gamma:float=1, phi:float=0,
                    *, deg:bool=False, dims="yx") -> PhaseArray:
        """
        Calculate filament angles using Gabor filter. For all the candidates of angles, Gabor response is
        calculated, and the strongest response is returned as output array.

        Parameters
        ----------
        n_sample : int, default is 180
            Number of `theta`s to calculate. By default, -90, -89, ..., 89 degree are calculated.
        lmd : float, default is 5
            Wave length of Gabor kernel. Make sure that the diameter of the objects you want to detect is
            around `lmd/2`.
        sigma : float, default is 2.5
            Standard deviation of Gaussian factor of Gabor kernel.
        gamma : float, default is 1
            Anisotropy of Gabor kernel, i.e. the standard deviation orthogonal to theta will be sigma/gamma.
        phi : float, by default 0
            Phase offset of harmonic factor of Gabor kernel.
        deg : bool, default is False
            If True, degree rather than radian is returned.
        dims : str, default is "yx"
            Spatial axes.
            
        Returns
        -------
        ImgArray
            Phase image with range [-90, 90) if deg==True, otherwise [-pi/2, pi/2).
        """        
        thetas = np.linspace(0, np.pi, n_sample, False)
        max_ = np.empty(self.shape, dtype=np.float32)
        argmax_ = np.zeros(self.shape, dtype=np.float32) # This is float32 because finally this becomes angle array.
        
        c_axes = complement_axes(dims, self.axes)
        for i, theta in enumerate(thetas):
            ker = skfil.gabor_kernel(1/lmd, theta, 0, sigma, sigma/gamma, 3, phi).astype(np.complex64)
            out_ = self.as_float().apply_dask(ndi.convolve, 
                                              c_axes=c_axes, 
                                              args=(ker.real,)
                                              )
            if i > 0:
                where_update = out_ > max_
                max_[where_update] = out_[where_update]
                argmax_[where_update] = i
            else:
                max_ = out_
        argmax_ *= (thetas[1] - thetas[0])
        argmax_[:] = np.pi/2 - argmax_
        
        argmax_ = PhaseArray(argmax_, border=(-np.pi/2, np.pi/2))
        argmax_.fix_border()
        deg and argmax_.rad2deg()
        return argmax_
    
    @record()
    def gabor_filter(self, lmd:float=5, theta:float=0, sigma:float=2.5, gamma:float=1, phi:float=0, 
                     *, return_imag:bool=False, dims="yx") -> ImgArray:
        """
        Make a Gabor kernel and convolve it.

        Parameters
        ----------
        lmd : float, default is 5
            Wave length of Gabor kernel. Make sure that the diameter of the objects you want to detect is
            around `lmd/2`.
        theta : float, default is 0
            Orientation of harmonic factor of Gabor kernel in radian (x-directional if `theta==0`).
        sigma : float, default is 2.5
            Standard deviation of Gaussian factor of Gabor kernel.
        gamma : float, default is 1
            Anisotropy of Gabor kernel, i.e. the standard deviation orthogonal to theta will be sigma/gamma.
        phi : float, default is 0
            Phase offset of harmonic factor of Gabor kernel.
        return_imag : bool, default is False
            If True, a complex image that contains both real and imaginary part of Gabor response is returned.
        dims : str, default is "yx"
            Spatial dimensions.

        Returns
        -------
        ImgArray (dtype is float32 or complex64)
            Filtered image.
        
        Example
        -------
        Edge Detection using multi-angle Gabor filtering.
        >>> thetas = np.deg2rad([0, 45, 90, 135])
        >>> out = np.zeros((4,)+img.shape, dtype=np.float32)
        >>> for i, theta in enumerate(thetas):
        >>>     out[i] = img.gabor_filter(theta=theta)
        >>> out = np.max(out, axis=0)
        """        
        # TODO: 3D Gabor filter
        ker = skfil.gabor_kernel(1/lmd, theta, 0, sigma, sigma/gamma, 3, phi).astype(np.complex64)
        if return_imag:
            out = self.as_float().apply_dask(_filters.gabor_filter, 
                                             c_axes=complement_axes(dims, self.axes), 
                                             args=(ker,), 
                                             dtype=np.complex64
                                             )
        else:
            out = self.as_float().apply_dask(ndi.convolve, 
                                             c_axes=complement_axes(dims, self.axes),
                                             args=(ker.real,), 
                                             dtype=np.float32
                                             )
        return out
    
    @record()
    def optimal_path(self, src, dst) -> ImgArray:
        # TODO: conbine with napari
        src = np.round(src).astype(np.uint16)
        dst = np.round(dst).astype(np.uint16)
        ind, cost = skgraph.route_through_array(self.value, np.round(src), dst, geometric=False)
        ind = np.array(ind)
        route = np.zeros(self.shape, dtype=bool)
        route[tuple(ind[:,i] for i in range(ind.shape[1]))] = True
        route = route.view(self.__class__)
        route.temp = cost
        return route
    
    @dims_to_spatial_axes
    @record()
    def fft(self, *, dims=None) -> ImgArray:
        """
        Fast Fourier transformation. This function returns complex array. Inconpatible with 
        some ImgArray functions.
        
        Parameters
        ----------
        dims : int or str, optional
            Spatial dimensions.
            
        Returns
        -------
        ImgArray
            FFT image.
        """
        freq = fft(self.value.astype(np.float32), axes=[self.axisof(a) for a in dims])
        freq[:] = np.fft.fftshift(freq)
        return freq
    
    @dims_to_spatial_axes
    @record()
    def ifft(self, real:bool=True, *, dims=None) -> ImgArray:
        """
        Fast Inverse Fourier transformation. Complementary function with `fft()`.
        
        Parameters
        ----------
        real : bool, default is True
            If True, only the real part is returned.
        dims : int or str, optional
            Spatial dimensions.
            
        Returns
        -------
        ImgArray
            IFFT image.
        """
        freq = np.fft.ifftshift(self.value)
        out = ifft(freq, axes=[self.axisof(a) for a in dims])
        
        if real:
            out = np.real(out)
        return out
    
    @dims_to_spatial_axes
    @record()
    def power_spectra(self, norm:bool=False, *, dims=None) -> ImgArray:
        """
        Return n-D power spectra of images, which is defined as:
            P = Re{F[img]}^2 + Im{F[img]}^2

        Parameters
        ----------
        norm : bool, default is False
            If True, maximum value of power spectra is adjusted to 1.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Power spectra
        """        
        freq = self.fft(dims=dims)
        pw = freq.real**2 + freq.imag**2
        if norm:
            pw /= pw.max()
        return pw
    
    @record()
    def threshold(self, thr:float|str="otsu", *, dims=None, **kwargs) -> ImgArray:
        """
        Parameters
        ----------
        thr: float or array or None, optional
            Threshold value, or thresholding algorithm.
        dims : int or str, default is all the axes except for channel axis.
            Dimensions that will share the same threshold.
        **kwargs:
            Keyword arguments that will passed to function indicated in 'method'.

        Returns
        -------
        ImgArray
            Boolian array.
        
        Example
        -------
        Substitute outliers to 0.
        >>> thr = img.threshold("99%")
        >>> img[thr] = 0
        """
        if dims is None:
            dims = complement_axes("c", self.axes)
            
        methods_ = {"isodata": skfil.threshold_isodata,
                    "li": skfil.threshold_li,
                    "local": skfil.threshold_local,
                    "mean": skfil.threshold_mean,
                    "min": skfil.threshold_minimum,
                    "minimum": skfil.threshold_minimum,
                    "niblack": skfil.threshold_niblack,
                    "otsu": skfil.threshold_otsu,
                    "sauvola": skfil.threshold_sauvola,
                    "triangle": skfil.threshold_triangle,
                    "yen": skfil.threshold_yen
                    }
        
        if isinstance(thr, str) and thr.endswith("%"):
            p = float(thr[:-1])
            out = np.zeros(self.shape, dtype=bool)
            for sl, img in self.iter(complement_axes(dims, self.axes)):
                thr = np.percentile(img, p)
                out[sl] = img >= thr
                
        elif isinstance(thr, str):
            method = thr.lower()
            try:
                func = methods_[method]
            except KeyError:
                s = ", ".join(list(methods_.keys()))
                raise KeyError(f"{method}\nmethod must be: {s}")
            
            out = np.zeros(self.shape, dtype=bool)
            for sl, img in self.iter(complement_axes(dims, self.axes)):
                thr = func(img, **kwargs)
                out[sl] = img >= thr
            

        elif np.isscalar(thr):
            out = self >= thr
        else:
            raise TypeError("'thr' must be numeric, or str specifying a thresholding method.")                
        
        return out
    
    @record(append_history=False)
    def label_multiotsu(self, classes:int=3, nbins:int=256, *, dims:str=None) -> ImgArray:
        """
        Label images using multi-Otsu method. Region lower than the lowest threshold will be labeled
        zero. This function will take very long time with large `classes` value.

        Parameters
        ----------
        classes : int, default is 3
            Number of classes input images will be classified. The result label will have values 0, 1,
            ..., classes-1.
        nbins : int, default is 256
            Number of bins.
        dims : str, optional
            Dimensions that will share the same thresholding results.

        Returns
        -------
        ImgArray
            Labeled image.
        """        
        if self.dtype != np.uint8:
            raise ValueError("dtypes other than uint8 seem not working.")
        if dims is None:
            dims = complement_axes("c", self.axes)
        labels = np.zeros(self.shape, dtype=np.uint8)
        for sl, img in self.iter(complement_axes(dims, self.axes)):
            thr = skfil.threshold_multiotsu(img, classes=classes, nbins=nbins)
            labels[sl] = np.digitize(img, bins=thr)
        
        self.labels = labels.view(Label)
        self.labels._set_info(self, "label_multiotsu")
        self.labels.set_scale(self)
        return self
    
    @dims_to_spatial_axes
    @record(only_binary=True)
    def distance_map(self, *, dims=None) -> ImgArray:
        """
        Calculate distance map from binary images.

        Parameters
        ----------
        dims : int or str, optional
            spatial dimensions.

        Returns
        -------
        ImgArray
            Distance map, the further the brighter
        """        
        return self.apply_dask(ndi.distance_transform_edt, 
                               c_axes=complement_axes(dims, self.axes)
                               )
    
    @record()
    def ncc(self, template:np.ndarray, bg:float=None) -> ImgArray:
        """
        Template matching using normalized cross correlation (NCC) method. This function is basically
        identical to that in `skimage.feature`, but is optimized for batch processing and improved 
        readability.

        Parameters
        ----------
        template : np.ndarray
            Template image. Must be 2 or 3 dimensional. 
        bg : float, optional
            Background intensity. If not given, it will calculated as the minimum value of 
            the original image.

        Returns
        -------
        ImgArray
            Response image with values between -1 and 1.
        """        
        template = _check_template(template)
        bg = _check_bg(self, bg)
        dims = "yx" if template.ndim == 2 else "zyx"
        return self.as_float().apply_dask(_misc.ncc,
                                          c_axes=complement_axes(dims, self.axes), 
                                          args=(template, bg)
                                          )
    
    @record(append_history=False)
    def track_template(self, template:np.ndarray, bg=None, along:str="t") -> MarkerFrame:
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
            
        pos = np.array(pos)
        pos = np.hstack([np.arange(self.sizeof(along), dtype=np.uint16).reshape(-1,1), pos])
        pos = MarkerFrame(pos, columns=along+dims)
        
        return pos
    
    @dims_to_spatial_axes
    @record(only_binary=True)
    def remove_large_objects(self, radius:float=5, *, dims=None, update:bool=False) -> ImgArray:
        """
        Remove large objects using opening. Those objects that were not removed by opening
        will be removed in output.

        Parameters
        ----------
        radius : float, optional
            Objects with radius larger than this value will be removed.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to output.

        Returns
        -------
        ImgArray
            Image with large objects removed.
        """        
        out = self.copy()
        large_obj = self.opening(radius, dims=dims)
        out.value[large_obj] = 0
            
        return out
    
    @dims_to_spatial_axes
    @record(only_binary=True)
    def remove_fine_objects(self, length:float=10, *, dims=None, update:bool=False) -> ImgArray:
        """
        Remove fine objects using diameter_opening.

        Parameters
        ----------
        length : float, default is 10
            Objects longer than this will be removed.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to output.

        Returns
        -------
        ImgArray
            Image with large objects removed.
        """        
        out = self.copy()
        fine_obj = self.diameter_opening(length, connectivity=len(dims))
        large_obj = self.opening(length//2)
        out.value[~large_obj & fine_obj] = 0
            
        return out
    
    @dims_to_spatial_axes
    @record(only_binary=True)
    def convex_hull(self, *, dims=None, update=False) -> ImgArray:
        """
        Compute convex hull image.

        Parameters
        ----------
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self.

        Returns
        -------
        ImgArray
            Convex hull image.
        """        
        return self.apply_dask(skimage.morphology.convex_hull_image, 
                               c_axes=complement_axes(dims, self.axes), 
                               dtype=bool
                               ).astype(bool)
        
    @dims_to_spatial_axes
    @record(only_binary=True)
    def skeletonize(self, radius:float=0, *, dims=None, update=False) -> ImgArray:
        """
        Skeletonize images. Only works for binary images.

        Parameters
        ----------
        radius : float, optional
            Radius of skeleton. This is achieved simply by dilation of skeletonized results.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self.

        Returns
        -------
        ImgArray
            Skeletonized image.
        """        
        if radius >= 1:
            selem = ball_like(radius, len(dims))
        else:
            selem = None
        
        return self.apply_dask(_filters.skeletonize, 
                               c_axes=complement_axes(dims, self.axes),
                               args=(selem,),
                               dtype=bool
                               ).astype(bool)
        
    @dims_to_spatial_axes
    @record(only_binary=True)
    def count_neighbors(self, *, connectivity:int=None, mask:bool=True, dims=None) -> ImgArray:
        """
        Count the number or neighbors of binary images. This function can be used for cross section
        or branch detection. Only works for binary images.

        Parameters
        ----------
        connectivity : int, optional
            See label().
        mask : bool,　default is True
            If True, only neighbors of pixels that satisfy self==True is returned.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            uint8 array of the number of neighbors.
            
        Example
        -------
        >>> skl = img.threshold().skeletonize()
        >>> edge = skl.count_neighbors()
        >>> np.argwhere(edge == 1) # get coordinates of filament edges.
        >>> np.argwhere(edge >= 3) # get coordinates of filament cross sections.
        
        """        
        ndim = len(dims)
        connectivity = ndim if connectivity is None else connectivity
        selem = ndi.morphology.generate_binary_structure(ndim, connectivity)
        selem[(1,)*ndim] = 0
        out = self.as_uint8().apply_dask(_filters.population, 
                                         c_axes=complement_axes(dims, self.axes), 
                                         args=(selem,))
        if mask:
            out[~self.value] = 0
            
        return out.astype(np.uint8)
    
    @dims_to_spatial_axes
    @record(only_binary=True)
    def remove_skeleton_structure(self, structure:str="tip", *, connectivity:int=None,
                                  dims=None, update:bool=False) -> ImgArray:
        """
        Remove certain structure from skeletonized images.

        Parameters
        ----------
        structure : str, default is "tip"
            What type of structure to remove.
        connectivity : int, optional
            See label().
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default if False
            If update self.

        Returns
        -------
        ImgArray
            Processed image.
        """        
        out = self.copy()
        neighbor = self.count_neighbors(connectivity=connectivity, dims=dims)
        if structure == "tip":
            sl = neighbor == 1
        elif structure == "branch":
            sl = neighbor > 2
        elif structure == "cross":
            sl = neighbor > 3
        else:
            raise ValueError("`mode` must be one of {'tip', 'branch', 'cross'}.")
        out.value[sl] = 0
        return out
    
    @record(append_history=False)
    def pointprops(self, coords:Coords, *, order:int=1, squeeze:bool=True) -> PropArray:
        """
        Measure interpolated intensity at points with float coordinates.

        Parameters
        ----------
        coords : MarkerFrame or array-like
            Coordinates of point to be measured.
        order : int, default is 1
            Spline interpolation order.
        squeeze : bool, default is True
            If True and only one point is measured, the redundant dimension ID_AXIS will be deleted.

        Returns
        -------
        PropArray
            Point properties.
        
        Example
        -------
        Calculate centroids and measure intensities.
        >>> coords = img.proj("t").centroid_sm()
        >>> prop = img.pointprops(coords)
        """        
        coords = _check_coordinates(coords, self)
        col_axes = coords.col_axes
        prop_axes = complement_axes(coords.col_axes, self.axes)
        coords = np.asarray(coords, dtype=np.float32).T
        shape = self.sizesof(prop_axes)
        l = coords.shape[1] # Number of points
        out = PropArray(np.empty((l,)+shape, dtype=np.float32), name=self.name, 
                        axes=Const["ID_AXIS"]+prop_axes, dirpath=self.dirpath,
                        propname = f"pointprops", dtype=np.float32)
        
        for sl, img in self.iter(prop_axes, exclude=col_axes):
            out[(slice(None),)+sl] = ndi.map_coordinates(img, coords, prefilter=order > 1,
                                                         order=order, mode="reflect")
        if l == 1 and squeeze:
            out = out[0]
        return out
    
    @record(append_history=False)
    def lineprops(self, src:Coords, dst:Coords, func:str|Callable="mean", *, order:int=1, 
                  squeeze:bool=True) -> PropArray:
        """
        Measure line property using func(line_scan).

        Parameters
        ----------
        src : MarkerFrame or array-like
            Source coordinates.
        dst : MarkerFrame of array-like
            Destination coordinates.
        func : str or callable, default is "mean".
            Measurement function.
        order : int, optional
            Spline interpolation order.
        squeeze : bool, default is True.
            If True and only one line is measured, the redundant dimension ID_AXIS will be deleted.

        Returns
        -------
        PropArray
            Line properties.

        Example
        -------
        Time-course measurement of intensities on lines.
        >>> pr = img.lineprops([[2,3], [8,9]], [[32,85], [66,73]])
        >>> pr.plot()
        """        
        func = _check_function(func)
        src = _check_coordinates(src, self)
        dst = _check_coordinates(dst, self)
        
        if src.shape != dst.shape:
            raise ValueError(f"Shape mismatch between `src` and `dst`: {src.shape} and {dst.shape}")
        
        l = src.shape[0]
        prop_axes = complement_axes(src.col_axes, self.axes)
        shape = self.sizesof(prop_axes)
        
        out = PropArray(np.empty((l,)+shape, dtype=np.float32), name=self.name, 
                        axes=Const["ID_AXIS"]+prop_axes, dirpath=self.dirpath,
                        propname = f"lineprops<{func.__name__}>", dtype=np.float32)
        
        for i, (s, d) in enumerate(zip(src.values, dst.values)):
            resliced = self.reslice(s, d, order=order)
            out[i] = np.apply_along_axis(func, axis=-1, arr=resliced.value)
        
        if l == 1 and squeeze:
            out = out[0]
        
        return out
    
    
    @dims_to_spatial_axes
    @record(need_labels=True)
    def watershed(self, coords:MarkerFrame=None, *, connectivity:int=1, input:str="distance", 
                  min_distance:float=2, dims=None) -> Label:
        """
        Label segmentation using watershed algorithm.

        Parameters
        ----------
        coords : MarkerFrame, optional
            Returned by such as `peak_local_max()`. Array of coordinates of peaks.
        connectivity : int, optional
            Passed to skimage.segmentation.watershed.
        input_ : str, optional
            What image will be the input of watershed algorithm.            
            - "self" ... self is used.
            - "distance" ... distance map of self.labels is used.
        dims : str, optional
            Spatial dimension.
            
        Returns
        -------
        Label
            Updated labels.
        """
        
        # Prepare the input image.
        if input == "self":
            input_img = self.copy()
        elif input == "distance":
            input_img = self.__class__(self.labels>0, axes=self.axes).distance_map(dims=dims)
        else:
            raise ValueError("'input_' must be either 'self' or 'distance'.")
                
        if input_img.dtype == bool:
            input_img = input_img.astype(np.uint8)
            
        input_img._view_labels(self)
        
        if coords is None:
            coords = input_img.peak_local_max(min_distance=min_distance, dims=dims)
        
        labels = largest_zeros(input_img.shape)
        shape = self.sizesof(dims)
        n_labels = 0
        c_axes = complement_axes(dims, self.axes)
        markers = np.zeros(shape, dtype=labels.dtype) # placeholder for maxima
        for sl, crd in coords.iter(c_axes):
            # crd.values is (N, 2) array so tuple(crd.values.T.tolist()) is two (N,) list.
            crd = crd.values.T.tolist()
            markers[tuple(crd)] = np.arange(1, len(crd[0])+1, dtype=labels.dtype)
            labels[sl] = skseg.watershed(-input_img.value[sl], markers, 
                                        mask=input_img.labels.value[sl], 
                                        connectivity=connectivity)
            labels[sl][labels[sl]>0] += n_labels
            n_labels = labels[sl].max()
            markers[:] = 0 # reset placeholder
        
        labels = labels.view(Label)
        self.labels = labels.optimize()
        self.labels._set_info(self)
        self.labels.set_scale(self)
        return self.labels
    
    @dims_to_spatial_axes
    @record(append_history=False, need_labels=True)
    def random_walker(self, beta:float=130, mode:str="cg_j", tol:float=1e-3, *, dims=None) -> Label:
        """
        Random walker segmentation. Only wrapped skimage segmentation. `self.labels` will be
        segmented.

        Parameters
        ----------
        beta, mode, tol
            see skimage.segmentation.random_walker
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        ImgArray
            Relabeled image.
        """        
        c_axes = complement_axes(dims, self.axes)
        
        for sl, img in self.iter(c_axes, israw=True):
            img.labels[:] = skseg.random_walker(img.value, img.labels.value, beta=beta, mode=mode, tol=tol)
            
        self.labels._set_info(self, "random_walker")
        return self.labels
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def slic(self, n_segments=100, *, compactness=10.0, max_iter=10, sigma=1, multichannel=False,
             min_size_factor=0.5, max_size_factor=3, mask=None, dims=None):
        # multichannel not working, needs sort_axes
        # BUG: slic returns a strange label with grayscale images.
        if multichannel:
            c_axes = complement_axes("c"+dims, self.axes)
            labels = largest_zeros(self["c=0"].shape)
            exclude = "c"
        else:
            c_axes = complement_axes(dims, self.axes)
            labels = largest_zeros(self.shape)
            exclude = ""
        
        for sl, img in self.iter(c_axes, exclude=exclude):
            plt.imshow(img)
            labels[sl] = \
            skseg.slic(img, n_segments=n_segments, compactness=compactness, max_iter=max_iter,
                       sigma=sigma, multichannel=multichannel, min_size_factor=min_size_factor,
                       max_size_factor=max_size_factor, start_label=1, mask=mask)
        
        self.labels = labels.view(Label).optimize()
        self.labels._set_info(self, "slic")
        self.labels.set_scale(self)
        return self
    
    def label_threshold(self, thr:float|str="otsu", *, dims=None, **kwargs) -> Label:
        """
        Make labels with threshold(). Be sure that keyword argument `dims` can be
        different (in most cases for >4D images) between threshold() and label().
        In this function, both function will have the same `dims` for simplicity.

        Parameters
        ----------
        All are passed to self.threshold()
        
        Returns
        -------
        ImgArray
            Same array but labels are updated.
        """        
        labels = self.threshold(thr=thr, dims=None, **kwargs)
        return self.label(labels, dims=dims)
    
    
    @record(append_history=False)
    def pathprops(self, paths:PathFrame, properties:str|Callable|Iterable[str|Callable]="mean", *, 
                  order:int=1) -> ArrayDict:
        """
        Measure line property using func(line_scan) for each func in properties.

        Parameters
        ----------
        paths : PathFrame
            Paths to measure properties.
        properties : str or callable, or their iterable
            Properties to be analyzed.
        order : int, optional
            Spline interpolation order.
        
        Returns
        -------
        ArrayDict of PropArray
            Line properties. Keys are property names and values are the corresponding PropArrays.

        Example
        -------
        (1) Time-course measurement of intensities on a path.
        >>> img.pathprops([[2,3], [102, 301], [200,400]])
        """        
        id_axis = Const["ID_AXIS"]
        # check path
        if not isinstance(paths, PathFrame):
            paths = np.asarray(paths)
            paths = np.hstack([np.zeros((paths.shape[0],1)), paths])
            paths = PathFrame(paths, columns=id_axis+str(self.axes)[-paths.shape[1]+1:])
            
        # make a function dictionary
        funcdict = dict()
        if isinstance(properties, str) or callable(properties):
            properties = (properties,)
        for f in properties:
            if isinstance(f, str):
                funcdict[f] = getattr(np, f)
            elif callable(f):
                funcdict[f.__name__] = f
            else:
                raise TypeError(f"Cannot interpret property {f}")
        
        # prepare output
        l = len(paths[id_axis].unique())
        prop_axes = complement_axes(paths._axes, id_axis + str(self.axes))
        shape = self.sizesof(prop_axes)
        
        out = ArrayDict({k: PropArray(np.empty((l,)+shape, dtype=np.float32), name=self.name, 
                                      axes=id_axis+prop_axes, dirpath=self.dirpath,
                                      propname = f"lineprops<{k}>", dtype=np.float32)
                         for k in funcdict.keys()}
                        )
        
        for i, path in enumerate(paths.split(id_axis)):
            resliced = self.reslice(path, order=order)
            for name, func in funcdict.items():
                out[name][i] = np.apply_along_axis(func, axis=-1, arr=resliced.value)
                
        return out
    
    @record(append_history=False, need_labels=True)
    def regionprops(self, properties:tuple[str,...]|str=("mean_intensity",), *, 
                    extra_properties=None) -> ArrayDict:
        """
        Run skimage's regionprops() function and return the results as PropArray, so
        that you can access using flexible slicing. For example, if a tcyx-image is
        analyzed with properties=("X", "Y"), then you can get X's time-course profile
        of channel 1 at label 3 by prop["X"]["p=5;c=1"] or prop.X["p=5;c=1"].

        Parameters
        ----------
        properties : iterable, optional
            properties to analyze, see skimage.measure.regionprops.
        extra_properties : iterable of callable, optional
            extra properties to analyze, see skimage.measure.regionprops.

        Returns
        -------
        ArrayDict of PropArray
            Dictionary has keys of properties that are specified by `properties`. Each value
            has the array of properties.
            
        Example
        -------
        Measure region properties around single molecules.
        >>> coords = img.centroid_sm()
        >>> img.specify(coords, 3, labeltype="circle")
        >>> props = img.regionprops()
        """        
        id_axis = Const["ID_AXIS"]
        if isinstance(properties, str):
            properties = (properties,)
        if extra_properties is not None:
            properties = properties + tuple(ex.__name__ for ex in extra_properties)

        if id_axis in self.axes:
            # this dimension will be label
            raise ValueError(f"axis '{id_axis}' is used for label ID in DataFrames.")
        
        prop_axes = complement_axes(self.labels.axes, self.axes)
        shape = self.sizesof(prop_axes)
        
        out = ArrayDict({p: PropArray(np.empty((self.labels.max(),) + shape, dtype=np.float32),
                                      name=self.name, 
                                      axes=id_axis+prop_axes,
                                      dirpath=self.dirpath,
                                      propname=p)
                         for p in properties})
        
        # calculate property value for each slice
        for sl, img in self.iter(prop_axes, exclude=self.labels.axes):
            props = skmes.regionprops(self.labels.value, img, cache=False,
                                      extra_properties=extra_properties)
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                # Both sides have length of p-axis (number of labels) so that values
                # can be correctly substituted.
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]
        
        for parr in out.values():
            parr.set_scale(self)
        return out
    
    @dims_to_spatial_axes
    @record()
    def lbp(self, p:int=12, radius:int=1, *, method:str="default", dims=None) -> ImgArray:
        """
        Local binary pattern feature extraction.

        Parameters
        ----------
        p : int, default is 12
            Number of circular neighbors
        radius : int, default is 1
            Radius of neighbours.
        method : str, optional
            Method to determined the pattern.
        dims : str or int, optional
            Spatial dimension.

        Returns
        -------
        ImgArray
            Local binary pattern image.
        """        
        
        return self.apply_dask(skfeat.local_binary_pattern,
                               c_axes=complement_axes(dims), 
                               args=(p, radius, method)
                               )
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def glcm(self, distances, angles, *, bins:int=None, rescale_max:bool=False, dims=None) -> ImgArray:
        """
        Compute "Gray Level Coocurrence Matrix". This matrix is used for texture classification. For the
        visualization and projection purpose, ImgArray (instead of PropArray) is returned but be aware
        that returned image does not have yx-axes.

        Parameters
        ----------
        distances : array_like
            List of pixel pair distance offsets.
        angles : array_like
            List of pixel pair angles in radians.
        bins : int, optional
            Number of bins.
        rescale_max : bool, default is False
            If True, the contrast of the input image is maximized by multiplying an integer.
        dims : str or int, optional
            Spatial dimension.

        Returns
        -------
        ImgArray
            GLCM with additional axes "ijd<", where "i" and "j" means intensity value, "d" means
            distance and "<" means angle.
        """        
        # BUG: not working
        self, bins, rescale_max = check_glcm(self, bins, rescale_max)
            
        c_axes = complement_axes(dims, self.axes)
        out = self.apply_dask(skfeat.greycomatrix, 
                              c_axes=c_axes,
                              new_axis=[-4,-3,-2,-1],
                              drop_axis=dims,
                              args=(distances, angles),
                              kwargs=dict(levels=bins),
                              dtype=np.uint32
                              )
        out._set_info(self, "glcm", new_axes=c_axes+"ijd<")
        
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def glcm_props(self, distances, angles, radius:int, properties:tuple=None, 
                   *, bins:int=None, rescale_max:bool=False, dims=None) -> ImgArray:
        """
        Compute properties of "Gray Level Coocurrence Matrix". This will take long time
        because of pure Python for-loop.

        Parameters
        ----------
        distances : array_like
            List of pixel pair distance offsets.
        angles : array_like
            List of pixel pair angles in radians.
        radius : int
            Window radius.
        properties : tuple of str
            contrast, dissimilarity, homogeneity, energy, mean, std, asm, max, entropy
        bins : int, optional
            Number of bins.
        rescale_max : bool, default is False
            If True, the contrast of the input image is maximized by multiplying an integer.
        dims : str or int, optional
            Spatial dimensions.

        Returns
        -------
        ArrayDict of ImgArray
            GLCM with additional axes "d<", where "d" means distance and "<" means angle.
            If input image has "tzyx" axes then output will have "tzd<yx" axes.
        
        Example
        -------
        Plot GLCM's IDM and ASM images
        >>> out = img.glcm_props([1], [0], 3, properties=("idm","asm"))
        >>> out.idm["d=0;<=0"].imshow()
        >>> out.asm["d=0;<=0"].imshow()
        """        
        self, bins, rescale_max = check_glcm(self, bins, rescale_max)
        if properties is None:
            properties = ("contrast", "dissimilarity", "idm", 
                          "asm", "max", "entropy", "correlation")
        c_axes = complement_axes(dims, self.axes)
        distances = np.asarray(distances, dtype=np.uint8)
        angles = np.asarray(angles, dtype=np.float32)
        outshape = self.sizesof(c_axes) + (len(distances), len(angles)) + self.sizesof(dims)
        out = {}
        for prop in properties:
            if isinstance(prop, str):
                out[prop] = np.empty(outshape, dtype=np.float32).view(self.__class__)
            elif callable(prop):
                out[prop.__name__] = np.empty(outshape, dtype=np.float32).view(self.__class__)
            else:
                raise TypeError("properties must be str or callable.")
        out = ArrayDict(out)
        self = self.pad(radius, mode="reflect", dims=dims)
        self.history.pop()
        for sl, img in self.iter(c_axes):
            propout = glcm_props_(img, distances, angles, bins, radius, properties)
            for prop in properties:
                out[prop].value[sl] = propout[prop]
            
        for k, v in out.items():
            v._set_info(self, f"glcm_props-{k}", new_axes=c_axes+"d<"+dims)
        return out
    
    
    @same_dtype()
    @record(append_history=False)
    def proj(self, axis:str=None, method:str|Callable="mean", mask=None, **kwargs) -> ImgArray:
        """
        Z-projection along any axis.

        Parameters
        ----------
        axis : str, optional
            Along which axis projection will be calculated. If None, most plausible one will be chosen.
        method : str or callable, default is mean-projection.
            Projection method. If str is given, it will converted to numpy function.
        mask : array-like, optional
            If provided, input image will be converted to np.ma.array and `method` will also be interpreted
            as an masked functio if possible.
        **kwargs
            Other keyword arguments that will passed to projection function.

        Returns
        -------
        ImgArray
            Projected image.
        """        
        # determine function
        if mask is not None:
            if isinstance(method, str):
                func = getattr(np.ma, method)
            else:
                func = method
        else:
            func = _check_function(method)
        
        if axis is None:
            axis = find_first_appeared("ztpi<c", include=self.axes, exclude="yx")
        elif not isinstance(axis, str):
            raise TypeError("`axis` must be str.")
        axisint = tuple(self.axisof(a) for a in axis)
        if func.__module__ == "numpy.ma.core":
            arr = np.ma.array(self.value, mask=mask, dtype=self.dtype)
            if func is np.ma.mean:
                out = func(arr, axis=axisint, dtype=np.float32, **kwargs)
            else:
                out = func(arr, axis=axisint, **kwargs)
        elif func is np.mean:
            out = func(self.value, axis=axisint, dtype=np.float32, **kwargs)
        else:
            out = func(self.value, axis=axisint, **kwargs)
        
        out = out.view(self.__class__)
        out._set_info(self, f"proj(axis={axis}, method={method})", del_axis(self.axes, axisint))
        return out

    @record()
    def clip(self, in_range:tuple[int|str, int|str]=("0%", "100%")) -> ImgArray:
        """
        Saturate low/high intensity using np.clip().

        Parameters
        ----------
        in_range : two scalar values, optional
            range of lower/upper limits, by default (0, 100)

        Returns
        -------
        ImgArray
            Clipped image with temporal attribute
        """        
        lowerlim, upperlim = _check_clip_range(in_range, self.value)
        out = np.clip(self.value, lowerlim, upperlim)
        out = out.view(self.__class__)
        out.temp = [lowerlim, upperlim]
        return out
    
    @record()
    def rescale_intensity(self, in_range:tuple[int|str, int|str]=("0%", "100%"), dtype=np.uint16) -> ImgArray:
        """
        Rescale the intensity of the image using skimage.exposure.rescale_intensity().

        Parameters
        ----------
        in_range : two scalar values, default is (0%, 100%)
            Range of lower/upper limit.
        dtype : numpy dtype, default is np.uint16
            Output dtype.

        Returns
        -------
        ImgArray
            Rescaled image with temporal attribute
        """        
        out = self.view(np.ndarray).astype(np.float32)
        lowerlim, upperlim = _check_clip_range(in_range, self.value)
            
        out = skexp.rescale_intensity(out, in_range=(lowerlim, upperlim), out_range=dtype)
        
        out = out.view(self.__class__)
        out.temp = [lowerlim, upperlim]
        return out
    
    @record(append_history=False)
    def track_drift(self, along:str=None, show_drift:bool=False, **kwargs) -> MarkerFrame:
        """
        Calculate xy-directional drift using `skimage.registration.phase_cross_correlation`.

        Parameters
        ----------
        along : str, optional
            Along which axis drift will be calculated.
        show_drift : bool, default is False
            If True, plot the result.

        Returns
        -------
        MarkerFrame
            DataFrame structure with x,y columns
        """        
        if along is None:
            along = find_first_appeared("tpzc", include=self.axes)
        elif len(along) != 1:
            raise ValueError("`along` must be single character.")
            
        if self.ndim != 3:
            raise TypeError(f"Input must be three dimensional, but got {self.shape}.")

        # slow drift needs large upsampling numbers
        corr_kwargs = {"upsample_factor": 10}
        corr_kwargs.update(kwargs)
        
        result = [[0.0, 0.0]]
        last_img = None
        for _, img in self.iter(along):
            if last_img is not None:
                shift = skreg.phase_cross_correlation(last_img, img, return_error=False, **corr_kwargs)
                shift_total = shift + result[-1]    # list + ndarray -> ndarray
                result.append(shift_total)
                last_img = img
            else:
                last_img = img
        
        result = MarkerFrame(np.array(result), columns="yx")
        
        show_drift and plot_drift(result)
        result.index.name = along
        return result
    
    @record()
    @same_dtype(asfloat=True)
    def drift_correction(self, shift:Coords=None, ref:ImgArray=None, *, order:int=1, 
                         along:str=None, dims="yx", update:bool=False) -> ImgArray:
        """
        Drift correction using iterative Affine translation. If translation vectors `shift`
        is not given, then it will be determined using `track_drift` method of ImgArray.

        Parameters
        ----------
        shift : DataFrame with columns "x" and "y" (MarkerFrame recommended) or (N, 2) array, optional
            Translation vectors
        ref : ImgArray, optional
            The reference 3D image to determine drift, if `shift` was not given.
        order : int, default is 1
            The order of interpolation.
        along : str, optional
            Along which axis drift will be corrected.
        dims : str, default is "yx"
            Spatial dimension.
        update : bool, optional
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Corrected image.
            
        Example
        -------
        Drift correction of multichannel image using the first channel as the reference.
        >>> img.drift_correction(ref=img["c=0"])
        """        
        
        if along is None:
            along = find_first_appeared("tpzc", include=self.axes, exclude=dims)
        elif len(along) != 1:
            raise ValueError("`along` must be single character.")
        
        if len(dims) == 3:
            raise NotImplementedError("3-dimensional correction is not implemented. yet")
        
        if shift is None:
            # determine 'ref'
            if ref is None:
                ref = self
            elif not isinstance(ref, self.__class__):
                raise TypeError(f"'ref' must be ImgArray object, but got {type(ref)}")
            elif ref.axes != along + dims:
                raise ValueError(f"Arguments `along`({along}) + `dims`({dims}) do not match "
                                 f"axes of `ref`({ref.axes})")

            shift = ref.track_drift(along=along)
        elif isinstance(shift, MarkerFrame):
            if len(shift) != self.sizeof("t"):
                raise ValueError("Wrong shape of 'shift'.")
        else:
            shift = MarkerFrame(shift, columns="yx", dtype=np.float32)

        out = np.empty(self.shape)
        t_index = self.axisof(along)
        shift = shift.reindex(columns=["x", "y"])
        for sl, img in self.iter(complement_axes(dims, self.axes)):
            out[sl] = _translate_image(img, shift.loc[sl[t_index]], order=order)
        
        out = out.view(self.__class__)
        return out

    @dims_to_spatial_axes
    @record(append_history=False)
    def estimate_sigma(self, *, squeeze:bool=True, dims=None) -> PropArray|float:
        """
        Wavelet-based estimation of Gaussian noise.

        Parameters
        ----------
        squeeze : bool, default is True
            If True and output can be converted to a scalar, then convert it.
        dims : str, optional
            Spatial dimension.

        Returns
        -------
        PropArray or float
            Estimated standard deviation. sigma["t=0;c=1"] means the estimated value of
            image slice at t=0 and c=1.
        """        
        c_axes = complement_axes(dims, self.axes)
        out = self.apply_dask(skres.estimate_sigma,
                              c_axes=c_axes,
                              drop_axis=dims
                              )
            
        if out.ndim == 0 and squeeze:
            out = out[()]
        else:
            out = PropArray(out, dtype=np.float32, name=self.name, 
                            axes=c_axes, propname="estimate_sigma")
            out._set_info(self, new_axes=c_axes)
        return out       
        
    
    @dims_to_spatial_axes
    @record()
    def pad(self, pad_width, mode:str="constant", *, dims=None, **kwargs) -> ImgArray:
        """
        Pad image only for spatial dimensions.

        Parameters
        ----------
        pad_width, mode, **kwargs : 
            See documentation of np.pad().
        dims : int or str, optional
            Which dimension to pad.
        **kwargs :
            Passed to np.pad().

        Returns
        -------
        ImgArray
            Padded image.
        
        Example
        -------
        Suppose `img` has zyx-axes.
        
        (1) Padding 5 pixels in zyx-direction:
        >>> img.pad(5)
        (2) Padding 5 pixels in yx-direction:
        >>> img.pad(5, dims="yx")
        (3) Padding 5 pixels in yx-direction and 2 pixels in z-direction:
        >>> img.pad([(5,5), (4,4), (4,4)])
        (4) Padding 10 pixels in z-(-)-direction and 5 pixels in z-(+)-direction.
        >>> img.pad([(10, 5)], dims="z")
        """        
        pad_width_ = []
        
        # for consistency with scipy-format
        if "cval" in kwargs.keys():
            kwargs["constant_values"] = kwargs["cval"]
            kwargs.pop("cval")
            
        if hasattr(pad_width, "__iter__") and len(pad_width) == len(dims):
            pad_iter = iter(pad_width)
            for a in self.axes:
                if a in dims:
                    pad_width_.append(next(pad_iter))
                else:
                    pad_width_.append((0, 0))
            
        elif isinstance(pad_width, int):
            for a in self.axes:
                if a in dims:
                    pad_width_.append((pad_width, pad_width))
                else:
                    pad_width_.append((0, 0))
        else:
            raise TypeError(f"pad_width must be iterable or int, but got {type(pad_width)}")
        
        padimg = np.pad(self.value, pad_width_, mode, **kwargs).view(self.__class__)
        return padimg
    
    @record()
    @same_dtype(asfloat=True)
    def defocus(self, kernel, *, depth:int=3, width:int=6, bg:float=None) -> ImgArray:
        """
        Make a z-directional padded image by defocusing the original image. This padding is
        useful when applying FFT to a 3D image.
        
        Parameters
        ----------
        kernel : 0-, 1- or 3-dimensional array.
            If 0- (scalar) or 1-dimensional array was given, this is interpreted as standard
            deviation of Gaussian kernel. If 3-dimensional array was given, this is directly
            used as convolution kernel. Other dimension will raise ValueError.
        depth : int, default is 3
            Depth of defocusing. For an image with z-axis size L, then output image will have
            size L + 2*depth.
        width : int, default is 6
            Width of defocusing. For an image with yx-shape (M, N), then output image will have
            shape (M * 2*width, N + 2*width).
        bg : float, optional
            Background intensity. If not given, it will calculated as the minimum value of 
            the original image.

        Returns
        -------
        ImgArray
            Padded image.
            
        Examples
        --------
        depth = 2,
        
        ----|   |----| o |--     o ... center of kernel
        ----| o |----|   |--
        ++++|   |++++|___|++  <- the upper edge of original image 
        ++++|___|+++++++++++

        """
        bg = _check_bg(self, bg)
        
        if np.isscalar(kernel):
            kernel = np.array([kernel]*3)
        else:
            kernel = np.asarray(kernel)
        
        if kernel.ndim <= 1:
            def filter_func(img):
                return ndi.gaussian_filter(img, kernel, mode="constant", cval=bg)
            dz, dy, dx = kernel*3 # 3-sigma
            
        elif kernel.ndim == 3:
            kernel = kernel.astype(np.float32)
            kernel = kernel / np.sum(kernel)
            def filter_func(img):
                return ndi.convolve(img, kernel, mode="constant", cval=bg)
            dz, dy, dx = np.array(kernel.shape)//2
        else:
            raise ValueError("`kernel` only take 0, 1, 3 dimensional array as an input.")
        
        pad_width = [(depth, depth), (width, width), (width, width)]
        padimg = self.pad(pad_width, mode="constant", constant_values=bg, dims="zyx")
        out = np.copy(padimg.value)
        # convolve psf
        z_edge0 = slice(None, depth, None)
        z_mid = slice(depth, -depth, None)
        z_edge1 = slice(-depth, None, None)
        y_edge0 = x_edge0 = slice(None, width, None)
        y_mid = slice(width, -width, None)
        y_edge1 = x_edge1 = slice(-width, None, None)
        
        for sl, img in padimg.iter(complement_axes("zyx", self.axes)):
            out_ = out[sl]
            out_[z_edge0] = filter_func(img[:depth+dz ])[z_edge0]
            out_[z_edge1] = filter_func(img[-depth-dz:])[z_edge1]
            out_[z_mid, y_edge0] = filter_func(img[:, :width+dy ])[z_mid, y_edge0]
            out_[z_mid, y_edge1] = filter_func(img[:, -width-dy:])[z_mid, y_edge1]
            out_[z_mid, y_mid, x_edge0] = filter_func(img[:, :, :width+dx ])[z_mid, y_mid, x_edge0]
            out_[z_mid, y_mid, x_edge1] = filter_func(img[:, :, -width-dx:])[z_mid, y_mid, x_edge1]
            
        return out.view(self.__class__)
    
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def wiener(self, psf:np.ndarray, lmd:float, *, dims=None, update:bool=False) -> ImgArray:
        """
        Classical wiener deconvolution. This algorithm has the serious ringing problem
        if parameters are set to wrong values.

        Parameters
        ----------
        psf : np.ndarray
            Point spread function
        lmd : float
            Constant value used in the deconvolution. See Formulation below.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Deconvolved image.
        
        Formulation
        -----------
        
                 F[Yo] x H*
        F[Yr] = -----------
                 |H|^2 + λ
        
         Yo: observed image
         Yr: restored image
         H : fft of psf
        `*`: conjugation of complex number
        """        
        if lmd <= 0:
            raise ValueError(f"lmd must be positive, but got: {lmd}")
        
        psf = _deconv.check_psf(self, psf, dims)
        psf_ft = rfft(psf)
        psf_ft_conj = np.conjugate(psf_ft)
        
        return self.apply_dask(_deconv.wiener, 
                               c_axes=complement_axes(dims, self.axes),
                               args=(psf_ft, psf_ft_conj, lmd)
                               )
        
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def lucy(self, psf:np.ndarray, niter:int=50, eps:float=1e-5, *, dims=None, 
             update:bool=False) -> ImgArray:
        """
        Deconvolution of N-dimensional image, using Richardson-Lucy's algorithm.
        
        Parameters
        ----------
        psf : np.ndarray
            Point spread function.
        niter : int, default is 50.
            Number of iterations.
        eps : float, default is 1e-5
            During deconvolution, division by small values in the convolve image of estimation and 
            PSF may cause divergence. Therefore, division by values under `eps` is substituted
            to zero.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to the result.
        
        Returns
        -------
        ImgArray
            Deconvolved image.
        """
        
        psf = _deconv.check_psf(self, psf, dims)
        
        # calculate FFT of PSF and its conjugate in advance
        psf_ft = rfft(psf)
        psf_ft_conj = np.conjugate(psf_ft)
        
        return self.apply_dask(_deconv.richardson_lucy, 
                               c_axes=complement_axes(dims, self.axes),
                               args=(psf_ft, psf_ft_conj, niter, eps)
                               )
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def lucy_tv(self, psf:np.ndarray, max_iter:int=50, lmd:float=1e-3, tol:float=1e-3, eps=1e-5, 
                *, dims=None, update:bool=False) -> ImgArray:
        """
        Deconvolution of N-dimensional image, using Richardson-Lucy's algorithm with total variance
        regularization (so called RL-TV algorithm). The TV regularization factor at pixel position x,
        Freg(x), is calculated as:
        
                                         1
            Freg(x) = ----------------------------------------       (I(x): image, λ: constant)
                       1 - λ*div( grad(I(x)) / |grad(I(x))| )
        
        and this factor is multiplied for every estimation made in each iteration.
        
        Parameters
        ----------
        psf : np.ndarray
            Point spread function.
        max_iter : int, default is 50.
            Maximum number of iterations.
        lmd : float, default is 1e-3
            The constant lambda of TV regularization factor.
        tol : float, default is 1e-3
            Iteration stops if regularized absolute summation is lower than this value.
            
                        Σ|I'(x) - I(x)|
                gain = -----------------
                            Σ|I(x)|
            (I'(x): estimation of k+1-th iteration, I(x): estimation of k-th iteration)
        
        eps : float, default is 1e-5
            During deconvolution, division by small values in the convolve image of estimation and 
            PSF may cause divergence. Therefore, division by values under `eps` is substituted
            to zero.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, optional
            If update self to the result.
        
        Returns
        -------
        ImgArray
            Deconvolved image.
        
        Reference
        ---------
        - Dey, N., Blanc-Féraud, L., Zimmer, C., Roux, P., Kam, Z., Olivo-Marin, J. C., 
          & Zerubia, J. (2004). 3D microscopy deconvolution using Richardson-Lucy algorithm 
          with total variation regularization (Doctoral dissertation, INRIA).
        """
        psf = _deconv.check_psf(self, psf, dims)
        if lmd <= 0:
            raise ValueError("In Richadson-Lucy with total-variance-regularization, parameter `lmd` "
                             "must be positive.")
        
        # calculate FFT of PSF and its conjugate in advance
        psf_ft = rfft(psf)
        psf_ft_conj = np.conjugate(psf_ft)
        
        return self.apply_dask(_deconv.richardson_lucy_tv, 
                               c_axes=complement_axes(dims, self.axes),
                               args=(psf_ft, psf_ft_conj, max_iter, lmd, tol, eps)
                               )

def _check_coordinates(coords, img, dims=None):
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

def _check_function(func):
    if isinstance(func, str):
        func = getattr(np, func, None)
    if callable(func):
        return func
    else:
        raise TypeError("Must be one of numpy methods or callable object.")

def _check_bg(img, bg):
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

def _translate_image(img, shift, order=1, cval=0):
    mx = sktrans.AffineTransform(translation=-np.asarray(shift))
    return sktrans.warp(img, mx, order=order, cval=cval)

def _calc_centroid(img, ndim):
    mom = skmes.moments(img, order=1)
    centroid = np.array([mom[(0,)*i + (1,) + (0,)*(ndim-i-1)] 
                        for i in range(ndim)]) / mom[(0,)*ndim]
    return centroid

def _check_clip_range(in_range, img):
    """
    Called in clip_outliers() and rescale_intensity().
    """    
    lower, upper = in_range
    if isinstance(lower, str) and lower.endswith("%"):
        lower = float(lower[:-1])
        lowerlim = np.percentile(img, lower)
    elif lower is None:
        lowerlim = np.min(img)
    else:
        lowerlim = float(lower)
    
    if isinstance(upper, str) and upper.endswith("%"):
        upper = float(upper[:-1])
        upperlim = np.percentile(img, upper)
    elif upper is None:
        upperlim = np.max(img)
    else:
        upperlim = float(upper)
    
    if lowerlim >= upperlim:
        raise ValueError(f"lowerlim is larger than upperlim: {lowerlim} >= {upperlim}")
    
    return lowerlim, upperlim

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