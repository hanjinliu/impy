from __future__ import annotations
import itertools
import numpy as np
import trackpy as tp
import os
import glob
import collections
from skimage import io
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import exposure as skexp
from skimage import measure as skmes
from skimage import segmentation as skseg
from skimage import feature as skfeat
from skimage import registration as skreg
from scipy.linalg import pinv as pseudo_inverse
from .func import *
from .deco import *
from .gauss import GaussianBackground, GaussianParticle
from .labeledarray import LabeledArray
from .label import Label
from .axes import Axes, ImageAxesError
from .specials import *
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
    def gaussfit(self, scale:float=1/16, p0=None, show_result:bool=True, method="Powell") -> ImgArray:
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
        result = gaussian.fit(rough, method=method)
        gaussian.rescale(1/scale)
        fit = gaussian.generate(self.shape).view(self.__class__)
        fit.temp = dict(params=gaussian.params, result=result)
        
        # show fitting result
        show_result and plot_gaussfit_result(self, fit)
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
        high_sigma = low_sigma * 1.6 if high_sigma is None else high_sigma
        
        return self.parallel(difference_of_gaussian_, complement_axes(dims, self.axes),
                             low_sigma, high_sigma)
    
    @dims_to_spatial_axes
    @record()
    def doh_filter(self, sigma:float=1, *, dims=None) -> ImgArray:
        """
        Determinant of Hessian filter. This function does not support `update`
        argument because output has total different scale of intensity. Because in
        most cases we want to find only bright dots, eigenvalues larger than 0 is
        ignored before computing determinant.

        Parameters
        ----------
        sigma : scalar or array of scalars, optional
            Standard deviation(s) of Gaussian filter.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        ndim = len(dims)
        sigma = check_nd_sigma(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])
        return self.as_float().parallel(hessian_det_, complement_axes(dims, self.axes), 
                                        sigma, pxsize)
    
    @dims_to_spatial_axes
    @record()
    def log_filter(self, sigma:float=1, *, dims=None) -> ImgArray:
        """
        Laplacian of Gaussian filter.

        Parameters
        ----------
        sigma : scalar or array of scalars, optional
            Standard deviation(s) of Gaussian filter.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        return -self.as_float().parallel(gaussian_laplace_, complement_axes(dims, self.axes),
                                        sigma)
    
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
                       topn:int=np.inf, topn_per_label:int=np.inf, exclude_border=True,
                       use_labels:bool=True, dims=None):
        """
        Find local maxima. This algorithm corresponds to ImageJ's 'Find Maxima' but
        is more flexible.

        Parameters
        ----------
        min_distance : int, by default 1
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
            Dimension of axes.
            
        Returns
        -------
        MarkerFrame
            DataFrame with columns same as axes of self. For example, if self.axes is "tcyx" then
            return value has "t", "c", "y" and "x" columns, and sub-frame at t=0, c=0 contains all
            the coordinates of peaks in the slice at t=0, c=0.
        """        
        
        # separate spatial dimensions and others
        ndim = len(dims)
        c_axes = complement_axes(dims, self.axes)
        
        if isinstance(exclude_border, bool):
            exclude_border = int(min_distance) if exclude_border else False
        
        thr = None if percentile is None else np.percentile(self.value, percentile)
                
        out = []
        
        self.ongoing = "peak_local_max"
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
            out += [sl + tuple(ind) for ind in indices]
        
            
        out = MarkerFrame(out, columns=self.axes, dtype="uint16")
        out.set_scale(self)
        
        self.ongoing = None
        del self.ongoing
            
        return out
    
    @dims_to_spatial_axes
    def corner_peaks(self, *, min_distance:int=1, percentile:float=None, 
                     topn:int=np.inf, topn_per_label:int=np.inf, exclude_border=True,
                     use_labels:bool=True, dims=None):
        """
        Find local corner maxima. Slightly different from peak_local_max.

        Parameters
        ----------
        min_distance : int, by default 1
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
            Dimension of axes.
            
        Returns
        -------
        MarkerFrame
            DataFrame with columns same as axes of self. For example, if self.axes is "tcyx" then
            return value has "t", "c", "y" and "x" columns, and sub-frame at t=0, c=0 contains all
            the coordinates of corners in the slice at t=0, c=0.
        """        
        
        # separate spatial dimensions and others
        ndim = len(dims)
        c_axes = complement_axes(dims, self.axes)
        
        if isinstance(exclude_border, bool):
            exclude_border = int(min_distance) if exclude_border else False
        
        thr = None if percentile is None else np.percentile(self.value, percentile)
                
        out = []
        
        self.ongoing = "corner_peaks"
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
            out += [sl + tuple(ind) for ind in indices]
        
            
        out = MarkerFrame(out, columns=self.axes, dtype="uint16")
        out.set_scale(self)
        
        self.ongoing = None
        del self.ongoing
            
        return out
    
    @dims_to_spatial_axes
    @record()
    def corner_harris(self, sigma=1, k=0.05, *, dims=None):
        
        return self.parallel(corner_harris_, complement_axes(dims, self.axes), k, sigma)
    
    @dims_to_spatial_axes
    def find_corners(self, sigma=1, k=0.05, *, dims=None):
        
        res = self.gaussian_filter(sigma=1).corner_harris(sigma=sigma, k=k, dims=dims)
        out = res.corner_peaks(min_distance=3, percentile=97, dims=dims)
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def trackpy_sm(self, radius, *, return_all=False, dims=None, **kwargs):
        out = pd.DataFrame()
        c_axes = complement_axes(dims, self.axes)
        for sl, data in self.iter(c_axes, exclude=dims):
            df = tp.locate(data, 2*radius+1, **kwargs)
            for a, i in zip(c_axes, sl):
                df[a] = i
            out = pd.concat([out, df], axis=0)
        
        out.index = np.arange(len(out))
        mf = MarkerFrame(out.reindex(columns=[a for a in self.axes]), columns=str(self.axes))
        mf.set_scale(self.scale)
        if return_all:
            df = out[out.columns[out.columns.isin([a for a in out.columns if a not in dims])]]
            return FrameDict(markers=mf, results=df)
        else:
            return mf
    
    @dims_to_spatial_axes
    def refine(self, coords=None, radius:float=4, *, percentile=90, n_iter=10, sigma=1.5, dims=None):
        # TODO: what should be this function like?
        if coords is None:
            coords = self.find_sm(sigma=sigma, dims=dims, percentile=percentile, exclude_border=radius)
        self.specify(coords, radius, labeltype="circle")
        
        radius = tp.utils.validate_tuple(radius, len(dims))
        sigma = tp.utils.validate_tuple(sigma, len(dims))
        sigma = tuple([int(x) for x in sigma])
        refined_coords = tp.refine.refine_com(self.value, self.value, radius, coords,
                                              max_iterations=n_iter, pos_columns=[a for a in dims])
        bg = self.value[self.labels==0]
        black_level = np.mean(bg)
        noise = np.std(bg)
        Npx = tp.masks.N_binary_mask(radius, len(dims))
        mass = refined_coords['raw_mass'].values - Npx * black_level
        ep = tp.uncertainty._static_error(mass, noise, radius, sigma)
        
        if ep.ndim == 1:
            refined_coords['ep'] = ep
        else:
            ep = pd.DataFrame(ep, columns=['ep_' + cc for cc in [a for a in dims]])
            refined_coords = pd.concat([refined_coords, ep], axis=1)
        mf = MarkerFrame(refined_coords.reindex(columns=[a for a in self.axes]), columns=str(self.axes))
        mf.set_scale(self.scale)
        df = refined_coords[refined_coords.columns[refined_coords.columns.isin([a for a in refined_coords.columns if a not in dims])]]
        return FrameDict(markers=mf, results=df)
        
    
    @dims_to_spatial_axes
    def find_sm(self, sigma:float=1.5, *, method="dog", percentile:float=95, topn:int=np.inf, 
                exclude_border=True, dims=None):
        """
        Single molecule detection using difference of Gaussian method.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of puncta.
        method : str, by default "dog"
            Which filter is used prior to finding local maxima. Currently supports "dog", "doh" 
            and "log".
        percentile, topn, exclude_border, dims
            Passed to peak_local_max()

        Returns
        -------
        MarkerFrame
            Peaks in uint16 type.
        """        
        methods_ = {"dog": "dog_filter",
                    "doh": "doh_filter",
                    "log": "log_filter",
                    }
        try:
            fil_img = getattr(self, methods_[method])(sigma, dims=dims)
        except KeyError:
            raise ValueError(f"Currently `method` only supports {', '.join(methods_.keys())}")
        
        coords = fil_img.peak_local_max(min_distance=sigma*2, percentile=percentile, 
                                        topn=topn, dims=dims, exclude_border=exclude_border)
        return coords
    
    
    @dims_to_spatial_axes
    def centroid_sm(self, coords=None, radius:float=4, sigma:float=1.5, filt=None,
                    percentile:float=90, *, dims=None) -> MarkerFrame:
        """
        Calculate positions of particles in subpixel precision using centroids.

        Parameters
        ----------
        coords : MarkerArray or MarkerFrame, optional
            Coordinates of peaks. If None, this will be determined by find_sm.
        radius : float, by default 4.
            Range to calculate centroids. Rectangular image with size 2r+1 x 2r+1 will be send 
            to calculate moments.
        sigma : float, by default 1.5
            Expected standard deviation of particles.
        filt : callable, optional
            For every slice `sl`, label is added only when filt(`input`) == True is satisfied.
        percentile, dims
            Passed to peak_local_max()
        dims : int or str, optional
            Dimension of axes.
        """     
        if coords is None:
            coords = self.find_sm(sigma=sigma, dims=dims, 
                                  percentile=percentile)
            return self.centroid_sm(coords, radius=radius, sigma=sigma, filt=filt, 
                                    percentile=percentile, dims=dims)
        
        elif isinstance(coords, MarkerFrame):
            ndim = len(dims)
            filt = check_filter_func(filt)
            
            if np.isscalar(radius):
                radius = np.full(ndim, radius)
            radius = np.asarray(radius)
            
            shape = self.sizesof(dims)
            
            centroids = []  # fitting results of means
            print("centroid_sm ... ", end="")
            timer = Timer()
            for marker in coords.values:
                center = tuple(marker[-ndim:])
                label_sl = tuple(marker[:-ndim])
                sl = specify_one(center, radius, shape, "square") # sl = (..., z,y,x)
                input_img = self.value[label_sl][sl]
                if input_img.size == 0 or not filt(input_img):
                    continue
                
                mom = skmes.moments(input_img, order=1)
                shift = center - radius
                centroid = np.array([mom[(0,)*i + (1,) + (0,)*(ndim-i-1)] for i in range(ndim)])/mom[(0,)*ndim]
                centroids.append(label_sl + tuple(centroid + shift))
                    
            timer.toc()
            print(f"\rcentroid_sm completed ({timer})")
            
            out = MarkerFrame(centroids, columns=coords.col_axes, dtype="float32").as_standard_type()
            out.set_scale(coords.scale)
        else:
            raise NotImplementedError

        return out
    
    @dims_to_spatial_axes
    def gauss_sm(self, coords:MarkerFrame=None, radius:float=4, sigma:float=1.5, filt=None,
                 percentile:float=95, *, return_all=False, dims=None) -> FrameDict:
        """
        Calculate positions of particles in subpixel precision using Gaussian fitting.

        Parameters
        ----------
        coords : MarkerFrame, optional
            Coordinates of peaks. If None, this will be determined by find_sm.
        radius : float, by default 4.
            Fitting range. Rectangular image with size 2r+1 x 2r+1 will be send to Gaussian
            fitting function.
        sigma : float, by default 1.5
            Expected standard deviation of particles.
        filt : callable, optional
            For every slice `sl`, label is added only when filt(`input`) == True is satisfied.
            This discrimination is conducted before Gaussian fitting so that stringent filter
            will save time.
        percentile, dims
            Passed to peak_local_max()
        return_all : bool, by default False
            If True, fitting results are all returned as Frame Dict.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        MarkerFrame, if return_all == False
            Gaussian centers.
        FrameDict with keys {means, sigmas, errors}, if return_all == True
            Dictionary that contains means, standard deviations and fitting errors.
        """        
        
        if coords is None:
            melted = self.find_sm(sigma=sigma, dims=dims, 
                                  percentile=percentile)
            return self.gauss_sm(melted, radius=radius, sigma=sigma, filt=filt, 
                                 percentile=percentile, dims=dims)
        
        elif isinstance(coords, MarkerFrame):
            ndim = len(dims)
            filt = check_filter_func(filt)
            
            if np.isscalar(radius):
                radius = np.full(ndim, radius)
            radius = np.asarray(radius)
            
            shape = self.sizesof(dims)
            
            means = []  # fitting results of means
            sigmas = [] # fitting results of sigmas
            errs = []   # fitting errors of means
            ab = []
            print("gauss_sm ... ", end="")
            timer = Timer()
            for marker in coords.values:
                center = tuple(marker[-ndim:])
                label_sl = tuple(marker[:-ndim])
                sl = specify_one(center, radius, shape, "square") # sl = (..., z,y,x)
                input_img = self.value[label_sl][sl]
                if input_img.size == 0 or not filt(input_img):
                    continue
                
                gaussian = GaussianParticle(initial_sg=sigma)
                res = gaussian.fit(input_img, method="BFGS")
                
                if gaussian.mu_inrange(0, radius*2) and gaussian.sg_inrange(sigma/3, sigma*3) and gaussian.a > 0:
                    gaussian.shift(center - radius)
                    # calculate fitting error with Jacobian
                    # TODO: is this error correct?
                    if return_all:
                        jac = res.jac[:2].reshape(1,-1)
                        cov = pseudo_inverse(jac.T @ jac)
                        err = np.sqrt(np.diag(cov))
                        sigmas.append(label_sl + tuple(gaussian.sg))
                        errs.append(label_sl + tuple(err))
                        ab.append(label_sl + (gaussian.a, gaussian.b))
                    
                    means.append(label_sl + tuple(gaussian.mu))
                    
            timer.toc()
            print(f"\rgauss_sm completed ({timer})")
            
            kw = dict(columns=coords.col_axes, dtype="float32")
            
            if return_all:
                out = FrameDict(means = MarkerFrame(means, **kw).as_standard_type(),
                                sigmas = MarkerFrame(sigmas, **kw).as_standard_type(),
                                errors = MarkerFrame(errs, **kw).as_standard_type(),
                                intensities = MarkerFrame(ab, 
                                                          columns=str(coords.col_axes)[:-ndim]+"ab",
                                                          dtype="float32"))
                
                out.means.set_scale(coords.scale)
                out.sigmas.set_scale(coords.scale)
                out.errors.set_scale(coords.scale)
                    
            else:
                out = MarkerFrame(means, **kw)
                out.set_scale(coords.scale)
            
        else:
            raise NotImplementedError
                            
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
        
        if isinstance(thr, str) and thr.endswith("%"):
            p = float(thr[:-1])
            out = np.zeros(self.shape, dtype=bool)
            for t, img in self.iter(complement_axes(dims, self.axes), False):
                thr = np.percentile(img, p)
                out[t] = img >= thr
                
        elif isinstance(thr, str):
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
    def specify(self, center, radius, filt=None, *, dims=None, labeltype="square") -> ImgArray:
        """
        Make rectangle or ellipse labels from points.
        
        Parameters
        ----------
        center : array like or MarkerFrame
            Coordinates of centers. For MarkerFrame, it must have the same axes order.
        radius : float or array
            Radius of labels.
        filt : callable, optional
            For every slice `sl`, label is added only when filt(self[sl]) is satisfied.
        dims : int or str, optional
            Dimension of axes.
        labeltype : str, by default "square"
            The shape of labels.

        Returns
        -------
        ImgArray
            Labeled image.
        """
        
        if isinstance(center, MarkerFrame):
            # determine dims to iterate.
            # dims = "".join(a for a in dims if a not in center.axes)
            dims = str(center.col_axes)
            ndim = len(dims)
            # convert radius to an array
            if np.isscalar(radius):
                radius = np.full(ndim, radius)
            radius = np.asarray(radius)
            
            shape = self.sizesof(dims)
            label_axes = str(center.col_axes)
            label_shape = self.sizesof(label_axes)
            if hasattr(self, "labels"):
                print("Existing labels are updated.")
            self.labels = Label(np.zeros(label_shape, dtype="uint8"), dtype="uint8", axes=label_axes)
            self.labels.set_scale(self)

            filt = check_filter_func(filt)
            
            print("specify ... ", end="")
            timer = Timer()
            for crd in center.values:
                c = tuple(crd[-ndim:])
                label_sl = tuple(crd[:-ndim])
                sl = specify_one(c, radius, shape, labeltype)
                img_ = self[label_sl][sl]
                if img_.size > 0 and filt(img_):
                    self.labels[label_sl][sl] = self.labels.max() + 1
                    # increase memory if needed
                    if self.labels.max() == np.iinfo(self.labels.dtype).max:
                        self.labels = self.labels.as_larger_type()
                        
            timer.toc()
            print(f"\rspecify completed ({timer})")
        
        else:
            center = np.asarray(center)
            if center.ndim == 1:
                center = center.reshape(1, -1)
            
            cols = {1:"x", 2:"yx", 3:"zyx"}[center.shape[1]]
            center = MarkerFrame(center, columns=cols, dtype="uint16")

            return self.specify(center, radius, filt=filt, dims=dims, labeltype=labeltype)     
        
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
    def skeletonize(self, *, dims=None, update=False) -> ImgArray:
        """
        Skeletonize images. Only works for binary images.

        Parameters
        ----------
        dims : int or str, optional
            Dimension of axes.
        update : bool, optional
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Skeletonized image.
        """        
        if self.dtype != bool:
            raise TypeError("Cannot run skeletonize() with non-binary image.")
        return self.parallel(skeletonize_, complement_axes(dims, self.axes), outdtype=bool)
    
    @dims_to_spatial_axes
    @record()
    def count_neighbors(self, connectivity=None, mask=True, *, dims=None) -> ImgArray:
        """
        Count the number or neighbors of binary images. This function can be used for cross section
        or branch detection. Only works for binary images.

        Parameters
        ----------
        connectivity : int , optional
            See label().
        mask : bool, by default True
            If True, only neighbors of pixels that satisfy self==True is returned.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            uint8 array of the number of neighbors.
        """        
        if self.dtype != bool:
            raise TypeError("Cannot run count_neighbors() with non-binary image.")
        
        ndim = len(dims)
        connectivity = ndim if connectivity is None else connectivity
        selem = ndi.morphology.generate_binary_structure(ndim, connectivity)
        
        out = self.parallel(count_neighbors_, complement_axes(dims, self.axes), selem)
        if mask:
            out[~self.value] = 0
            
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def reslice(self, src, dst, linewidth=1, *, order=None, dims=None) -> ImgArray:
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
    def watershed(self, markers:PropArray=None, *, connectivity=1, input="distance", 
                  min_distance=2, dims=None) -> ImgArray:
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
        
        # Prepare the input image.
        if input == "self":
            input_img = self.copy()
        elif input == "distance":
            input_img = self.__class__(self.labels>0, axes=self.axes).distance_map(dims=dims)
        else:
            raise ValueError("'input_' must be either 'self' or 'distance'.")
        
                
        if input_img.dtype == bool:
            input_img = input_img.astype("uint8")
            
        input_img._view_labels(self)
        
        if markers is None:
            markers = input_img.peak_local_max(min_distance=min_distance, dims=dims)
        
        labels = np.zeros(input_img.shape, dtype="uint32")
        input_img.ongoing = "watershed"
        shape = self.sizesof(dims)
        n_labels = 0
        c_axes = complement_axes(dims, self.axes)
        marker_input = np.zeros(shape, dtype="uint32") # placeholder for maxima
        for (sl, img), (_, crd) in zip(input_img.iter(c_axes, israw=True),
                                       markers.groupby([a for a in c_axes])):
            crd = crd.values.T.tolist()
            marker_input[tuple(crd)] = np.arange(1, len(crd[0])+1, dtype="uint32")
            labels[sl] = skseg.watershed(-img.value, marker_input, 
                                         mask=img.labels.value, 
                                         connectivity=connectivity)
            labels[sl][labels[sl]>0] += n_labels
            n_labels = labels[sl].max()
            marker_input[:] = 0 # reset placeholder
            
        input_img.ongoing = None
        del input_img.ongoing
        
        labels = labels.view(Label)
        self.labels = labels.optimize()
        self.labels._set_info(self)
        self.labels.set_scale(self)
        return self
    
    @dims_to_spatial_axes
    def label_threshold(self, thr="otsu", *, dims=None, **kwargs) -> ImgArray:
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
        labels = self.threshold(thr=thr, dims=dims, **kwargs)
        return self.label(labels, dims=None)
    
        
    @need_labels
    @record(append_history=False)
    def regionprops(self, properties:tuple[str,...]=("mean_intensity",), *, 
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
        
        for sl, img in self.iter(prop_axes, exclude=self.labels.axes):
            props = skmes.regionprops(self.labels, img, cache=False,
                                      extra_properties=extra_properties)
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                # Both sides have length of p-axis (number of labels) so that values
                # can be correctly substituted.
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]
                
        timer.toc()
        
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
    def track_drift(self, axis="t", show_drift=True, **kwargs) -> MarkerFrame:
        """
        Calculate xy-directional drift using `skimage.registration.phase_cross_correlation`.

        Parameters
        ----------
        axis : str, by default "t"
            Along which axis drift will be calculated.
        show_drift : bool, by default True
            If True, plot the result.

        Returns
        -------
        MarkerArray
            An array with rt-axis.

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
        
        result = MarkerFrame(np.array(result), columns="yx")
        
        show_drift and plot_drift(result)
        result.index.name = axis
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
        
        elif isinstance(shift, MarkerFrame):
            if len(shift) != self.sizeof("t"):
                raise ValueError("Wrong shape of 'shift'.")
        
        else:
            shift = MarkerFrame(shift, columns="yx", dtype="float32")
            return self.drift_correction(shift, ref, order=order, along=along, 
                                         dims=dims,update=update)

        out = np.empty(self.shape)
        t_index = self.axisof(along)
        shift = shift.reindex(columns=["x", "y"])
        for sl, img in self.iter(complement_axes(dims, self.axes)):
            trans = -shift.loc[sl[t_index]]
            mx = sktrans.AffineTransform(translation=trans)
            out[sl] = sktrans.warp(img.astype("float32"), mx, order=order)
        
        out = out.view(self.__class__)
        return out

    def estimate_sigma(self):
        # TODO: multi-dimensional
        return skres.estimate_sigma(self.value)
    
    @dims_to_spatial_axes
    @record()
    def pad(self, pad_width, mode="constant", *, dims=None,  **kwargs) -> ImgArray:
        """
        Pad image only for spatial dimensions.

        Parameters
        ----------
        pad_width, mode, **kwargs : 
            See documentation of np.pad().
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Padded image.
        """        
        pad_width_ = []
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
        niters : int, by default 50.
            Number of iteration.
        dims : int or str, optional
            Dimension of axes.
        update : bool, optional
            If update self to filtered image.
        """
        
        psf = check_psf(self, psf, dims)
        
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
    """
    Read the metadata of a tiff file. 

    Parameters
    ----------
    path : str
        Path to the tiff file.

    Returns
    -------
    dict
        Dictionary of metadata with following keys.
        "axes": axes information
        "ijmeta": ImageJ metadata
        "history": impy history
        "tags": tiff tags
    """    
    
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


