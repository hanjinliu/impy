from __future__ import annotations
import numpy as np
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import exposure as skexp
from skimage import measure as skmes
from skimage import segmentation as skseg
from skimage import feature as skfeat
from skimage import registration as skreg
from scipy.linalg import pinv as pseudo_inverse
from scipy.spatial import Voronoi
from .func import *
from .deco import *
from .labeledarray import LabeledArray
from .label import Label
from .phasearray import PhaseArray
from .specials import *
from .utilcls import *
from ._process import *


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
    @check_value
    def __mul__(self, value):
        return super().__mul__(value)
    
    @same_dtype(asfloat=True)
    @check_value
    def __imul__(self, value):
        return super().__imul__(value)
    
    @check_value
    def __truediv__(self, value):
        self = self.astype(np.float32)
        if isinstance(value, np.ndarray):
            value[value==0] = np.inf
        return super().__truediv__(value)
    
    @check_value
    def __itruediv__(self, value):
        self = self.astype(np.float32)
        if isinstance(value, np.ndarray):
            value[value==0] = np.inf
        return super().__itruediv__(value)
    
    def freeze(self):
        """
        To avoid further image analysis, convert to LabeledArray.
        """        
        return self.view(LabeledArray)
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def affine(self, *, dims=None, order:int=1, **kwargs) -> ImgArray:
        """
        Convert image by Affine transformation. 2D Affine transformation is written as:
        [x']   [A00 A01 A02]   [x]
        [y'] = [A10 A11 A12] * [y]
        [1 ]   [  0   0   1]   [1]

        Parameters
        ----------
        dims : int or str, optional
            Spatial dimensions.
        order : int, default is 1.
            Interpolation order after transformation.
        kwargs : matrix, scale, rotation, shear, translation. 
            See `skimage.transform.AffineTransform`.
            
        Returns
        -------
        ImgArray
            Transformed image.
        """
        mx = sktrans.AffineTransform(**kwargs)
        out = self.parallel(affine_, complement_axes(dims, self.axes), mx, order)
        return out
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def translate(self, translation=None, *, dims=None, order:int=1) -> ImgArray:
        """
        Translation of an image. for skimage < 0.19, only 2D translation is implemented.

        Parameters
        ----------
        translation : array-like, optional
            Inverse map of translation. This is xyz-order., by default None
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
        out = self.parallel(affine_, complement_axes(dims, self.axes), mx, order)
        return out


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
        scale_ = [scale if a in dims else 1 for a in self.axes]
        out = sktrans.rescale(self.value, scale_, order=order, anti_aliasing=False)
        out = out.view(self.__class__)
        out._set_info(self, f"rescale(scale={scale})")
        out.axes = str(self.axes) # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out
    
    @record()
    def gaussfit(self, scale:float=1/16, p0=None, show_result:bool=True, method="Powell") -> ImgArray:
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
    @same_dtype(True)
    def gauss_correction(self, ref=None, scale:float=1/16, median_radius:float=15):
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
            Radius of median prefilter's kernel.

        Returns
        -------
        ImgArray
            Corrected and background subtracted image.
        """        
        if "c" in self.axes:
            out = np.empty(self.shape, dtype=np.float32)
            if isinstance(ref, self.__class__) and "c" in ref.axes:
                refs = ref.split("c")
            else:
                refs = [ref]*self.sizeof("c")
                
            for i, (sl, img) in enumerate(self.iter("c", israw=True)):
                ref_ = refs[i]
                out[sl] = img.gauss_correction(ref=ref_, scale=scale, median_radius=median_radius)
            out = out.view(self.__class__)
            return out
        
        else:
            if ref is None:
                if self.ndim != 2:
                    raise ValueError("`ref` must be given except for images with axes 'yx' or 'cyx'.")
                else:
                    return self.gauss_correction(ref=self, scale=scale, median_radius=median_radius)
            elif isinstance(ref, self.__class__):
                pass
            else:
                raise TypeError(f"`ref` must be None or ImgArray, but got {type(ref)}")
        
        if median_radius >= 1:
            ref = ref.median_filter(radius=median_radius)
        fit = ref.gaussfit(scale=scale, show_result=False).value
        a = fit.max()
        out = self.value / fit * a - a
        out = out.view(self.__class__)
        return out
                
    
    @record()
    def affine_correction(self, matrices=None, *, bins:int=256, order:int=1, 
                          prefilter:bool=True, along:str="c") -> ImgArray:
        """
        Correct chromatic aberration using Affine transformation. Input matrix is
        determined by maximizing normalized mutual information.
        
        Parameters
        ----------
        matrices : array or iterable of arrays, optional
            Affine matrices.
        bins : int, optional
            Number of bins that is generated on calculating mutual information, 
            by default 256
        order : int, optional
            Interporation order, by default 3
        prefilter : bool
            If median filter is applied to all images before fitting. This does not
            change original images. By default True.
        along : str
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
            matrices = check_matrix(matrices)
            
        # Determine matrices by fitting
        # if Affine matrix is not given
        if matrices is None:
            if prefilter:
                imgs = self.median_filter(radius=1).split(along)
            else:
                imgs = self.split(along)
            matrices = [1] + [affinefit(img, imgs[0], bins, order) for img in imgs[1:]]
        
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
        
        Example
        -------
        Extract filament
        >>> eig = -img.hessian_eigval()["l=0"]
        >>> eig[eig<0] = 0
        """        
        ndim = len(dims)
        sigma = check_nd(sigma, ndim)
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
        sigma = check_nd(sigma, ndim)
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
        sigma = check_nd(sigma, ndim)
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
        sigma = check_nd(sigma, ndim)
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
    def edge_filter(self, method="scharr", *, dims=None, update:bool=False) -> ImgArray:
        """
        Sobel filter. This filter is useful for edge detection.

        Parameters
        ----------
        method : str, {"sobel", "farid", "scharr", "prewitt"}, default is "scharr"
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
        method_dict = {"sobel": sobel_,
                       "farid": farid_,
                       "scharr": scharr_,
                       "prewitt": prewitt_}
        try:
            f = method_dict[method]
        except KeyError:
            raise ValueError("`method` must be 'sobel', 'farid' 'scharr', or 'prewitt'.")
        
        return self.parallel(f, complement_axes(dims, self.axes))
    
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @record()
    def convolve(self, kernel, *, mode="reflect", cval=0, dims=None, update:bool=False) -> ImgArray:
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
        return self.parallel(convolve_, complement_axes(dims, self.axes), kernel, mode, cval, outdtype=self.dtype)
    
    @dims_to_spatial_axes
    @same_dtype()
    def _running_kernel(self, radius:float, function=None, *, dims=None, update:bool=False) -> ImgArray:
        disk = ball_like(radius, len(dims))
        return self.parallel(function, complement_axes(dims, self.axes), disk, outdtype=self.dtype)
    
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
        f = binary_erosion_ if self.dtype == bool else erosion_
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
        f = binary_dilation_ if self.dtype == bool else dilation_
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
        f = binary_opening_ if self.dtype == bool else opening_
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
        f = binary_closing_ if self.dtype == bool else closing_
        return self._running_kernel(radius, f, dims=dims, update=update)
    
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
        return self._running_kernel(radius, tophat_, dims=dims, update=update)
    
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
        return self._running_kernel(radius, mean_, dims=dims, update=update)
    
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
        return self.as_float().parallel(std_, complement_axes(dims, self.axes), disk)
    
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
        return self.as_float().parallel(coef_, complement_axes(dims, self.axes), disk)
    
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
        return self._running_kernel(radius, median_, dims=dims, update=update)
    
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def diameter_opening(self, diameter:int=8, *, connectivity:int=1, dims=None, 
                         update:bool=False) -> ImgArray:
        return self.parallel(diameter_opening_, complement_axes(dims, self.axes), 
                             diameter, connectivity)
        
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def diameter_closing(self, diameter:int=8, *, connectivity:int=1, dims=None,
                         update:bool=False) -> ImgArray:
        return self.parallel(diameter_closing_, complement_axes(dims, self.axes), 
                             diameter, connectivity)
    
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def area_opening(self, area:int=64, *, connectivity:int=1, dims=None, update:bool=False) -> ImgArray:
        f = binary_area_opening_ if self.dtype == bool else area_opening_
        return self.parallel(f, complement_axes(dims, self.axes), area, connectivity)
        
    @record()
    @dims_to_spatial_axes
    @same_dtype()
    def area_closing(self, area:int=64, *, connectivity:int=1, dims=None, update:bool=False) -> ImgArray:
        f = binary_area_closing_ if self.dtype == bool else area_closing_
        return self.parallel(f, complement_axes(dims, self.axes), area, connectivity)
        
    @dims_to_spatial_axes
    @record()
    @same_dtype()
    def directional_median_filter(self, radius:int=2, *, dims=None, update:bool=False) -> ImgArray:
        """
        Median filtering in the directional method. Median is calculated in four directions and
        the median value in which direction standard deviation is smallest is used. This method
        retains edge sharpness. Also, this method is not slower than classical median filter in many
        cases because the kernel size is smaller.
        
        Parameters
        ----------
        radius : int, optional
            Kernel radius of the filter. Here, radius must be int.
        dims : int or str, optional
            Spatial dimensions.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Filtered image.
            
        Reference
        ---------
        Modified from following paper:
        Chen, Z., & Zhang, L. (2009). Multi-stage directional median filter. International Journal 
        of Signal Processing, 5(4), 249-252.
        """        
        if len(dims) != 2:
            raise ValueError("Directional median filter is defined only for 2D images.")
        elif not isinstance(radius, int):
            raise TypeError(f"`radius` must be int, but got {type(radius)}")
        return self.parallel(directional_median_, complement_axes(dims, self.axes), radius)
    
    @dims_to_spatial_axes
    @record()
    def entropy_filter(self, radius:float=5, *, dims=None) -> ImgArray:
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
        self = self.as_float() / self.max() # skimage's entropy filter only accept [-1, 1] float images.
        return self.parallel(entropy_, complement_axes(dims, self.axes), disk)
    
    @dims_to_spatial_axes
    @record()
    def enhance_contrast(self, radius:float=1, *, dims=None, update:bool=False) -> ImgArray:
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
        return self._running_kernel(radius, enhance_contrast_, dims=dims, update=update)
    
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
            Dimension of axes.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        ndim = len(dims)
        _, laplace_op = skres.uft.laplacian(ndim, (2*radius+1,) * ndim)
        return self.parallel(convolve_, complement_axes(dims, self.axes), laplace_op, 
                             "reflect", 0, outdtype=self.dtype)
    
    @record(append_history=False)
    def focus_map(self, radius:int=1, *, dims="yx") -> PropArray:
        """
        Compute focus map using variance of Laplacian method. yx-plane with higher variance is likely a
        focal plane because sharper image causes higher value of Laplacian on the edges.

        Parameters
        ----------
        radius : int, by default 1
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
    @same_dtype(True)
    def unmix(self, matrix, bg=None, *, update:bool=False) -> ImgArray:
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
        n_chn = self.sizeof("c")
        c_ax = self.axisof("c")
        
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
        out = (input_ - bg) @ np.linalg.inv(matrix) + bg
        # restore the axes order
        out = np.moveaxis(out, -1, c_ax)
        
        return out.view(self.__class__)
        
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def fill_hole(self, thr="otsu", *, dims=None, update:bool=False) -> ImgArray:
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
            Dimension of axes.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        ImgArray
            Hole-filled image.
        """        
        # TODO: use ndimage.binary_fill_holes?
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
        sigma : scalar or array of scalars, default is 1.
            Standard deviation(s) of Gaussian filter.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        ndim = len(dims)
        sigma = check_nd(sigma, ndim)
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
        sigma : scalar or array of scalars, default is 1.
            Standard deviation(s) of Gaussian filter.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        return -self.as_float().parallel(gaussian_laplace_, complement_axes(dims, self.axes), sigma)
    
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(True)
    def rolling_ball(self, radius:float=50, smoothing:bool=True, *, dims=None, update:bool=False) -> ImgArray:
        """
        Subtract Background using rolling-ball algorithm.

        Parameters
        ----------
        radius : int, default is 50.
            Radius of rolling ball.
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
    @record()
    @same_dtype()
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
        return self.parallel(wavelet_denoising_, complement_axes(dims, self.axes), 
                             func_kw, max_shifts, shift_steps)
    
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
        >>> ip.window.add(img_pol.proj)
        >>> ip.window.add(dpol.aop.rad2deg())
        
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
                       topn:int=np.inf, topn_per_label:int=np.inf, exclude_border=True,
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
        dims_list = [a for a in dims]
        c_axes = complement_axes(dims, self.axes)
        c_axes_list = [a for a in c_axes]
        
        if isinstance(exclude_border, bool):
            exclude_border = int(min_distance) if exclude_border else False
        
        thr = None if percentile is None else np.percentile(self.value, percentile)
                
        out = pd.DataFrame()
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
            out = pd.concat([out, indices], axis=0)
            
        out = MarkerFrame(out, columns=self.axes, dtype="uint16")
        out.set_scale(self)
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def corner_peaks(self, *, min_distance:int=1, percentile:float=None, 
                     topn:int=np.inf, topn_per_label:int=np.inf, exclude_border=True,
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
        dims_list = [a for a in dims]
        c_axes = complement_axes(dims, self.axes)
        c_axes_list = [a for a in c_axes]
        
        if isinstance(exclude_border, bool):
            exclude_border = int(min_distance) if exclude_border else False
        
        thr = None if percentile is None else np.percentile(self.value, percentile)
                
        out = pd.DataFrame()
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
            out = pd.concat([out, indices], axis=0)
            
        out = MarkerFrame(out, columns=self.axes, dtype="uint16")
        out.set_scale(self)
        return out
    
    @dims_to_spatial_axes
    @record()
    def corner_harris(self, sigma=1, k=0.05, *, dims=None) -> ImgArray:
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
        return self.parallel(corner_harris_, complement_axes(dims, self.axes), k, sigma)
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def find_corners(self, sigma:float=1, k:float=0.05, *, dims=None) -> ImgArray:
        """
        Corner detection using Harris response.

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
        MarkerFrame
            Coordinates of corners. For details see `corner_peaks` method.
        """        
        res = self.gaussian_filter(sigma=1).corner_harris(sigma=sigma, k=k, dims=dims)
        out = res.corner_peaks(min_distance=3, percentile=97, dims=dims)
        return out
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def voronoi(self, coords, *, inf=None, dims="yx") -> ImgArray:
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
    def flood(self, seeds, *, connectivity:int=1, tolerance:float=None, dims=None):
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
        # TODO:check zcyx-image
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
                fill_area = skmorph.flood(self.value[sl], crd, connectivity=connectivity, tolerance=tolerance)
                labels[sl][fill_area] = n_label
        
        self.labels = Label(labels, name=self.name, axes=self.axes, dirpath=self.dirpath).optimize()
        self.labels.set_scale(self)
        return self
    
    @dims_to_spatial_axes
    def refine_sm(self, coords=None, radius:float=4, *, percentile=90, n_iter=10, sigma=1.5, dims=None):
        """
        Refine coordinates of peaks and calculate positional errors using `trackpy`'s functions. Mean
        and noise level are determined using original method.

        Parameters
        ----------
        coords : MarkerFrame or (N, D) array, optional
            Coordinates of peaks. If None, this will be determined by `find_sm`.
        radius : float, default is 4.
            Range to mask single molecules.
        percentile : int, default is 90
            Passed to peak_local_max()
        n_iter : int, default is 10
            Number of iteration of refinement.
        sigma : float, default is 1.5
            Expected standard deviation of particles.
        dims : int or str, optional
            Dimension of axes.

        Returns
        -------
        FrameDict
            Coordinates in MarkerFrame and refinement results in pd.DataFrame.
        """        
        if coords is None:
            coords = self.find_sm(sigma=sigma, dims=dims, percentile=percentile, exclude_border=radius)
        else:
            coords = _check_coordinates(coords, self, dims=self.axes)
        
        if hasattr(self, "labels"):
            labels_now = self.labels.copy()
        else:
            labels_now = None
        self.labels = None
        self.specify(coords, radius, labeltype="circle")
        
        # set parameters
        radius = check_nd(radius, len(dims))
        sigma = tuple(map(int, check_nd(sigma, len(dims))))
        sigma = tuple([int(x) for x in sigma])
        
        df_all = pd.DataFrame()
        c_axes = complement_axes(dims, self.axes)
        c_axes_list = [a for a in c_axes]
        dims_list = [a for a in dims]
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
                df_all = pd.concat([df_all, refined_coords])
                
        mf = MarkerFrame(df_all.reindex(columns=[a for a in self.axes]), columns=str(self.axes))
        mf.set_scale(self.scale)
        df = df_all[df_all.columns[df_all.columns.isin([a for a in df_all.columns if a not in dims])]]
        if labels_now is not None:
            self.labels = labels_now
        return FrameDict(coords=mf, results=df)
        
    
    @dims_to_spatial_axes
    def find_sm(self, sigma:float=1.5, *, method="dog", percentile:float=95, topn:int=np.inf, 
                exclude_border=True, dims=None) -> MarkerFrame:
        """
        Single molecule detection using difference of Gaussian, determinant of hessian or
        laplacian of Gaussian method.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of puncta.
        method : str, default is "dog"
            Which filter is used prior to finding local maxima. Currently supports "dog", "doh" 
            and "log".
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
        >>> ip.window.add(img)
        >>> ip.window.add(lnk)
        """        
        methods_ = {"dog": "dog_filter",
                    "doh": "doh_filter",
                    "log": "log_filter",
                    }
        try:
            fil_img = getattr(self, methods_[method.lower()])(sigma, dims=dims)
        except KeyError:
            raise ValueError(f"Currently `method` only supports {', '.join(methods_.keys())}")
        
        coords = fil_img.peak_local_max(min_distance=sigma*2, percentile=percentile, 
                                        topn=topn, dims=dims, exclude_border=exclude_border)
        return coords
    
        
    @dims_to_spatial_axes
    def centroid_sm(self, coords=None, radius:float=4, sigma:float=1.5, filt=None,
                    percentile:float=90, *, dims=None) -> MarkerFrame:
        """
        Calculate positions of particles in subpixel precision using centroid.

        Parameters
        ----------
        coords : MarkerFrame or (N, 2) array, optional
            Coordinates of peaks. If None, this will be determined by find_sm.
        radius : float, default is 4.
            Range to calculate centroids. Rectangular image with size 2r+1 x 2r+1 will be send 
            to calculate moments.
        sigma : float, default is 1.5
            Expected standard deviation of particles.
        filt : callable, optional
            For every slice `sl`, label is added only when filt(`input`) == True is satisfied.
        percentile, dims
            Passed to peak_local_max()
        dims : int or str, optional
            Dimension of axes.
        
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
        radius = np.asarray(check_nd(radius, ndim))
        shape = self.sizesof(dims)
        
        centroids = []  # fitting results of means
        with Progress("centroid_sm"):
            for crd in coords.values:
                center = tuple(crd[-ndim:])
                label_sl = tuple(crd[:-ndim])
                sl = specify_one(center, radius, shape) # sl = (..., z,y,x)
                input_img = self.value[label_sl][sl]
                if input_img.size == 0 or not filt(input_img):
                    continue
                
                mom = skmes.moments(input_img, order=1)
                shift = center - radius
                centroid = np.array([mom[(0,)*i + (1,) + (0,)*(ndim-i-1)] for i in range(ndim)])/mom[(0,)*ndim]
                centroids.append(label_sl + tuple(centroid + shift))
                
        out = MarkerFrame(centroids, columns=coords.col_axes, dtype=np.float32).as_standard_type()
        out.set_scale(coords.scale)

        return out
    
    @dims_to_spatial_axes
    def gauss_sm(self, coords=None, radius:float=4, sigma:float=1.5, filt=None,
                 percentile:float=95, *, return_all=False, dims=None) -> MarkerFrame|FrameDict:
        """
        Calculate positions of particles in subpixel precision using Gaussian fitting.

        Parameters
        ----------
        coords : MarkerFrame or (N, 2) array, optional
            Coordinates of peaks. If None, this will be determined by find_sm.
        radius : float, default is 4.
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
            Dimension of axes.

        Returns
        -------
        MarkerFrame, if return_all == False
            Gaussian centers.
        FrameDict with keys {means, sigmas, errors}, if return_all == True
            Dictionary that contains means, standard deviations and fitting errors.
        """        
        
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
                sl = specify_one(center, radius, shape) # sl = (..., z,y,x)
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
    def edge_grad(self, sigma:float=1.0, method="scharr", *, deg=False, dims="yx") -> PhaseArray:
        """
        Calculate gradient direction using horizontal and vertical edge operation. Gradient direction
        is the direction with maximum gradient, i.e., intensity increase is largest.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of Gaussian prefilter, by default 1.0
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
        """        
        # Get operator
        method_dict = {"sobel": (sobel_h_, sobel_v_),
                       "farid": (farid_h_, farid_v_),
                       "scharr": (scharr_h_, scharr_v_),
                       "prewitt": (prewitt_h_, prewitt_v_)}
        try:
            op_h, op_v = method_dict[method]
        except KeyError:
            raise ValueError("`method` must be 'sobel', 'farid' 'scharr', or 'prewitt'.")
        
        # Start
        c_axes = complement_axes(dims, self.axes)
        if sigma > 0:
            self = self.gaussian_filter(sigma, dims=dims)
        grad_h = self.parallel(op_h, c_axes)
        grad_v = self.parallel(op_v, c_axes)
        grad = np.arctan2(-grad_h, grad_v)
        
        grad = PhaseArray(grad, border=(-np.pi, np.pi))
        grad.fix_border()
        deg and grad.rad2deg()
        return grad
    
    @record()
    def hessian_angle(self, sigma:float=1., *, deg=False, dims="yx") -> PhaseArray:
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
    def gabor_angle(self, n_sample=180, lmd:float=5, sigma:float=2.5, gamma=1, phi=0, *, deg=False, 
                    dims="yx") -> PhaseArray:
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
            out_ = self.as_float().parallel(gabor_real_, c_axes, ker)
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
    def gabor_filter(self, lmd:float=5, theta:float=0, sigma:float=2.5, gamma=1, phi=0, *, return_imag=False,
                     dims="yx") -> ImgArray:
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
        ker = skfil.gabor_kernel(1/lmd, theta, 0, sigma, sigma/gamma, 3, phi).astype(np.complex64)
        if return_imag:
            out = self.as_float().parallel(gabor_, complement_axes(dims, self.axes), ker, outdtype=np.complex64)
        else:
            out = self.as_float().parallel(gabor_real_, complement_axes(dims, self.axes), ker, outdtype=np.float32)
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
        freq = fft(self.value.astype(np.float32), shape=self.sizesof(dims), 
                   axes=[self.axisof(a) for a in dims])
        out = np.fft.fftshift(freq)
        return out
    
    @dims_to_spatial_axes
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
        
    
    @dims_to_spatial_axes
    @only_binary
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
        return self.parallel(distance_transform_edt_, complement_axes(dims, self.axes))
    
    @dims_to_spatial_axes
    @only_binary
    @record()
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
    @only_binary
    @record()
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
    @only_binary
    @record()
    def convex_hull(self, *, dims=None, update=False):
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
        return self.parallel(convex_hull_, complement_axes(dims, self.axes), outdtype=bool)
        
    @dims_to_spatial_axes
    @only_binary
    @record()
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
        
        return self.parallel(skeletonize_, complement_axes(dims, self.axes), selem, outdtype=bool)
        
    
    @dims_to_spatial_axes
    @only_binary
    @record()
    def count_neighbors(self, *, connectivity=None, mask=True, dims=None) -> ImgArray:
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
            Dimension of axes.

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
        out = self.as_uint8().parallel(population_, complement_axes(dims, self.axes), selem)
        if mask:
            out[~self.value] = 0
            
        return out.astype(np.uint8)
    
    @dims_to_spatial_axes
    @only_binary
    @record()
    def remove_skeleton_structure(self, structure="tip", *, connectivity=None,
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
    def pointprops(self, coords, *, order:int=1, squeeze:bool=True) -> PropArray:
        """
        Measure interpolated intensity at points with float coordinates.

        Parameters
        ----------
        coords : MarkerFrame or array-like
            Coordinates of point to be measured.
        order : int, default is 1
            Spline interpolation order.
        squeeze : bool, default is True
            If True and only one point is measured, the redundant dimension "p" will be deleted.

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
                        axes="p"+prop_axes, dirpath=self.dirpath,
                        propname = f"pointprops", dtype=np.float32)
        
        for sl, img in self.iter(prop_axes, exclude=col_axes):
            out[(slice(None),)+sl] = ndi.map_coordinates(img, coords, prefilter=order > 1,
                                                order=order, mode="reflect")
        if l == 1 and squeeze:
            out = out[0]
        return out
    
    @record(append_history=False)
    def lineprops(self, src, dst, func="mean", *, order:int=1, squeeze:bool=True) -> PropArray:
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
            If True and only one line is measured, the redundant dimension "p" will be deleted.

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
                        axes="p"+prop_axes, dirpath=self.dirpath,
                        propname = f"lineprops<{func.__name__}>", dtype=np.float32)
        
        for i, (s, d) in enumerate(zip(src.values, dst.values)):
            resliced = self.reslice(s, d, order=order)
            out[i] = np.apply_along_axis(func, axis=-1, arr=resliced)
        
        if l == 1 and squeeze:
            out = out[0]
        
        return out
    
    @dims_to_spatial_axes
    @need_labels
    @record(record_label=True)
    def watershed(self, coords:MarkerFrame=None, *, connectivity:int=1, input:str="distance", 
                  min_distance:float=2, dims=None) -> ImgArray:
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
            input_img = input_img.astype(np.uint8)
            
        input_img._view_labels(self)
        
        if coords is None:
            coords = input_img.peak_local_max(min_distance=min_distance, dims=dims)
        
        labels = largest_zeros(input_img.shape)
        shape = self.sizesof(dims)
        n_labels = 0
        c_axes = complement_axes(dims, self.axes)
        markers = np.zeros(shape, dtype=labels.dtype) # placeholder for maxima
        
        for (sl, img), (_, crd) in zip(input_img.iter(c_axes, israw=True),
                                       coords.groupby([a for a in c_axes])):
            # crd.values is (N, 2) array so tuple(crd.values.T.tolist()) is two (N,) list.
            crd = crd.values.T.tolist()
            markers[tuple(crd)] = np.arange(1, len(crd[0])+1, dtype=labels.dtype)
            labels[sl] = skseg.watershed(-img.value, markers, 
                                        mask=img.labels.value, 
                                        connectivity=connectivity)
            labels[sl][labels[sl]>0] += n_labels
            n_labels = labels[sl].max()
            markers[:] = 0 # reset placeholder
        
        labels = labels.view(Label)
        self.labels = labels.optimize()
        self.labels._set_info(self)
        self.labels.set_scale(self)
        return self
    
    @dims_to_spatial_axes
    @need_labels
    @record(record_label=True)
    def random_walker(self, beta=130, mode="cg_j", tol=1e-3, *, dims=None) -> ImgArray:
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
            img.labels[:] = skseg.random_walker(img, img.labels, beta=beta, mode=mode, tol=tol)
        
        return self
    
    # @dims_to_spatial_axes
    # @record(append_history=False)
    # def slic(self, n_segments=100, *, compactness=10.0, max_iter=10, sigma=1, multichannel=False,
    #          min_size_factor=0.5, max_size_factor=3, mask=None, dims=None):
    #     # multichannel not working, needs sort_axes
    #     # issue: slic returns a strange label with grayscale images.
    #     if multichannel:
    #         c_axes = complement_axes("c"+dims, self.axes)
    #         labels = largest_zeros(self["c=0"].shape)
    #         exclude = "c"
    #     else:
    #         c_axes = complement_axes(dims, self.axes)
    #         labels = largest_zeros(self.shape)
    #         exclude = ""
        
    #     for sl, img in self.iter(c_axes, exclude=exclude):
    #         plt.imshow(img)
    #         labels[sl] = \
    #         skseg.slic(img, n_segments=n_segments, compactness=compactness, max_iter=max_iter,
    #                    sigma=sigma, multichannel=multichannel, min_size_factor=min_size_factor,
    #                    max_size_factor=max_size_factor, start_label=1, mask=mask)
        
    #     self.labels = labels.view(Label).optimize()
    #     self.labels._set_info(self, "slic")
    #     self.labels.set_scale(self)
    #     return self
    
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
        labels = self.threshold(thr=thr, dims=None, **kwargs)
        return self.label(labels, dims=dims)
    
        
    @need_labels
    @record(append_history=False)
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
        
        if isinstance(properties, str):
            properties = (properties,)
        if extra_properties is not None:
            properties = properties + tuple(ex.__name__ for ex in extra_properties)

        if "p" in self.axes:
            # this dimension will be label
            raise ValueError("axis 'p' is forbidden in regionprops().")
        
        prop_axes = complement_axes(self.labels.axes, self.axes)
        shape = self.sizesof(prop_axes)
        
        out = ArrayDict({p: PropArray(np.empty((self.labels.max(),) + shape, dtype=np.float32),
                                      name=self.name, 
                                      axes="p"+prop_axes,
                                      dirpath=self.dirpath,
                                      propname=p)
                         for p in properties})
        
        # calculate property value for each slice
        for sl, img in self.iter(prop_axes, exclude=self.labels.axes):
            props = skmes.regionprops(self.labels, img, cache=False,
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
        return self.parallel(lbp_, complement_axes(dims), p, radius, method)
    
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
        self, bins, rescale_max = check_glcm(self, bins, rescale_max)
            
        c_axes = complement_axes(dims, self.axes)
        outshape = self.sizesof(c_axes) + (bins, bins, len(distances), len(angles))
        out = self.parallel(glcm_, c_axes, distances, angles, bins, outshape=outshape, outdtype=np.uint32)
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
    def proj(self, axis=None, method="mean") -> ImgArray:
        """
        Z-projection along any axis.

        Parameters
        ----------
        axis : str, optional
            Along which axis projection will be calculated. If None, most plausible one will be chosen.
        method : str or callable, default is mean-projection.
            Projection method. If str is given, it will converted to numpy function.

        Returns
        -------
        ImgArray
            Projected image.
        """        
        func = _check_function(method)
        if axis is None:
            axis = find_first_appeared(self.axes, exclude="yx")
        axisint = self.axisof(axis)
        out = func(self.value, axis=axisint).view(self.__class__)
        out._set_info(self, f"proj(axis={axis}, method={method})", del_axis(self.axes, axisint))
        return out

    @record()
    def clip(self, in_range=("0%", "100%")) -> ImgArray:
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
        lowerlim, upperlim = check_clip_range(in_range, self.value)
        out = np.clip(self.value, lowerlim, upperlim)
        out = out.view(self.__class__)
        out.temp = [lowerlim, upperlim]
        return out
    
    @record()
    def rescale_intensity(self, in_range=("0%", "100%"), dtype=np.uint16) -> ImgArray:
        """
        Rescale the intensity of the image using skimage.exposure.rescale_intensity().

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
        out = self.view(np.ndarray).astype(np.float32)
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
        axis : str, default is "t"
            Along which axis drift will be calculated.
        show_drift : bool, default is True
            If True, plot the result.

        Returns
        -------
        MarkerFrame
            DataFrame structure with x,y columns
        """        
        if self.ndim != 3:
            raise TypeError(f"input must be three dimensional, but got {self.shape}")

        # slow drift needs large upsampling numbers
        corr_kwargs = {"upsample_factor": 10}
        corr_kwargs.update(kwargs)
        
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
    def drift_correction(self, shift=None, ref:ImgArray=None, *, order:int=1, 
                         along:str="t", dims=None, update:bool=False):
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
        along : str, default is "t"
            Along which axis drift will be corrected.
        dims : str, optional
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
        
        elif isinstance(shift, MarkerFrame):
            if len(shift) != self.sizeof("t"):
                raise ValueError("Wrong shape of 'shift'.")
        
        else:
            shift = MarkerFrame(shift, columns="yx", dtype=np.float32)
            return self.drift_correction(shift, ref, order=order, along=along, 
                                         dims=dims,update=update)

        out = np.empty(self.shape)
        t_index = self.axisof(along)
        shift = shift.reindex(columns=["x", "y"])
        for sl, img in self.iter(complement_axes(dims, self.axes)):
            trans = -shift.loc[sl[t_index]]
            mx = sktrans.AffineTransform(translation=trans)
            out[sl] = sktrans.warp(img.astype(np.float32), mx, order=order)
        
        out = out.view(self.__class__)
        return out

    @dims_to_spatial_axes
    @record(append_history=False)
    def estimate_sigma(self, *, squeeze=True, dims=None) -> ImgArray|float:
        """
        Wavelet-based estimation of Gaussian noise.

        Parameters
        ----------
        squeeze : bool, by default True
            If True and output can be converted to a scalar, then convert it.
        dims : str, optional
            Spatial dimension.

        Returns
        -------
        ImgArray or float
            Estimated standard deviation. sigma["t=0;c=1"] means the estimated value of
            image slice at t=0 and c=1.
        """        
        c_axes = complement_axes(dims, self.axes)
        out = self.parallel(estimate_sigma_, c_axes, outshape=self.sizesof(c_axes))
        if out.ndim == 0 and squeeze:
            out = out[()]
        else:
            out = out.view(self.__class__) # should return PropArray??
            out._set_info(self, f"estimate_sigma(dims={dims})", new_axes=c_axes)
        return out       
        
    
    @dims_to_spatial_axes
    @record()
    def pad(self, pad_width, mode="constant", *, dims=None, **kwargs) -> ImgArray:
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
    def defocus(self, sigma, depth:int, bg:float=None) -> ImgArray:
        """
        Make a z-directional padded image by defocusing the original image. This padding is
        useful when applying FFT to 3D images.
        
        Parameters
        ----------
        sigma : float or array or float
            Standard deviation of Gaussian filter for defocusing.
        depth : int
            Depth of defocusing. For an image with z-axis size L, then output image will have
            size L + 2*depth.
        bg : float, optional
            Background intensity. If not given, it will calculated as the minimum value of 
            the original image.

        Returns
        -------
        ImgArray
            Padded image.
            
        Example
        -------
        depth = 2, radius = 2
        
        ----|   |----| o |--     o ... center of kernel
        ----| o |----|   |--
        ++++|   |++++|___|++  <- the upper edge of original image 
        ++++|___|+++++++++++

        """        
        
        if bg is None:
            bg = self.min()
            
        # convolve psf
        out = self.pad(depth, mode="constant", constant_values=bg, dims="z")
        for sl, img in out.iter(complement_axes("zyx", self.axes), israw=True):
            img[:depth] = ndi.gaussian_filter(img[:depth*2].value, sigma, mode="constant", cval=bg)[:depth]
            img[-depth:] = ndi.gaussian_filter(img[-depth*2:].value, sigma, mode="constant", cval=bg)[-depth:]
            
        return out
    
    @dims_to_spatial_axes
    @record()
    @same_dtype(asfloat=True)
    def wiener(self, psf, lmd, *, dims=None, update:bool=False) -> ImgArray:
        """
        Classical wiener deconvolution. This algorithm has the serious ringing problem.

        Parameters
        ----------
        psf : np.ndarray
            Point spread function
        lmd : float
            Constant value used in the deconvolution. See Formulation below.
        dims : int or str, optional
            Dimension of axes.
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
        
        psf_ft = fft(psf)
        psf_ft_conj = np.conjugate(psf_ft)
        
        return self.parallel(wiener_, complement_axes(dims, self.axes),
                             psf_ft, psf_ft_conj, lmd)
        
    
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
        niters : int, default is 50.
            Number of iteration.
        dims : int or str, optional
            Dimension of axes.
        update : bool, optional
            If update self to filtered image.
        
        Returns
        -------
        ImgArray
            Deconvolved image.
        """
        
        psf = check_psf(self, psf, dims)
        
        psf_ft = fft(psf)
        psf_ft_conj = np.conjugate(psf_ft)
        # start deconvolution
        return self.parallel(richardson_lucy_, complement_axes(dims), 
                             psf_ft, psf_ft_conj, niter)

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
    