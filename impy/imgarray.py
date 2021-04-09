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
from skimage.feature.corner import _symmetric_image
from skimage import feature as skfeat
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft
from .func import get_meta, record, same_dtype, gaussfit, affinefit, circle, del_axis, add_axes, ball_like
from .base import BaseArray
from .axes import Axes
from .proparray import PropArray

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
    return (sl, skfil.gaussian(data, sigma=sigma))

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

def _hessian_eigh(args):
    sl, data, sigma = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    sigma = np.asarray(sigma)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (sigma.reshape(-1,1) * sigma.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return (sl, eigval, eigvec)

def _hessian_eigval(args):
    sl, data, sigma = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    sigma = np.asarray(sigma)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (sigma.reshape(-1,1) * sigma.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return (sl, eigval)


class ImgArray(BaseArray):
    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 history=None, metadata=None, lut=None):
        super().__init__(obj, name=name, axes=axes, dirpath=dirpath, 
                         history=history, metadata=metadata, lut=lut)

    @same_dtype(True)
    @record
    def affine(self, dims:int=2, order:int=1, **kwargs) -> ImgArray:
        """
        Affine transformation
        kwargs: matrix, scale, rotation, shear, translation
        """
        if dims != 2:
            raise ValueError("dims != 2 version have yet been implemented")
        mx = sktrans.AffineTransform(**kwargs)
        out = self.as_uint16().parallel(_affine, dims, mx, order)
        out._set_info(self, f"{dims}D-Affine-Transform")
        return out
    
    @same_dtype(True)
    @record
    def translate(self, dims=2, translation=None) -> ImgArray:
        """
        Simple translation of image, i.e. (x, y) -> (x+dx, y+dy)
        """
        mx = sktrans.AffineTransform(translation=translation)
        out = self.as_uint16().parallel(_affine, dims, mx)
        out._set_info(self, f"Translate{translation}")
        return out

    @same_dtype(True)
    @record
    def rescale(self, scale:float=1/16, axes:str="yx", order:int=None) -> ImgArray:
        """
        Rescale image.

        Parameters
        ----------
        scale : float, optional
            scale of the new image, by default 1/16
        axes : str, optional
            axes to rescale, by default "yx"
        order : float, optional
            order of rescaling, by default None
        """        
        scale_ = [scale if a in axes else 1 for a in self.axes]
        out = sktrans.rescale(self.value, scale_, order=order, anti_aliasing=False).view(self.__class__)
        out._set_info(self, f"Rescale(x1/{np.round(1/scale, 1)})")
        return out
    
    
    @record
    def gaussfit(self, scale:float=1/16, p0=None, show_result:bool=True) -> ImgArray:
        """
        Fit the image to 2-D Gaussian.

        Parameters
        ----------
        scale : float, optional
            Scale of rough image (to speed up fitting).
        p0 : list or None, optional
            Initial parameters, by default None
        show_result : bool, optional
            If show the fitting result.

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
        
        param, fit = gaussfit(self, p0, scale=scale, show_result=show_result)
        out = fit.view(self.__class__)
        out._set_info(self, f"Gaussian-Fit(x1/{np.round(1/scale, 1)})")
        out.temp = param

        return out
    
    @record
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
        out._set_info(self, f"Affine-Correction(order={order})")
        out.temp = mtx
        return out
    
    @record
    def hessian_eigval(self, sigma=1, dims:int=2) -> list[ImgArray]:
        """
        Calculate Hessian's eigenvalues for each image. If dims=2, every yx-image 
        is considered to be a single spatial image, and if dims=3, zyx-image.

        Parameters
        ----------
        sigma : float, optional
            sigma of Gaussian filter applied before calculating Hessian, by default 1.
        dims : 2 or 3, optional
            spatial dimension, by default 2

        Returns
        -------
        list of float ImgArray
            eigenvalues in smaller->larger order
        """        
        # check sigma
        if isinstance(sigma, (int, float)):
            sigma = [sigma] * dims
        elif len(sigma) != dims:
            raise ValueError("length of sigma and dims must match.")
        
        eigval = self.as_float().parallel(_hessian_eigval, dims, sigma, 
                                              outshape=self.shape+(dims,))
        
        eigval = list(np.moveaxis(eigval, -1, 0))
        for i, each in enumerate(eigval):
            each._set_info(self, f"{dims}D-Hessian-eigenvalue[{i}]")
        
        return eigval
    
    @record
    def hessian_eig(self, sigma=1, dims:int=2) -> tuple:
        """
        Calculate Hessian's eigenvalues and eigenvectors.

        Parameters
        ----------
        sigma : int, optional
            sigma of Gaussian filter applied before calculating Hessian, by default 1.
        dims : 2 or 3, optional
            spatial dimension, by default 2

        Returns
        -------
        list of float ImgArray and 2D-list of float ImgArray
        """        
        # check sigma
        if isinstance(sigma, (int, float)):
            sigma = [sigma] * dims
        elif len(sigma) != dims:
            raise ValueError("length of sigma and dims must match.")
        
        eigval = np.empty(self.shape+(dims,), dtype="float32")
        eigvec = np.empty(self.shape+(dims,dims), dtype="float32")
        
        if self.__class__.n_cpu > 1:
            results = self._parallel(_hessian_eigh, dims, sigma)
            for sl, eigval_, eigvec_ in results:
                eigval[sl] = eigval_
                eigvec[sl] = eigvec_
        else:
            for sl, img in self.iter(dims):
                sl, eigval_, eigvec_ = _hessian_eigh((sl, img, dims, sigma))
                eigval[sl] = eigval_
                eigvec[sl] = eigvec_
        
        # set information of eigenvalue list
        eigval = list(np.moveaxis(eigval, -1, 0))
        for i, each in enumerate(eigval):
            each._set_info(self, f"{dims}D-Hessian-eigenvalue[{i}]")
        
        # set information of eigenvector list
        eigvec = list(np.moveaxis(eigvec, -1, 0))
        allaxes = "yx" if dims==2 else "zyx"
        for i, e in enumerate(eigvec):
            eigvec[i] = list(np.moveaxis(e, -1 ,0))
            for j, each in enumerate(eigvec[i]):
                each._set_info(self, f"{dims}D-Hessian-eigenvector[{i}][{allaxes[j]}]")
                
        return eigval, eigvec
    
    def hessian_filter(self, sigma=1):
        # only puncta detection is available now
        # TODO: filament detection etc.
        eigval = self.hessian_eigval(sigma)
        for e in eigval:
            e[e>0] = 0
        return np.product(-eigval, axis=0)**(1/len(eigval))
        
    
    def _running_kernel(self, radius:float, dims:int, function=None, annotation:str="") -> ImgArray:
        disk = ball_like(radius, dims)
        out = self.as_uint16().parallel(function, dims, disk)
        out._set_info(self, annotation)
        return out
    
    @same_dtype()
    @record
    def erosion(self, radius:float=1, dims:int=2) -> ImgArray:
        f = _binary_erosion if self.dtype == bool else _erosion
        return self._running_kernel(radius, dims, f, f"{dims}D-Erosion(R={radius})")
    
    @same_dtype()
    @record
    def dilation(self, radius:float=1, dims:int=2) -> ImgArray:
        f = _binary_dilation if self.dtype == bool else _dilation
        return self._running_kernel(radius, dims, f, f"{dims}D-Dilation(R={radius})")
    
    @same_dtype()
    @record
    def opening(self, radius:float=1, dims:int=2) -> ImgArray:
        f = _binary_opening if self.dtype == bool else _opening
        return self._running_kernel(radius, dims, f, f"{dims}D-Opening(R={radius})")
    
    @same_dtype()
    @record
    def closing(self, radius:float=1, dims:int=2) -> ImgArray:
        f = _binary_closing if self.dtype == bool else _closing
        return self._running_kernel(radius, dims, f, f"{dims}D-Closing(R={radius})")
    
    @same_dtype()
    @record
    def tophat(self, radius:float=50, dims:int=2) -> ImgArray:
        return self._running_kernel(radius, dims, _tophat, f"{dims}D-Top-Hat(R={radius})")
    
    @same_dtype()
    @record
    def mean_filter(self, radius:float=1, dims:int=2) -> ImgArray:
        return self._running_kernel(radius, dims, _mean, f"{dims}D-Mean-Filter(R={radius})")
    
    @same_dtype()
    @record
    def median_filter(self, radius:float=1, dims:int=2) -> ImgArray:
        return self._running_kernel(radius, dims, _median, f"{dims}D-Median-Filter(R={radius})")
    
    @same_dtype()
    @record
    def gaussian_filter(self, sigma:float=1, dims:int=2) -> ImgArray:
        """
        Run Gaussian filter (Gaussian blur).
        Parameters
        ----------
        sigma : scalar or array of scalars, optional
            standard deviation(s) of Gaussian, by default 1
        dims : int, optional
            Dimension of axes, i.e. xy or xyz, by default 2
        Returns
        -------
        ImgArray
            Filtered image.
        """        
        out = self.parallel(_gaussian, dims, sigma)
        out._set_info(self, f"{dims}D-Gaussian-Filter(sigma={sigma})")
        return out
    
    @same_dtype()
    @record
    def rolling_ball(self, radius:float=50, smoothing:bool=True) -> ImgArray:
        """
        Subtract Background using rolling-ball algorithm.

        Parameters
        ----------
        radius : int, optional
            Radius of rolling ball, by default 50
        smoothing : bool, optional
            If apply 3x3 averaging before creating background.
            
        Returns
        -------
        ImgArray
            Background subtracted image.
        """        
        out = self.as_uint16().parallel(_rolling_ball, "ptzc", radius, smoothing)
        out._set_info(self, f"Rolling-Ball(R={radius})")
        return out
    
    
    @record
    def fft(self) -> ImgArray:
        """
        Fast Fourier transformation.
        This function returns complex array. Inconpatible with some functions here.
        """
        freq = fft(self.value.astype("float32"))
        out = np.fft.fftshift(freq).view(self.__class__)
        out._set_info(self, "FFT")
        return out
    
    @record
    def ifft(self) -> ImgArray:
        freq = np.fft.fftshift(self.value)
        out = np.real(ifft(freq)).view(self.__class__)
        out._set_info(self, "IFFT")
        return out
    
    @record
    def threshold(self, thr=None, method:str="otsu", light_bg:bool=False, 
                  iters:str="c", **kwargs) -> ImgArray:
        """
        Parameters
        ----------
        thr: int or array or None, optional
            Threshold value. If None, this will be determined by 'method'.
        method: str, optional
            Thresholding algorithm.
        light_bg: bool, default is False
            If background is brighter
        iters: str, default is 'c'
            Around which axes images will be iterated.
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
        if thr is None:
            method = method.lower()
            try:
                func = methods_[method]
            except KeyError:
                s = ", ".join(list(methods_.keys()))
                raise KeyError(f"{method}\nmethod must be: {s}")
            thr = func(self.view(np.ndarray), **kwargs)

            out = np.zeros(self.shape, dtype=bool)
            for t, img in self.iter(iters):
                if light_bg:
                    out[t] = img <= thr
                else:
                    out[t] = img >= thr
            out = out.view(self.__class__)
            out._set_info(self, f"Thresholding({method})")

        else:
            if light_bg:
                out = self <= thr
            else:
                out = self >= thr
            out._set_info(self, f"Thresholding({thr:.1f})")
        
        out.temp = thr
        return out
        
        
    def crop_circle(self, radius=None, outzero=True):
        """
        Make a circular window function.
        """
        if radius is None:
            radius = np.min(self.xyshape()) // 2
        
        circ = circle(radius, self.xyshape())
        if not outzero:
            circ = ~circ
        circ = add_axes(self.axes, self.shape, circ)
        out = np.array(self)
        out[~circ] = 0
        out = out.view(self.__class__)
        out._set_info(self, f"Crop-Circle(R={radius}, {'outzero' if outzero else 'inzero'})")
        return out
    
    def specify(self, xy:tuple[int], dxdy:tuple[int], position="corner"):
        """
        Make a rectancge ROI.
        """
        x, y = xy
        dx, dy = dxdy
        
        if position == "corner":
            pass
        elif position == "center":
            x -= dx//2
            y -= dy//2
        else:
            raise ValueError("'position' must be 'corner' or 'center'")
        
        if hasattr(self, "labels"):
            self.labels[y:y+dy, x:x+dx] = self.labels.max() + 1
        else:
            labels = np.zeros((self.sizeof("y"), self.sizeof("x")), dtype="uint8")
            labels[y:y+dy, x:x+dx] = 1
            self.labels = labels
        
        return self.labels

    
    def crop_center(self, scale:float=0.5):
        """
        Crop out the center of an image.
        e.g. when scale=0.5, create 512x512 image from 1024x1024 image.
        """
        if scale <= 0 or 1 < scale:
            raise ValueError(f"scale must be (0, 1], but got {scale}")
        
        sizex, sizey = self.xyshape()
        
        x0 = int(sizex / 2 * (1 - scale)) + 1
        x1 = int(sizex / 2 * (1 + scale))
        y0 = int(sizey / 2 * (1 - scale)) + 1
        y1 = int(sizey / 2 * (1 + scale))

        out = self[f"x={x0}-{x1};y={y0}-{y1}"]
        out.history[-1] = f"Crop-Center(scale={scale})"
        
        return out
    
    def label(self, label_image=None, connectivity=None):
        """
        Run skimage's label() and store the results as attribute.

        Parameters
        ----------
        label_image : array, optional
            image to make label, by default self is used.
        connectivity : int, optional
            passed to skimage's label(), by default None

        Returns
        -------
        labeled image
        """        
        # check the shape of label_image
        if label_image is None:
            label_image = self
            
        elif not isinstance(label_image, np.ndarray):
            raise TypeError("label_image must be an array")
        
        elif label_image.ndim == 2:
            if label_image.shape != self.xyshape():
                raise ValueError("Incompatible yx-shape")
        elif label_image.ndim == 3:
            zyxshape = tuple(self.sizeof(a) for a in "zyx")
            if label_image.shape != zyxshape:
                raise ValueError("Incompatible zyx-shape")
        else:
            raise ValueError("'label_image' must be 2 or 3 dimensional")
            
        labels, nlabel = skmes.label(label_image, background=0, 
                                     return_num=True, connectivity=connectivity)
        
        # use memory efficient dtype
        if nlabel < 256:
            labels = labels.astype("uint8")
        elif nlabel < 65536:
            labels = labels.astype("uint16")
        
        self.labels = labels
        
        return labels
    
    def regionprops(self, properties=("mean_intensity", "area"), extra_properties=None) -> dict[str, ImgArray]:
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
        dict of PropArray
        """        
        if not hasattr(self, "labels"):
            raise AttributeError("Use label() to add labels to the image.")
        
        if isinstance(properties, str):
            properties = (properties,)

        if "p" in self.axes:
            # this dimension will be label
            raise ValueError("axis 'p' is forbidden.")
        
        if self.labels.ndim == 2:
            axes = "tzc"
        elif self.labels.ndim == 3:
            axes = "tc"
        else:
            raise ValueError("'label_image' must be 2 or 3 dimensional")
        
        prop_axes = "".join([a for a in axes if a in self.axes])
        shape = tuple(self.sizeof(a) for a in prop_axes)
        out = {p: PropArray(np.zeros((self.labels.max(),) + shape, dtype="float32"),
                            name=self.name, 
                            axes="p" + prop_axes,
                            dirpath=self.dirpath,
                            propname = p)
               for p in properties}
        
        # calculate property value for each slice
        for sl in itertools.product(*map(range, shape)):
            props = skmes.regionprops(self.labels, self.value[sl], 
                                      cache=False,
                                      extra_properties=extra_properties)
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]
        
        return out
    
    @record
    def split(self, axis=None) -> list[ImgArray]:
        """
        Split n-dimensional image into (n-1)-dimensional images.

        Parameters
        ----------
        axis : str or int, optional
            Along which axis the original image will be split, by default "c"

        Returns
        -------
        list of ImgArray
            Separate images
        """
        # determine axis in int.
        if axis is None:
            axisint = 0
        else:
            axisint = self.axisof(axis)
            
        imgs = list(np.moveaxis(self, axisint, 0))
        for i, img in enumerate(imgs):
            img.history[-1] = f"Split(axis={axis})"
            img.axes = del_axis(self.axes, axisint)
            if axis == "c" and self.lut is not None:
                img.lut = [self.lut[i]]
            else:
                img.lut = None
        return imgs

    @same_dtype()
    @record
    def proj(self, axis="z", method="mean") -> ImgArray:
        """
        Z-projection.
        'method' must be in func_dict.keys() or some function like np.mean.
        """
        func_dict = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max, "median": np.median}
        if method in func_dict.keys():
            func = func_dict[method]
        elif callable(method):
            func = method
        else:
            raise TypeError(f"'method' must be one of {', '.join(list(func_dict.keys()))} or callable object.")
        
        axisint = self.axisof(axis)
        out = func(self.value, axis=axisint).view(self.__class__)
        out._set_info(self, f"{method}-Projection(axis={axis})", del_axis(self.axes, axisint))
        return out

    
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
        lower, upper = in_range
        if isinstance(lower, str) and lower.endswith("%"):
            lower = float(lower[:-1])
            lowerlim = np.percentile(self, lower)
        elif lower is None:
            lowerlim = np.min(self)
        else:
            lowerlim = float(lower)
        
        if isinstance(upper, str) and upper.endswith("%"):
            upper = float(upper[:-1])
            upperlim = np.percentile(self, upper)
        elif upper is None:
            upperlim = np.max(self)
        else:
            upperlim = float(lower)
        
        if lowerlim >= upperlim:
            raise ValueError(f"lowerlim is larger than upperlim: {lowerlim} >= {upperlim}")
        out = np.clip(self.value, lowerlim, upperlim)
        out = out.view(self.__class__)
        out._set_info(self, f"Clip-Outliers({lower:.2f}%-{upper:.2f}%)")
        out.temp = [lowerlim, upperlim]
        return out
        
        
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
        lower, upper = in_range
        if isinstance(lower, str) and lower.endswith("%"):
            lower = float(lower[:-1])
            lowerlim = np.percentile(out, lower)
        elif lower is None:
            lowerlim = np.min(out)
        else:
            lowerlim = float(lower)
        
        if isinstance(upper, str) and upper.endswith("%"):
            upper = float(upper[:-1])
            upperlim = np.percentile(out, upper)
        elif upper is None:
            upperlim = np.max(out)
        else:
            upperlim = float(lower)
        
        if lowerlim >= upperlim:
            raise ValueError(f"lowerlim is larger than upperlim: {lowerlim} >= {upperlim}")
            
        out = skexp.rescale_intensity(out, in_range=(lowerlim, upperlim), out_range=dtype)
        
        out = out.view(self.__class__)
        out._set_info(self, f"Rescale-Intensity({lower:.2f}%-{upper:.2f}%)")
        out.temp = [lowerlim, upperlim]
        return out

        

# non-member functions.

def array(arr, dtype="uint16", name=None, axes=None, lut=None) -> ImgArray:
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

def zeros(shape, dtype="uint16", name=None, axes=None, lut=None) -> ImgArray:
    return array(np.zeros(shape, dtype=dtype), dtype=dtype, name=name, axes=axes, lut=lut)

def imread(path:str, dtype:str="uint16", axes=None, lut=None) -> ImgArray:
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
    # read tif metadata
    if fext == ".tif":
        meta = get_meta(path)
    else:
        meta = {"axes":axes, "ijmeta":{}, "history":[]}
    
    img = io.imread(path)
    
    dirpath = os.path.dirname(path)
    
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
    
    if (self.axes.is_none()):
        return self
    else:
        return self.sort_axes().as_img_type(dtype) # arrange in ptzcyx-order

def imread_collection(dirname:str, axis:str="p", ext:str="tif", 
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


