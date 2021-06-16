from __future__ import annotations
import numpy as np
import multiprocessing as multi
import matplotlib.pyplot as plt
import pandas as pd
import os
import inspect
from skimage.color import label2rgb
from tifffile import imwrite
from ..axes import ImageAxesError
from ..func import *
from ..deco import *
from ..utilcls import *
from ._process import label_
from .bases import HistoryArray
from .label import Label
from .specials import *
from ._skimage import *
from .._types import *

class LabeledArray(HistoryArray):
    n_cpu = 4
        
    @property
    def range(self):
        return self.min(), self.max()
    
    def set_scale(self, other=None, **kwargs) -> None:
        super().set_scale(other, **kwargs)
        if hasattr(self, "labels"):
            self.labels.set_scale(other, **kwargs)
        return None
        
    def __repr__(self):
        if hasattr(self, "labels"):
            labels_shape_info = self.labels.shape_info
        else:
            labels_shape_info = "No label"
            
        return f"\n"\
               f"    shape     : {self.shape_info}\n"\
               f"  label shape : {labels_shape_info}\n"\
               f"    dtype     : {self.dtype}\n"\
               f"  directory   : {self.dirpath}\n"\
               f"original image: {self.name}\n"\
               f"   history    : {'->'.join(self.history)}\n"
    
    
    def imsave(self, tifname:str, dtype=None):
        """
        Save image at the same directory as the original image by default. If the image contains
        wrong axes for ImageJ (= except for tzcyx), then it will converted automatically if possible.
        zyx-scale is also saved.

        Parameters
        ----------
        tifname : str
            File name.
        dtype : Any that can be interpreted as numpy.dtype, optional
            In what data type img will be saved.
        """        
        # set default values
        if not tifname.endswith(".tif"):
            tifname += ".tif"
        if os.sep not in tifname:
            tifname = os.path.join(self.dirpath, tifname)
        if self.metadata is None:
            self.metadata = {}
        if dtype is None:
            dtype = self.dtype
            
        # change axes if needed
        rest_axes = complement_axes(self.axes, "tzcyx")
        new_axes = ""
        axes_changed = False
        for a in self.axes:
            if a in "tzcyx":
                new_axes += a
            else:
                if len(rest_axes) == 0:
                    raise ImageAxesError(f"Cannot save image with axes {self.axes}")
                new_axes += rest_axes[0]
                rest_axes = rest_axes[1:]
                axes_changed = True
        
        # make a copy of the image for saving
        img = self.__class__(self.as_img_type(dtype).value, axes=new_axes)
        img = img.sort_axes()
        
        metadata = self.metadata.copy()
        metadata.update({"min":np.percentile(self, 1), 
                         "max":np.percentile(self, 99)})
        # set lateral scale
        try:
            res = (1/self.scale["x"], 1/self.scale["y"])
        except Exception:
            res = None
        # set z-scale
        if "z" in self.axes:
            metadata["spacing"] = self.scale["z"]
        # add history to Info
        try:
            info = load_json(metadata["Info"])
        except:
            info = {}
        info["impyhist"] = "->".join([self.name] + self.history)
        metadata["Info"] = str(info)
        # set axes in tiff metadata
        metadata["axes"] = str(img.axes).upper()
        if img.ndim > 3:
            metadata["hyperstack"] = True
        # save image
        imwrite(tifname, img, imagej=True, resolution=res, metadata=metadata)
        # notifications
        if axes_changed:
            if new_axes == str(img.axes):
                print(f"Image axes changed: {self.axes} -> {new_axes}")
            else:
                print(f"Image axes changed and sorted: {self.axes} -> {new_axes} -> {img.axes}")
        print(f"Succesfully saved: {tifname}")
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Basic Functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._view_labels(obj)
    
    def _view_labels(self, other):
        """
        Make a view of label **if possible**.
        """
        if hasattr(other, "labels") and axes_included(self, other.labels) and shape_match(self, other.labels):
            self.labels = other.labels
    
    def _getitem_additional_set_info(self, other, **kwargs):
        super()._getitem_additional_set_info(other, **kwargs)
        key = kwargs["key"]
        if other.axes and hasattr(other, "labels") and not isinstance(key, np.ndarray):
            if isinstance(key, tuple):
                _keys = key
            else:
                _keys = (key,)
            label_sl = [_keys[i] for i, a in enumerate(other.axes) 
                        if a in other.labels.axes and i < len(_keys)]
                    
            if len(label_sl) == 0 or len(label_sl) > other.labels.ndim:
                label_sl = (slice(None),)
            try:
                self.labels = other.labels[tuple(label_sl)]
            except IndexError as e:
                print("`labels` was not inherited due to IndexError :", e)
        
        return None

    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        super()._set_info(other, next_history, new_axes)
        
        # inherit labels
        self._view_labels(other)
        return None
    
    def _update(self, out):
        self.value[:] = out.as_img_type(self.dtype).value[:]
        self.history.append(out.history[-1])
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Type Conversions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def as_uint8(self) -> LabeledArray:
        if self.dtype == np.uint8:
            return self
        
        if self.dtype == np.uint16:
            out = self.value / 256
        elif self.dtype == bool:
            out = self.value
        elif self.dtype.kind == "f":
            if 0 <= self.min() and self.max() < 1:
                out = self.value * 256
            else:
                out = self.value + 0.5
            out = np.clip(out, 0, 255)
        else:
            raise TypeError(f"invalid data type: {self.dtype}")
        out = out.astype(np.uint8)
        out = out.view(self.__class__)
        out._set_info(self)
        return out


    def as_uint16(self) -> LabeledArray:
        if self.dtype == np.uint16:
            return self
        if self.dtype == np.uint8:
            out = self.value * 256
        elif self.dtype == bool:
            out = self.value
        elif self.dtype.kind == "f":
            if 0 <= self.min() and self.max() < 1:
                out = self.value * 65535
            else:
                out = self.value + 0.5
            out = np.clip(out, 0, 65535)
            
        else:
            raise TypeError(f"invalid data type: {self.dtype}")
        out = out.astype(np.uint16)
        out = out.view(self.__class__)
        out._set_info(self)
        return out
    
    def as_float(self) -> LabeledArray:
        out = self.value.astype(np.float32).view(self.__class__)
        out._set_info(self)
        return out
        
    
    def as_img_type(self, dtype=np.uint16) -> LabeledArray:
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        elif dtype == "uint16":
            return self.as_uint16()
        elif dtype == "uint8":
            return self.as_uint8()
        elif dtype == "float32":
            return self.as_float()
        elif dtype == "bool":
            return self.astype("bool")
        else:
            raise ValueError(f"dtype: {dtype}")
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Simple Visualizations
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def hist(self, contrast=None, newfig=True):
        """
        Show intensity profile.
        """
        if newfig:
            plt.figure(figsize=(4, 1.7))

        nbin = min(int(np.sqrt(self.size / 3)), 256)
        d = self.astype(np.uint8).ravel() if self.dtype==bool else self.ravel()
        y, x = skexp.histogram(d, nbins=nbin)
        plt.plot(x, y, color="gray")
        plt.fill_between(x, y, np.zeros(len(y)), facecolor="gray", alpha=0.4)
        
        if contrast is None:
            contrast = [self.min(), self.max()]
        x0, x1 = contrast
        
        plt.xlim(x0, x1)
        plt.ylim(0, None)
        plt.yticks([])
        
        return None

    def imshow(self, dims="yx", **kwargs):
        if self.ndim == 1:
            plt.plot(self.value)
        elif self.ndim == 2:
            vmax, vmin = determine_range(self)
            interpol = "bilinear" if self.dtype == bool else "none"
            imshow_kwargs = {"cmap": "gray", "vmax": vmax, "vmin": vmin, "interpolation": interpol}
            imshow_kwargs.update(kwargs)
            plt.imshow(self.value, **imshow_kwargs)
            
            self.hist()
            
        elif self.ndim == 3:
            if "c" not in self.axes:
                imglist = self.split(axis=find_first_appeared(self.axes, exclude=dims))
                if len(imglist) > 24:
                    print("Too many images. First 24 images are shown.")
                    imglist = imglist[:24]

                vmax, vmin = determine_range(self)

                interpol = "bilinear" if self.dtype == bool else "none"
                imshow_kwargs = {"cmap": "gray", "vmax": vmax, "vmin": vmin, "interpolation": interpol}
                imshow_kwargs.update(kwargs)
                
                n_img = len(imglist)
                n_col = min(n_img, 4)
                n_row = int(n_img / n_col + 0.99)
                fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row))
                ax = ax.flat
                for i, img in enumerate(imglist):
                    ax[i].imshow(img, **imshow_kwargs)
                    ax[i].axis("off")
                    ax[i].set_title(f"Image-{i+1}")

            else:
                n_chn = self.sizeof("c")
                fig, ax = plt.subplots(1, n_chn, figsize=(4*n_chn, 4))
                for i in range(n_chn):
                    img = self[f"c={i}"]
                    vmax, vmin = determine_range(self)
                    interpol = "bilinear" if img.dtype == bool else "none"
                    imshow_kwargs = {"cmap": "gray", "vmax": vmax, "vmin": vmin, "interpolation": interpol}
                    imshow_kwargs.update(kwargs)
                    
                    ax[i].imshow(self[f"c={i}"], **imshow_kwargs)
        else:
            raise ValueError("Image must have three or less dimensions.")
        
        plt.show()

        return self

    def imshow_comparewith(self, other, **kwargs):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        for i, img in enumerate([self, other]):
            vmax, vmin = determine_range(img)
            interpol = "bilinear" if img.dtype == bool else "none"
            imshow_kwargs = {"cmap": "gray", "vmax": vmax, "vmin": vmin, "interpolation": interpol}
            imshow_kwargs.update(kwargs)
            ax[i].imshow(img, **imshow_kwargs)
        
        plt.show()
        return self
    
    @need_labels
    def imshow_label(self, alpha=0.3, dims="yx", **kwargs):
        if self.ndim == 2:
            vmax, vmin = determine_range(self)
            imshow_kwargs = {"vmax": vmax, "vmin": vmin, "interpolation": "none"}
            imshow_kwargs.update(kwargs)
            vmin = imshow_kwargs["vmin"]
            vmax = imshow_kwargs["vmax"]
            if vmin and vmax:
                image = (np.clip(self.value, vmin, vmax) - vmin)/(vmax - vmin)
            else:
                image = self.value
            overlay = label2rgb(self.labels.value, image=image, bg_label=0, 
                                alpha=alpha, image_alpha=1)
            plt.imshow(overlay, **imshow_kwargs)
            self.hist()
        elif self.ndim == 3:
            if "c" not in self.axes:
                imglist = self.split(axis=find_first_appeared(self.axes, exclude=dims))
                if len(imglist) > 24:
                    print("Too many images. First 24 images are shown.")
                    imglist = imglist[:24]

                vmax, vmin = determine_range(self)

                imshow_kwargs = {"vmax": vmax, "vmin": vmin, "interpolation": "none"}
                imshow_kwargs.update(kwargs)
                
                n_img = len(imglist)
                n_col = min(n_img, 4)
                n_row = int(n_img / n_col + 0.99)
                fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row))
                ax = ax.flat
                for i, img in enumerate(imglist):
                    vmin = imshow_kwargs["vmin"]
                    vmax = imshow_kwargs["vmax"]
                    if vmin and vmax:
                        image = (np.clip(img.value, vmin, vmax) - vmin)/(vmax - vmin)
                    else:
                        image = self.value
                    overlay = label2rgb(img.labels.value, image=image, bg_label=0, 
                                        alpha=alpha, image_alpha=1)
                    ax[i].imshow(overlay, **imshow_kwargs)
                    ax[i].axis("off")
                    ax[i].set_title(f"Image-{i+1}")

            else:
                n_chn = self.sizeof("c")
                fig, ax = plt.subplots(1, n_chn, figsize=(4*n_chn, 4))
                for i in range(n_chn):
                    img = self[f"c={i}"]
                    vmax, vmin = determine_range(img)
                    imshow_kwargs = {"vmax": vmax, "vmin": vmin, "interpolation": "none"}
                    imshow_kwargs.update(kwargs)
                    vmin = imshow_kwargs["vmin"]
                    vmax = imshow_kwargs["vmax"]
                    if vmin and vmax:
                        image = (np.clip(img.value, vmin, vmax) - vmin)/(vmax - vmin)
                    else:
                        image = self.value
                    overlay = label2rgb(img.labels, image=image, bg_label=0, 
                                        alpha=alpha, image_alpha=1)
                    ax[i].imshow(overlay, **imshow_kwargs)
                    
        else:
            raise ValueError("Image must be two or three dimensional.")
        
        plt.show()
        return self
    
    def split(self, axis=None) -> list[LabeledArray]:
        """
        Split n-dimensional image into (n-1)-dimensional images.

        Parameters
        ----------
        axis : str or int, optional
            Along which axis the original image will be split, by default "c"

        Returns
        -------
        list of arrays
            Separate images
        """
        # determine axis in int.
        if axis is None:
            axis = find_first_appeared(self.axes, "cztp<")
        axisint = self.axisof(axis)
        
        imgs = super().split(axisint)
        if hasattr(self, "labels"):
            labels = self.labels.split(axisint)
            for img, lbl in zip(imgs, labels):
                lbl.axes = del_axis(self.labels.axes, axisint)
                lbl.set_scale(self.labels)
                img.labels = lbl
            
        return imgs
    
    @need_labels
    def extract(self, label_ids=None, filt=None, cval:float=0) -> LabeledArray:
        """
        Extract certain regions of the image and substitute others to `cval`.

        Parameters
        ----------
        label_ids : int or iterable of int, by default all the label IDs.
            Which label regions are extracted.
        filt : callable, optional
            If given, only regions `X` that satisfy filt(self, X) will extracted.
        cval : float, by default 0.
            Constant value to fill regions outside the extracted labeled regions.
            
        Returns
        -------
        LabeledArray
            Extracted image
        """        
        if filt is None:
            filt = lambda arr, lbl: True
        elif not callable(filt):
            raise TypeError("`filt` must be callable if given.")
        
        if np.isscalar(label_ids):
            label_ids = [label_ids]
        elif label_ids is None:
            # All the labels except for 0 (which means not labeled)
            label_ids = [i for i in np.unique(self.labels) if i != 0]
            
        region = np.zeros_like(self.labels.value, dtype=np.uint8)
        for i in label_ids:
            subregion = (self.labels == i)
            if filt(self, subregion):
                region += subregion.astype(np.uint8)
            
        out = self.copy()
        out[region == 0] = cval
        return out
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Multi-processing
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def parallel(self, func, axes, *args, outshape:tuple[int]=None, outdtype=np.float32) -> LabeledArray:
        """
        Multiprocessing tool.

        Parameters
        ----------
        func : callable
            Function applied to each image.
            sl, img = func(arg). arg must be packed into tuple or list.
        axes : str or int
            passed to iter()
        args
            Additional arguments of `func`.
        outshape : tuple, optional
            shape of output. By default shape of input image because this
            function is used almost for filtering.
        
        Returns
        -------
        LabeledArray
        """
        if outshape is None:
            outshape = self.shape
            
        out = np.empty(outshape, dtype=outdtype)
        
        # multi-processing has an overhead (~1 sec) so that with a small numbers of
        # images it will be slower with multi-processing.
        if self.__class__.n_cpu > 1 and self.size > 10**7:
            results = self._parallel(func, axes, *args)
            for sl, imgf in results:
                out[sl] = imgf
        else:
            for sl, img in self.iter(axes):
                sl, imgf = func((sl, img, *args))
                out[sl] = imgf
        
        out = out.view(self.__class__)
        return out
    
    def parallel_eig(self, func, dims, *args) -> tuple[LabeledArray, LabeledArray]:
        """
        Similar to `parallel()` but this function returns two arrays.
        `eigval` has shape of `(L,) + self.shape` and contains eigenvalues for 
        every L, while `eigvec` has shape of `(R, L) + self.shape` and contains
        eigenvectors for every L.

        Parameters
        ----------
        func : callable
            Function applied to each image.
            sl, img = func(arg). arg must be packed into tuple or list.
        dims : str or int
            passed to iter()
        args
            Additional arguments of `func`.

        Returns
        -------
        LabeledArray and LabeledArray
            Eigenvalues and eigenvectors
        """        
        ndim = len(complement_axes(dims, self.axes))
        eigval = np.empty(self.shape+(ndim,), dtype=np.float32)
        eigvec = np.empty(self.shape+(ndim, ndim), dtype=np.float32)
        
        if self.__class__.n_cpu > 1 and self.size > 10**7:
            results = self._parallel(func, dims, *args)
            for sl, eigval_, eigvec_ in results:
                eigval[sl] = eigval_
                eigvec[sl] = eigvec_
        else:
            for sl, img in self.iter(dims):
                sl, eigval_, eigvec_ = func((sl, img, *args))
                eigval[sl] = eigval_
                eigvec[sl] = eigvec_
            
        return eigval.view(self.__class__), eigvec.view(self.__class__)
    
    
    def _parallel(self, func, axes, *args, israw=False):
        lmd = lambda x : (x[0], x[1], *args)
        inputs = list(self.iter(axes, israw))
        with multi.Pool(self.__class__.n_cpu) as p:
            results = p.map(func, map(lmd, inputs))
        return results
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Cropping
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
    @record()
    def crop_center(self, scale:nDFloat=0.5, *, dims="yx") -> LabeledArray:
        """
        Crop out the center of an image. 
        
        Parameters
        ----------
        scale : float or array-like, default is 0.5
            Scale of the cropped image. If an array is given, each axis will be cropped in different scales,
            using each value respectively.
        dims : str, default is "yx"
            Dimensions to be cropped.
            
        Example
        -------
        (1) Create 512x512 image from 1024x1024 image.
        >>> img_cropped = img.crop_center(scale=0.5)
        (2) Create 21x256x256 image from 63x1024x1024 image.
        >>> img_cropped = img.crop_center(scale=[1/3, 1/2, 1/2])
        """
        # check scale
        if hasattr(scale, "__iter__") and len(scale) == 3 and dims == "yx":
            dims = "zyx"
        scale = np.asarray(check_nd(scale, len(dims)))
        if np.any((scale <= 0) | (1 < scale)):
            raise ValueError(f"scale must be (0, 1], but got {scale}")
        
        # Make axis-targeted slicing string
        sizes = self.sizesof(dims)
        slices = []
        for a, size, sc in zip(dims, sizes, scale):
            x0 = int(size / 2 * (1 - sc))
            x1 = int(size / 2 * (1 + sc)) + 1
            slices.append(f"{a}={x0}:{x1}")

        out = self[";".join(slices)]
        
        return out
    
    @record()
    def crop_kernel(self, radius:nDInt=2) -> LabeledArray:
        """
        Make a kernel from an image by cropping out the center region. This function is useful especially
        in `ImgArray.defocus()`.

        Parameters
        ----------
        radius : int or array-like of int, default is 2
            Radius of the kernel.

        Returns
        -------
        LabeledArray
            Kernel
        
        Examples
        --------
        Make a 4x4x4 kernel from a point spread function image (suppose the image shapes are all even numbers).
        >>> psf = ip.imread(r".../PSF.tif")
        >>> psfker = psf.crop_kernel()
        >>> psfer.shape
        (4, 4, 4)
        """        
        sizes = self.shape
        radii = check_nd(radius, len(sizes))
        return self[tuple(slice(s//2-r, (s+1)//2+r) for s, r in zip(sizes, radii))]
    
    @record()
    def remove_edges(self, pixel:nDInt=1, *, dims="yx") -> LabeledArray:
        """
        Remove pixels from the edges.

        Parameters
        ----------
        pixel : int or array-like, default is 1
            Number of pixels to remove. If an array is given, each axis will be cropped with different pixels,
            using each value respectively.

        Returns
        -------
        LabeledArray
            Cropped image.
        """        
        if hasattr(pixel, "__iter__") and len(pixel) == 3 and dims == "yx":
            dims = "zyx"
        pixel = np.asarray(check_nd(pixel, len(dims)), dtype=np.int64)
        if np.any(pixel < 0):
            raise ValueError("`pixel` must be positive.")
            
        slices = []
        for a, px in zip(dims, pixel):
            slices.append(f"{a}={px}:-{px}")
        
        out = self[";".join(slices)]
        return out
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Label handling
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @dims_to_spatial_axes
    def specify(self, center:Coords, radius:Coords, *, dims=None, labeltype:str="square") -> Label:
        """
        Make rectangle or ellipse labels from points.
        
        Parameters
        ----------
        center : array like or MarkerFrame
            Coordinates of centers. For MarkerFrame, it must have the same axes order.
        radius : float or array-like
            Radius of labels.
        dims : int or str, optional
            Dimension of axes.
        labeltype : str, default is "square"
            The shape of labels.

        Returns
        -------
        ImgArray
            Labeled image.
        
        Example
        -------
        Find single molecules, draw circular labels around them if mean values were greater than 100.
        >>> coords = img.find_sm()
        >>> filter_func = lambda a: np.mean(a) > 100
        >>> img.specify(coords, 3.5, filt=filter_func, labeltype="circle")
        >>> ip.gui.add(img)
        """
        if isinstance(center, MarkerFrame):
            from ._process_numba import _specify_circ_2d, _specify_circ_3d, _specify_square_2d, _specify_square_3d
            ndim = len(dims)
            radius = np.asarray(check_nd(radius, ndim), dtype=np.float32)
            
            if labeltype in ("square", "s"):
                radius = radius.astype(np.uint8)
                _specify = {2: _specify_square_2d,
                            3: _specify_square_3d}[ndim]
                
            elif labeltype in ("circle", "c"):
                _specify = {2: _specify_circ_2d,
                            3: _specify_circ_3d}[ndim]
            
            else:
                raise ValueError("`labeltype` must be 'square' or 'circle'.")
            
            label_axes = str(center.col_axes)
            label_shape = self.sizesof(label_axes)
            labels = largest_zeros(label_shape)
            
            with Progress("specify"):
                n_label = 1
                for sl, crds in center.iter(complement_axes(dims, center.col_axes)):
                    _specify(labels[sl], crds.values, radius, n_label)
                    n_label += len(crds)
            
            if hasattr(self, "labels"):
                print("Existing labels are updated.")
            self.labels = Label(labels, axes=label_axes).optimize()
            self.labels.set_scale(self)
        
        else:
            center = np.asarray(center)
            if center.ndim == 1:
                center = center.reshape(1, -1)
            
            cols = {2:"yx", 3:"zyx"}[center.shape[1]]
            center = MarkerFrame(center, columns=cols, dtype=np.uint16)

            return self.specify(center, radius, dims=dims, labeltype=labeltype)     
        
        return self.labels
    
    @record(append_history=False)
    def reslice(self, src, dst, *, order:int=1) -> PropArray:
        """
        Measure line profile iteratively for every slice of image. This function is almost same as
        `skimage.measure.profile_line`, but can reslice 3D-images. The argument `linewidth` is not 
        implemented here because it is useless.

        Parameters
        ----------
        src : array-like
            Source coordinate.
        dst : array-like
            Destination coordinate.
        order : int, default is 1
            Spline interpolation order.

        Returns
        -------
        PropArray
            Line scans.
        
        Example
        -------
        Rescile along a line and fit to a model function for every time frame.
        >>> scan = img.reslice([18,32], [53,48])
        >>> out = scan.curve_fit(func, init, return_fit=True)
        >>> plt.plot(scan[0])
        >>> plt.plot(out.fit[0])
        """        
        src = np.asarray(src, dtype=np.float32)
        dst = np.asarray(dst, dtype=np.float32)
        d = dst - src
        length = int(np.ceil(np.sqrt(np.sum(d**2)) + 1))
        perp_lines = np.stack([np.linspace(src_, dst_, length).reshape(-1,1) 
                               for src_, dst_ in zip(src, dst)])
        
        ndim = src.size
        if ndim == self.ndim:
            dims = self.axes
        else:
            dims = complement_axes("c", self.axes)[-ndim:]
        c_axes = complement_axes(dims, self.axes)
        out = PropArray(np.empty(self.sizesof(c_axes) + (length,), dtype=np.float32),
                        name=self.name, dtype=np.float32, axes=c_axes+dims[-1], propname="reslice")
        
        for sl, img in self.iter(c_axes, exclude=dims):
            out[sl] = ndi.map_coordinates(img, perp_lines, prefilter=order > 1,
                                         order=order, mode="reflect")[:, 0]
            
        out.set_scale(self)
        return out
    
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def label(self, label_image=None, *, dims=None, connectivity=None) -> Label:
        """
        Run skimage's label() and store the results as attribute.

        Parameters
        ----------
        label_image : array, optional
            Image to make label, by default self is used.
        dims : int or str, optional
            Dimension of axes.
        connectivity : int, optional
            Passed to `skimage.measure.label()`.

        Returns
        -------
        Label
            Labeled image.
        
        Example
        -------
        Label the image with threshold and visualize with napari.
        >>> thr = img.threshold()
        >>> img.label(thr)
        >>> ip.gui.add(img)
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
        labels = largest_zeros(label_image.shape)
        labels[:] = label_image.parallel(label_, c_axes, connectivity, outdtype=labels.dtype).view(np.ndarray)
        # correct the label numbers of `labels`
        self.labels = labels.view(Label)
        self.labels._set_info(label_image, "label")
        self.labels = self.labels.increment_iter(c_axes).optimize()
        self.labels.set_scale(self)
        return self.labels
    
    @dims_to_spatial_axes
    @record(append_history=False)
    def label_if(self, label_image=None, filt=None, *, dims=None, connectivity=None) -> Label:
        """
        Label image using `label_image` as reference image only if certain condition
        dictated in `filt` is satisfied. skimage.measure.regionprops_table is called
        inside everytime image is labeled.

        Parameters
        ----------
        label_image : array, optional
            Image to make label, by default self is used.
        filt : callable, positional argument but not optional
            Filter function. This function must take argument in following style:
                def filt(img, lbl, area, major_axis_length):
                    return area>10 and major_axis_length>5
            where the first argument is intensity image sliced from `self`, the second
            is label image sliced from labeled `label_image`, and the rest arguments is 
            properties that will be calculated using `regionprops` function. The property 
            arguments **must be named exactly same** as the properties in `regionprops`.
            Number of arguments can be two.
            
        dims : int or str, optional
            Dimension of axes.
        connectivity : int, optional
            Passed to `skimage.measure.label()`.

        Returns
        -------
        LabeledArray
            Labeled image
        
        Example
        -------
        1. Label regions if only intensity is high.
        >>> def high_intensity(img, lbl, slice):
        >>>     return np.mean(img[slice]) > 10000
        >>> img.label_if(lbl, filt)
        
        2. Label regions if no hole exists.
        >>> def no_hole(img, lbl, euler_number):
        >>>     return euler_number > 0
        >>> img.label_if(lbl, filt)
        
        3. Label regions if centroids are inside themselves.
        >>> def no_hole(img, lbl, centroid):
        >>>     yc, xc = map(int, centroid)
        >>>     return lbl[yc, xc] > 0
        >>> img.label_if(lbl, filt)
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
        
        # check filter function
        if filt is None:
            raise ValueError("`filt` must be given.")
        if not callable(filt):
            raise TypeError("`filt` must be callable.")
        
        properties = tuple(inspect.signature(filt).parameters)[2:]
            
        c_axes = complement_axes(dims, self.axes)
        labels = largest_zeros(label_image.shape)
        offset = 1
        for sl, lbl in label_image.iter(c_axes):
            lbl = skmes.label(lbl, background=0, connectivity=connectivity)
            img = self.value[sl]
            # Following lines are essentially doing the same thing as `skmes.regionprops_table`.
            # However, `skmes.regionprops_table` returns tuples in the separated columns in
            # DataFrame and rename property names like "centroid-0" and "centroid-1".
            props_obj = skmes.regionprops(lbl, img, cache=False)
            d = {prop_name: [getattr(prop, prop_name) for prop in props_obj]
                    for prop_name in properties}
            df = pd.DataFrame(d)
            del_list = [i+1 for i, r in df.iterrows() if not filt(img, lbl, **r)]
            labels[sl] = skseg.relabel_sequential(np.where(np.isin(lbl, del_list),
                                                        0, lbl), offset=offset)[0]
            offset += labels.max()
        
        self.labels = labels.view(Label).optimize()
        self.labels._set_info(label_image, "label_if")
        self.labels.set_scale(self)
        return self.labels
            
    
    @dims_to_spatial_axes
    @need_labels
    @record(append_history=False)
    def expand_labels(self, distance:int=1, *, dims=None) -> Label:
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
        Label
            Same array but labels are updated.
        """        
        
        labels = np.empty_like(self.labels).value
        for sl, img in self.iter(complement_axes(dims, self.axes), israw=True, exclude=dims):
            labels[sl] = skseg.expand_labels(img.labels.value, distance)
        
        self.labels.value[:] = labels
        
        return self.labels
    
    @record(append_history=False)
    def append_label(self, label_image:np.ndarray, new:bool=False) -> Label:
        """
        Append new labels from an array. This function works for boolean or signed int arrays.

        Parameters
        ----------
        label_image : np.ndarray
            Labeled image.
        new : bool, default is False
            If True, existing labels will be removed anyway.
        
        Returns
        -------
        Label
            New labels.
        
        Example
        -------
        Make label from different channels.
        >>> thr0 = img["c=0"].threshold("90%")
        >>> thr0.label() # binary to label
        >>> thr1 = img["c=1"].threshold("90%")
        >>> thr1.label() # binary to label
        >>> img.append_label(thr0.labels)
        >>> img.append_label(thr1.labels)
        If `thr0` has 100 labels and `thr1` has 150 labels then `img` will have 100+150=250 labels.
        """
        # check and cast label dtype
        if not isinstance(label_image, np.ndarray):
            raise TypeError(f"`label_image` must be ndarray, but got {type(label_image)}")
        elif label_image.dtype.kind == "u":
            pass
        elif label_image.dtype == bool:
            label_image = label_image.astype(np.uint8)
        elif label_image.dtype == np.int32:
            label_image = label_image.astype(np.uint16)
        elif label_image.dtype == np.int64:
            label_image = label_image.astype(np.uint32)
        elif label_image.dtype.kind == "i":
            label_image = label_image.astype(np.uint8)
        else:
            raise ValueError(f"`label_image` has dtype {label_image.dtype}, which is unable to be "
                             "interpreted as an label.")
            
        if hasattr(self, "labels") and not new:
            if label_image.shape != self.labels.shape:
                raise ImageAxesError(f"Shape mismatch. Existing labels have shape {self.labels.shape} "
                                     f"while labels with shape {label_image.shape} is given.")
            
            self.labels = self.labels.add_label(label_image)
        else:
            # when label_image is simple ndarray
            if not hasattr(label_image, "axes"):
                if label_image.shape == self.shape:
                    axes = self.axes
                elif label_image.ndim == 2 and "y" in self.axes and "x" in self.axes:
                    axes = "yx"
                else:
                    raise ValueError("Could not infer axes of `label_image`.")
            else:
                axes = label_image.axes
                if not axes_included(self, label_image):
                    raise ImageAxesError(f"Axes mismatch. Image has {self.axes}-axes but {axes} was given.")
                
            self.labels = Label(label_image, axes=axes, dirpath=self.dirpath)
        return self.labels
    
    @need_labels
    @dims_to_spatial_axes
    @record(append_history=False)
    def proj_labels(self, *, dims=None, forbid_overlap=False) -> Label:
        """
        Label projection. This function is useful when yx-labels are drawn in different z but
        you want to merge them.

        Parameters
        ----------
        dims : int or str, optional
            Spatial dimensions.
        forbid_overlap : bool, default is False
            If True and there were any label overlap, this function will raise ValueError.

        Returns
        -------
        Label
            Projected labels.
        """        
        c_axes = complement_axes(dims, self.labels.axes)
        new_labels = np.max(self.labels, axis=c_axes)
        if forbid_overlap:
            test_array = np.sum(self.labels>0, axis=c_axes)
            if (test_array>1).any():
                raise ValueError("Label overlapped.")
        new_labels._set_info(self.labels, "proj", new_axes=dims)
        self.labels = new_labels
        return self.labels