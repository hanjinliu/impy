from __future__ import annotations
import numpy as np
import multiprocessing as multi
import matplotlib.pyplot as plt
import os
from .axes import ImageAxesError
from .func import *
from .deco import *
from .utilcls import *
from ._process import label_
from .historyarray import HistoryArray
from .label import Label
from .specials import *
from tifffile import imwrite
from skimage.exposure import histogram
from skimage import segmentation as skseg
from skimage import measure as skmes
from skimage.color import label2rgb

class LabeledArray(HistoryArray):
    n_cpu = 4
    show_progress = True
        
    @property
    def range(self):
        return self.min(), self.max()
    
        
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
    
    
    def imsave(self, tifname:str, dtype="uint16"):
        """
        Save image (at the same directory as the original image by default).
        """
        if not tifname.endswith(".tif"):
            tifname += ".tif"
        if os.sep not in tifname:
            tifname = os.path.join(self.dirpath, tifname)
        if self.metadata is None:
            self.metadata = {}
            
        metadata = self.metadata.copy()
        metadata.update({"min":np.percentile(self, 1), 
                         "max":np.percentile(self, 99)})
        
        try:
            info = load_json(metadata["Info"])
        except:
            info = {}
        
        info["impyhist"] = "->".join([self.name] + self.history)
        metadata["Info"] = str(info)
        if self.axes:
            metadata["axes"] = str(self.axes).upper()

        imwrite(tifname, self.as_img_type(dtype).value, imagej=True, metadata=metadata)
        
        print(f"Succesfully saved: {tifname}")
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Basic Functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
        self = self.astype("float32")
        if isinstance(value, np.ndarray):
            value[value==0] = np.inf
        return super().__truediv__(value)
    
    @check_value
    def __itruediv__(self, value):
        self = self.astype("float32")
        if isinstance(value, np.ndarray):
            value[value==0] = np.inf
        return super().__itruediv__(value)
    
    def __array_finalize__(self, obj):
        
        super().__array_finalize__(obj)
        self._view_labels(obj)
        if hasattr(obj, "ongoing"):
            self.ongoing = obj.ongoing    
    
    def _view_labels(self, other):
        """
        Make a view of label **if possible**.
        """
        if (hasattr(other, "labels") and 
            axes_included(self, other.labels) and
            shape_match(self, other.labels)):
            self.labels = other.labels
    
    def _getitem_additional_set_info(self, other, **kwargs):
        super()._getitem_additional_set_info(other, **kwargs)
        # set labels correctly
        key = kwargs["key"]
        if other.axes and hasattr(other, "labels") and not isinstance(key, np.ndarray):
            label_sl = []
            if isinstance(key, tuple):
                _keys = key
            else:
                _keys = (key,)
            for i, a in enumerate(other.axes):
                if a in other.labels.axes and i < len(_keys):
                    label_sl.append(_keys[i])
                    
            if len(label_sl) == 0 or len(label_sl) > other.labels.ndim:
                label_sl = (slice(None),)
            try:
                self.labels = other.labels[tuple(label_sl)]
            except IndexError as e:
                print("`labels` was not inherited due to IndexError :", e)
        
        return None

    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        super()._set_info(other, next_history, new_axes)
        # if any function is on-going
        if hasattr(other, "ongoing"):
            self.ongoing = other.ongoing
        
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
    
    def as_uint8(self):
        if self.dtype == "uint8":
            return self
        
        if self.dtype == "uint16":
            out = self.value / 256
        elif self.dtype == "bool":
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


    def as_uint16(self):
        if self.dtype == "uint16":
            return self
        if self.dtype == "uint8":
            out = self.value * 256
        elif self.dtype == "bool":
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
    
    def as_float(self):
        out = self.value.astype("float32").view(self.__class__)
        out._set_info(self)
        return out
        
    
    def as_img_type(self, dtype="uint16"):
        if str(self.dtype) == dtype:
            return self
        elif dtype == "uint16":
            return self.as_uint16()
        elif dtype == "uint8":
            return self.as_uint8()
        elif dtype in ("float", "f", "float32"):
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
        d = self.astype("uint8").ravel() if self.dtype == bool else self.ravel()
        y, x = histogram(d, nbins=nbin)
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
    def imshow_label(self, alpha=0.3, **kwargs):
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
            overlay = label2rgb(self.labels, image=image, bg_label=0, 
                                alpha=alpha, image_alpha=1)
            plt.imshow(overlay, **imshow_kwargs)
            self.hist()
        elif self.ndim == 3:
            if "c" not in self.axes:
                imglist = [s[1] for s in self.iter("ptz", False, israw=True)]
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
                    overlay = label2rgb(img.labels, image=image, bg_label=0, 
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
            axis = find_first_appeared(self.axes, "cztp")
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
    def extract(self, label_ids=None, filt=None, cval:float=0):
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
        if not callable(filt):
            raise TypeError("`filt` must be callable if given.")
        elif filt is None:
            filt = lambda arr, lbl: True
        
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
    
    def iter(self, axes, showprogress:bool=True, israw:bool=False, exclude:str=""):
        """
        Iteration along axes.

        Parameters
        ----------
        axes : str or int
            On which axes iteration is performed. Or the number of spatial dimension.
        showprogress : bool, optional
            If show progress of algorithm, by default True
        israw : bool, default is False
            If True, MetaArray will be returned. If False, np.ndarray will be returned.
        exclude : str, optional
            Which axes will be excluded in output. For example, self.axes="tcyx" and 
            exclude="c" then the axes of output will be "tyx" and slice is also correctly 
            arranged.

        Yields
        -------
        np.ndarray
            Subimage
        """        
        name = getattr(self, "ongoing", "iteration")
        timer = Timer()
        if showprogress and self.__class__.show_progress:
            print(f"{name} ...", end="")
        for x in super().iter(axes, israw=israw, exclude=exclude):
            yield x
            
        timer.toc()
        if showprogress and self.__class__.show_progress:
            print(f"\r{name} completed ({timer})")
    
    def parallel(self, func, axes, *args, outshape:tuple[int]=None, outdtype=np.float32):
        """
        Multiprocessing tool.

        Parameters
        ----------
        func : callable
            Function applied to each image.
            sl, img = func(arg). arg must be packed into tuple or list.
        axes : str or int
            passed to iter()
        outshape : tuple, optional
            shape of output. By default shape of input image because this
            function is used almost for filtering
        
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
                sl, out2d = func((sl, img, *args))
                out[sl] = out2d
        
        out = out.view(self.__class__)
        return out
    
    def parallel_eig(self, func, dims, *args):
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

        Returns
        -------
        LabeledArray and LabeledArray
        """        
        ndim = len(complement_axes(dims, self.axes))
        eigval = np.empty(self.shape+(ndim,), dtype="float32")
        eigvec = np.empty(self.shape+(ndim, ndim), dtype="float32")
        
        if self.__class__.n_cpu > 1:
            results = self._parallel(func, dims, *args)
            for sl, eigval_, eigvec_ in results:
                eigval[sl] = eigval_
                eigvec[sl] = eigvec_
        else:
            for sl, img in self.iter(dims):
                sl, eigval_, eigvec_ = func((sl, img, dims, *args))
                eigval[sl] = eigval_
                eigvec[sl] = eigvec_
            
        return eigval.view(self.__class__), eigvec.view(self.__class__)
    
    
    def _parallel(self, func, axes, *args, israw=False):
        lmd = lambda x : (x[0], x[1], *args)
        name = getattr(self, "ongoing", "iteration")
        timer = Timer()
        if self.__class__.show_progress:
            print(f"{name} ...", end="")
        with multi.Pool(self.__class__.n_cpu) as p:
            results = p.map(func, map(lmd, self.iter(axes, False, israw)))
        timer.toc()
        if self.__class__.show_progress:
            print(f"\r{name} completed ({timer})")
        return results
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Label handling
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
    @record()
    def crop_center(self, scale:float=0.5) -> LabeledArray:
        """
        Crop out the center of an image.
        e.g. when scale=0.5, create 512x512 image from 1024x1024 image.
        """
        if scale <= 0 or 1 < scale:
            raise ValueError(f"scale must be (0, 1], but got {scale}")
        
        sizey, sizex = self.sizesof("yx")
        
        x0 = int(sizex / 2 * (1 - scale))
        x1 = int(sizex / 2 * (1 + scale)) + 1
        y0 = int(sizey / 2 * (1 - scale))
        y1 = int(sizey / 2 * (1 + scale)) + 1

        out = self[f"y={y0}:{y1};x={x0}:{x1}"]
        
        return out
    
    
    @dims_to_spatial_axes
    def specify(self, center, radius, filt=None, *, dims=None, labeltype="square") -> LabeledArray:
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
        
        Example
        -------
        Find single molecules, draw circular labels around them if mean values were greater than 100.
        >>> coords = img.find_sm()
        >>> filter_func = lambda a: np.mean(a) > 100
        >>> img.specify(coords, 3.5, filt=filter_func, labeltype="circle")
        >>> ip.window.add(img)
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
            self.labels = Label(np.zeros(label_shape, dtype=np.uint8), dtype=np.uint8, axes=label_axes)
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
            center = MarkerFrame(center, columns=cols, dtype=np.uint16)

            return self.specify(center, radius, filt=filt, dims=dims, labeltype=labeltype)     
        
        return self
    
    def reslice(self, src, dst, linewidth:int=1, *, order:int=None, dims="yx") -> PropArray:
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
        # determine length, TODO: test
        src = np.asarray(src, dtype=float)
        dst = np.asarray(dst, dtype=float)
        d_row, d_col = dst - src
        length = int(np.ceil(np.hypot(d_row, d_col) + 1))
        
        c_axes = complement_axes(dims, self.axes)
        out = PropArray(np.empty(self.sizesof(c_axes) + (length,), dtype=np.float32),
                        name=self.name, dtype=np.float32, axes=c_axes+dims[-1], propname="reslice")
        self.ongoing = "reslice"
        for sl, img in self.iter(c_axes, exclude=dims):
            out[sl] = skmes.profile_line(img, src, dst, linewidth=linewidth, 
                                         order=order, mode="reflect")
        out.set_scale(self)
        return out
    
    @dims_to_spatial_axes
    @record(False)
    def label(self, label_image=None, *, dims=None, connectivity=None) -> LabeledArray:
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
        LabeledArray
            Labeled image.
        
        Example
        -------
        Label the image with threshold and visualize with napari.
        >>> thr = img.threshold()
        >>> img.label(thr)
        >>> ip.window.add(img)
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
        labels = largest_zeros(label_image.shape)
        labels[:] = label_image.parallel(label_, c_axes, connectivity, outdtype=labels.dtype).view(np.ndarray)
        label_image.ongoing = None
        del label_image.ongoing
        
        min_nlabel = 0
        for sl, _ in label_image.iter(c_axes, False):
            labels[sl][labels[sl]>0] += min_nlabel
            min_nlabel += labels[sl].max()
        
        self.labels = labels.view(Label).optimize()
        self.labels._set_info(label_image, "label")
        self.labels.set_scale(self)
        return self
    
    @dims_to_spatial_axes
    @need_labels
    @record(record_label=True)
    def expand_labels(self, distance:int=1, *, dims=None) -> LabeledArray:
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
    
    def append_label(self, label_image:np.ndarray, new:bool=False) -> LabeledArray:
        if not isinstance(label_image, np.ndarray):
            raise TypeError(f"`label_image` must be ndarray, but got {type(label_image)}")
        
        if hasattr(self, "labels") and not new:
            if label_image.shape != self.labels.shape:
                raise ImageAxesError(f"Shape mismatch. Existing labels have shape {self.labels.shape} "
                                     f"while labels with shape {label_image.shape} is given.")
            if label_image.dtype == bool:
                label_image = label_image.astype(np.uint8)
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
        return self