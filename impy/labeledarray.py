import numpy as np
import multiprocessing as multi
import matplotlib.pyplot as plt
import os
from .func import *
from .utilcls import *
from .historyarray import HistoryArray
from tifffile import imwrite
from skimage.exposure import histogram
from skimage.color import label2rgb
    
def check_value(__op__):
    def wrapper(self, value):
        if isinstance(value, np.ndarray):
            value = value.astype("float32")
            if self.ndim >= 3 and value.shape == self.sizesof("yx"):
                value = add_axes(self.axes, self.shape, value)
        elif isinstance(value, (int, float)) and value < 0:
            raise ValueError("Cannot multiply or divide negative value.")

        out = __op__(self, value)
        return out
    return wrapper


class LabeledArray(HistoryArray):
    n_cpu = 4
    
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                history=None, metadata=None, lut=None):
        
        self = super().__new__(cls, obj, name, axes, dirpath, metadata)
        self.history = [] if history is None else history
        self.lut = lut
        return self

    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 history=None, metadata=None, lut=None):
        pass
    
    @property
    def lut(self):
        return self._lut
    
    @lut.setter
    def lut(self, value):
        # number of channel
        if self.axes.is_none():
            n_lut = 1
        elif "c" in self.axes.axes[:self.ndim]:
            n_lut = self.sizeof("c")
        else:
            n_lut = 1
        
        if value is None:
            self._lut = ["gray"] * n_lut
        elif n_lut == 1 and type(value) is str:
            self._lut = [value]
        elif n_lut == len(value):
            self._lut = list(value)
        else:
            self._lut = ["gray"] * n_lut
            raise ValueError(f"Incorrect LUT for {n_lut}-channel image: {value}")
    
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
    
    
    def imsave(self, tifname:str):
        """
        Save image (at the same directory as the original image by default).
        """
        if not tifname.endswith(".tif"):
            tifname += ".tif"
        if os.sep not in tifname:
            tifname = os.path.join(self.dirpath, tifname)
        
        metadata = self.metadata
        metadata.update({"min":np.percentile(self, 1), "max":np.percentile(self, 99)})
        
        try:
            info = load_json(metadata["Info"])
        except:
            info = {}
        
        info["impyhist"] = "->".join([self.name] + self.history)
        metadata["Info"] = str(info)
        if (self.axes):
            metadata["axes"] = str(self.axes).upper()

        imwrite(tifname, self.as_uint16().value, imagej=True, metadata=metadata)
        
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
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Overloaded Numpy Functions to Inherit Attributes
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def __array_finalize__(self, obj):
        """
        Every time an np.ndarray object is made by numpy functions inherited to ImgArray,
        this function will be called to set essential attributes.
        Therefore, you can use such as img.copy() and img.astype("int") without problems (maybe...).
        """
        
        super().__array_finalize__(obj)
        
        self._view_labels(obj)
    
    
    def _view_labels(self, other):
        """
        Make a view of label **if possible**.
        """
        if (hasattr(other, "labels") and 
            axes_included(self, other.labels) and
            shape_match(self, other.labels)):
            self.labels = other.labels

    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        super()._set_info(other, next_history, new_axes)
        # if any function is on-going
        if hasattr(other, "ongoing"):
            self.ongoing = other.ongoing
        
        # inherit labels
        self._view_labels(other)
        return None
    
    def _update(self, out):
        self.value[:] = out.as_img_type(self.dtype)
        self.history.append(out.history[-1])
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Type Conversions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def as_uint8(self):
        if self.dtype == "uint8":
            return self
        out = self.value
        if self.dtype == "uint16":
            out = out / 256
        elif self.dtype == "bool":
            pass
        elif self.dtype in ("float16", "float32", "float64"):
            if 0 <= np.min(out) and np.max(out) < 1:
                out = out * 256
            else:
                out = out + 0.5
        else:
            raise TypeError(f"invalid data type: {self.dtype}")
        out[out < 0] = 0
        out[out >= 256] = 255
        out = out.view(self.__class__)
        out._set_info(self)
        out = out.astype("uint8")
        return out


    def as_uint16(self):
        if self.dtype == "uint16":
            return self
        out = self.value
        if self.dtype == "uint8":
            out = out * 256
        elif self.dtype == "bool":
            pass
        elif self.dtype in ("float16", "float32", "float64"):
            if 0 <= np.min(out) and np.max(out) < 1:
                out = out * 65536
            else:
                out = out + 0.5
        else:
            raise TypeError(f"invalid data type: {self.dtype}")
        out[out < 0] = 0
        out[out >= 65536] = 65535
        out = out.view(self.__class__)
        out._set_info(self)
        out = out.astype("uint16")
        return out
    
    def as_float(self):
        out = self.value.astype("float32").view(self.__class__)
        out._set_info(self)
        return out
        
    
    def as_img_type(self, dtype="uint16"):
        if dtype == "uint16":
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

    def imshow(self, **kwargs):
        if self.ndim == 2:
            vmax, vmin = determine_range(self)
            cmaps = self.get_cmaps()
            imshow_kwargs = {"cmap": cmaps[0], "vmax": vmax, "vmin": vmin, "interpolation": "none"}
            imshow_kwargs.update(kwargs)
            plt.imshow(self.value, **imshow_kwargs)
            
            self.hist()
            
        elif self.ndim == 3:
            if "c" not in self.axes:
                imglist = [s[1] for s in self.iter("ptz", False)]
                if len(imglist) > 24:
                    print("Too many images. First 24 images are shown.")
                    imglist = imglist[:24]

                vmax, vmin = determine_range(self)

                cmaps = self.get_cmaps()
                imshow_kwargs = {"cmap": cmaps[0], "vmax": vmax, "vmin": vmin, "interpolation": "none"}
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
                cmaps = self.get_cmaps()
                n_chn = self.sizeof("c")
                fig, ax = plt.subplots(1, n_chn, figsize=(4*n_chn, 4))
                for i in range(n_chn):
                    img = self[f"c={i+1}"]
                    vmax, vmin = determine_range(self)
                    imshow_kwargs = {"cmap": cmaps[i], "vmax": vmax, "vmin": vmin, "interpolation": "none"}
                    imshow_kwargs.update(kwargs)
                    
                    ax[i].imshow(self[i], **imshow_kwargs)
        else:
            raise ValueError("Image must be two or three dimensional.")
        
        plt.show()

        return self
    
    def imshow_label(self, alpha=0.3, image_alpha=1, **kwargs):
        if self.ndim == 2:
            vmax, vmin = determine_range(self)
            imshow_kwargs = {"vmax": vmax, "vmin": vmin, "interpolation": "none"}
            imshow_kwargs.update(kwargs)
            vmin = imshow_kwargs["vmin"]
            vmax = imshow_kwargs["vmax"]
            if vmin and vmax:
                image = (np.clip(self.value, vmin, vmax) - vmin)/(vmax - vmin)
            overlay = label2rgb(self.labels, image=image, bg_label=0, 
                                alpha=alpha, image_alpha=image_alpha)
            plt.imshow(overlay, **imshow_kwargs)
            self.hist()
        else:
            raise NotImplementedError("not implemented for ndim != 2")
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Others
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def iter(self, axes, showprogress:bool=True, israw=False):
        """
        Iteration along axes.

        Parameters
        ----------
        axes : str or int
            On which axes iteration is performed. Or the number of spatial dimension.
        showprogress : bool, optional
            If show progress of algorithm, by default True

        Yields
        -------
        np.ndarray
            Subimage
        """        
        name = getattr(self, "ongoing", "iteration")
        timer = Timer()
        if showprogress:
            print(f"{name} ...", end="")
        for x in super().iter(axes, israw):
            yield x
            
        timer.toc()
        if showprogress:
            print(f"\r{name} completed ({timer})")
    
    def parallel(self, func, axes, *args, outshape=None, outdtype="float32"):
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
        BaseArray
        """
        if outshape is None:
            outshape = self.shape
            
        out = np.empty(outshape, dtype=outdtype)
        
        if self.__class__.n_cpu > 1:
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
        eigval = np.empty(self.shape+(dims,), dtype="float32")
        eigvec = np.empty(self.shape+(dims,dims), dtype="float32")
        
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
        
        # eigenvalues as 1D-list
        eigval = list(np.moveaxis(eigval, -1, 0))
        
        # eigenvectors as 2D-list
        eigvec = list(np.moveaxis(eigvec, -1, 0))
        for i, e in enumerate(eigvec):
            eigvec[i] = SpatialList(np.moveaxis(e, -1 ,0))
            
        return eigval, eigvec
    
    
    def _parallel(self, func, axes, *args, israw=False):
        lmd = lambda x : (x[0], x[1], *args)
        name = getattr(self, "ongoing", "iteration")
        timer = Timer()
        print(f"{name} ...", end="")
        with multi.Pool(self.__class__.n_cpu) as p:
            results = p.map(func, map(lmd, self.iter(axes, False, israw)))
        timer.toc()
        print(f"\r{name} completed ({timer})")
        return results
    
    
