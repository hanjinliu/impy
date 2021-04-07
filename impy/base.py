import numpy as np
import multiprocessing as multi
import matplotlib.pyplot as plt
import os
from .func import del_axis, add_axes, get_lut, Timer, load_json, same_dtype, _key_repr
from .metaarray import MetaArray
from tifffile import imwrite
from skimage.exposure import histogram
    
def check_value(__op__):
    def wrapper(self, value):
        if isinstance(value, np.ndarray):
            value = value.astype("float32")
            if (value < 0).any():
                raise ValueError("Cannot multiply or divide array containig negative value.")
            if self.ndim >= 3 and value.shape == self.xyshape():
                value = add_axes(self.axes, self.shape, value)
        elif isinstance(value, (int, float)) and value < 0:
            raise ValueError("Cannot multiply or divide negative value.")

        out = __op__(self, value)
        return out
    return wrapper

# TODO: make lut compatible with imagej
class BaseArray(MetaArray):
    """
    Array implemented with basic functions.
    - axes information such as tzyx.
    - Image visualization and LUT for it.
    - auto dtype conversion upon image division.
    - saturation upon multiplying.
    - intuitive sub-array viewing in ImageJ format such as img["t=1,z=5"].
    - auto history recording.
    """
    n_cpu = 4
    
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                history=[], metadata={}, lut=None):
        
        self = super().__new__(cls, obj, name, axes, dirpath, metadata)
        self.history = [] if history is None else history
        self.lut = lut
        return self

    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 history=[], metadata={}, lut=None):
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
        if (self.axes.is_none()):
            shape_info = self.shape
        else:
            shape_info = ", ".join([f"{s}({o})" for s, o in zip(self.shape, self.axes)])

        return f"\n"\
               f"    shape     : {shape_info}\n"\
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
        if (isinstance(value, np.ndarray)):
            value[value==0] = np.inf
        return super().__truediv__(value)
    
    @check_value
    def __itruediv__(self, value):
        self = self.astype("float32")
        if (isinstance(value, np.ndarray)):
            value[value==0] = np.inf
        return super().__itruediv__(value)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            # img["t=2,z=4"] ... ImageJ-like method
            sl = self.str_to_slice(key)
            return self.__getitem__(sl)

        if isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
            # img[arr] ... where arr is 2-D boolean array
            key = add_axes(self.axes, self.shape, key)

        out = np.ndarray.__getitem__(self, key)          # get item as np.ndarray
        keystr = _key_repr(key)                 # write down key e.g. "0,*,*"
        
        if isinstance(out, self.__class__):   # cannot set attribution to such as numpy.int32 
            if self.axes:
                del_list = []
                for i, s in enumerate(keystr.split(",")):
                    if s != "*":
                        del_list.append(i)
                        
                new_axes = del_axis(self.axes, del_list)
                if hasattr(key, "__array__"):
                    new_axes = None
            else:
                new_axes = None
            
            new_history = f"getitem[{keystr}]"
            out._set_info(self, new_history, new_axes)
        
        return out
    
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)         # set item as np.ndarray
        keystr = _key_repr(key)                 # write down key e.g. "0,*,*"
        new_history = f"setitem[{keystr}]"
        
        self._set_info(self, new_history)
    
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
        self.history = getattr(obj, "history", [])
        try:
            self.lut = getattr(obj, "lut", None)
        except:
            self.lut = None
    
    def _inherit_meta(self, ufunc, *inputs, **kwargs):
        # set attributes for output
        name = "no name"
        dirpath = ""
        history = []
        input_ndim = -1
        axes = None
        metadata = None
        lut = None
        for input_ in inputs:
            if isinstance(input_, self.__class__):
                name = input_.name
                dirpath = input_.dirpath
                history = input_.history.copy()
                axes = input_.axes
                history.append(ufunc.__name__)
                input_ndim = input_.ndim
                metadata = input_.metadata.copy()
                lut = input_.lut
                break

        self.dirpath = dirpath
        self.name = name
        self.history = history
        self.metadata = metadata
        
        # set axes
        if axes is None:
            self.axes = None
            self.lut = None
        elif input_ndim == self.ndim:
            self.axes = axes
            self.lut = lut
        elif input_ndim > self.ndim:
            self.lut = None
            if "axis" in kwargs.keys() and not self.axes.is_none():
                axis = kwargs["axis"]
                self.axes = del_axis(axes, axis)
            else:
                self.axes = None
        else:
            self.axes = None
            self.lut = None

        return self
    
    

    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        super()._set_info(other, new_axes)
        # if any function is on-going
        if hasattr(other, "ongoing"):
            self.ongoing = other.ongoing
        
        # set history
        if next_history is not None:
            self.history = other.history + [next_history]
        else:
            self.history = other.history.copy()
            
        # set lut
        try:
            self.lut = other.lut
        except:
            self.lut = None
        if self.axes.is_none():
            self.lut = None
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
    
    def as_img_type(self, dtype="uint16"):
        if dtype == "uint16":
            return self.as_uint16()
        elif dtype == "uint8":
            return self.as_uint8()
        elif dtype in ("float", "f", "float32"):
            return self.astype("float32")
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
            if self.dtype == bool:
                vmax = vmin = None
            else:
                vmax = np.percentile(self[self>0], 99.99)
                vmin = np.percentile(self[self>0], 0.01)
            cmaps = self.get_cmaps()
            imshow_kwargs = {"cmap": cmaps[0], "vmax": vmax, "vmin": vmin, "interpolation": "none"}
            imshow_kwargs.update(kwargs)
            
            plt.imshow(self, **imshow_kwargs)
            self.hist()
            
        elif self.ndim == 3:
            if "c" not in self.axes:
                imglist = [s[1] for s in self.iter("ptzs", False)]
                if len(imglist) > 24:
                    print("Too many images. First 24 images are shown.")
                    imglist = imglist[:24]

                if self.dtype == bool:
                    vmax = vmin = None
                else:
                    vmax = np.percentile(self[self>0], 99.99)
                    vmin = np.percentile(self[self>0], 0.01)

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
                    if self.dtype == bool:
                        vmax = vmin = None
                    else:
                        vmax = np.percentile(self[self>0], 99.99)
                        vmin = np.percentile(self[self>0], 0.01)
                    imshow_kwargs = {"cmap": cmaps[i], "vmax": vmax, "vmin": vmin, "interpolation": "none"}
                    imshow_kwargs.update(kwargs)
                    
                    ax[i].imshow(self[i], **imshow_kwargs)
        else:
            raise ValueError("Image must be two or three dimensional.")
        
        plt.show()

        return self
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Others
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def iter(self, axes, showprogress:bool=True):
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
        for x in super().iter(axes):
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
        ImgArray
        """
        if outshape is None:
            outshape = self.shape
            
        out = np.zeros(outshape, dtype=outdtype)
        
        if self.__class__.n_cpu > 1:
            lmd = lambda x : (x[0], x[1], *args)
            name = getattr(self, "ongoing", "iteration")
            timer = Timer()
            print(f"{name} ...", end="")
            with multi.Pool(self.__class__.n_cpu) as p:
                results = p.map(func, map(lmd, self.iter(axes, False)))
            timer.toc()
            print(f"\r{name} completed ({timer})")
            
            for sl, imgf in results:
                out[sl] = imgf
        else:
            for sl, img in self.iter(axes):
                sl, out2d = func((sl, img, *args))
                out[sl] = out2d
                
        out = out.view(self.__class__)
        return out
    
    def get_cmaps(self):
        """
        From self.lut get colormap used in plt.
        Default colormap is gray.
        """
        if "c" in self.axes:
            if self.lut is None:
                cmaps = ["gray"] * self.sizeof("c")
            else:
                cmaps = [get_lut(c) for c in self.lut]
        else:
            if self.lut is None:
                cmaps = ["gray"]
            elif (len(self.lut) != len(self.axes)):
                cmaps = ["gray"] * len(self.axes)
            else:
                cmaps = [get_lut(self.lut[0])]
        return cmaps
    

