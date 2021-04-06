import numpy as np
import multiprocessing as multi
import matplotlib.pyplot as plt
import os
from .func import del_axis, add_axes, get_lut, Timer, load_json, same_dtype, _key_repr
from .axes import Axes
from tifffile import imwrite
from skimage.exposure import histogram
import itertools
    
def check_value(__op__):
    def wrapper(self, value):
        if (isinstance(value, np.ndarray)):
            value = value.astype("float32")
            if ((value < 0).any()):
                raise ValueError("Cannot multiply or divide array containig negative value.")
            if (self.ndim >= 3 and value.shape == self.xyshape()):
                value = add_axes(self.axes, self.shape, value)
        elif (isinstance(value, (int, float)) and value < 0):
            raise ValueError("Cannot multiply or divide negative value.")

        out = __op__(self, value)
        return out
    return wrapper

# TODO: make lut compatible with imagej
class BaseArray(np.ndarray):
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
        if isinstance(obj, cls):
            return obj
        
        self = np.array(obj).view(cls)
        self.dirpath = "" if dirpath is None else dirpath
        self.name = "Image from impy" if name is None else name
        
        # MicroManager
        if self.name.endswith("_MMStack_Pos0.ome"):
            self.name = self.name[:-17]
        
        self.history = [] if history is None else history
        self.axes = axes
        self.metadata = {} if metadata is None else metadata
        self.lut = lut
        return self

    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 history=[], metadata={}, lut=None):
        pass
    
    @property
    def axes(self):
        return self._axes
    
    @axes.setter
    def axes(self, value):
        if (value is None):
            self._axes = Axes()
        else:
            self._axes = Axes(value, self.ndim)
        
    @property
    def lut(self):
        return self._lut
    
    @lut.setter
    def lut(self, value):
        # number of channel
        if (self.axes.is_none()):
            n_lut = 1
        elif ("c" in self.axes.axes[:self.ndim]):
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
    def value(self):
        return np.asarray(self)
    
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
    
    def __str__(self):
        return self.name
    

    def showinfo(self):
        print(repr(self))
        return None
    
    def imsave(self, tifname:str):
        """
        Save image (at the same directory as the original image by default).
        """
        if (not tifname.endswith(".tif")):
            tifname += ".tif"
        if (os.sep not in tifname):
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
        elif hasattr(key, "__as_roi__"):
            # img[roi] ... get item from ROI.
            return key.__as_roi__(self)

        if isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
            # img[arr] ... where arr is 2-D boolean array
            key = add_axes(self.axes, self.shape, key)

        out = super().__getitem__(key)          # get item as np.ndarray
        keystr = _key_repr(key)                 # write down key e.g. "0,*,*"
        
        if isinstance(out, self.__class__):   # cannot set attribution to such as numpy.int32 
            new_history = f"getitem[{keystr}]"
            if self.axes:
                del_list = []
                for i, s in enumerate(keystr.split(",")):
                    if (s != "*"):
                        del_list.append(i)
                        
                new_axes = del_axis(self.axes, del_list)
                if hasattr(key, "__array__"):
                    new_axes = None
            else:
                new_axes = None
            out._set_info(self, new_history, new_axes)
        
        return out
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            # img["t=2,z=4"] ... ImageJ-like method
            sl = self.str_to_slice(key)
            return self.__setitem__(sl, value)
        super().__setitem__(key, value)         # set item as np.ndarray
        keystr = _key_repr(key)                 # write down key e.g. "0,*,*"
        new_history = f"setitem[{keystr}]"
        
        self._set_info(self, new_history)
    
    def str_to_slice(self, string):
        """
        get subslices using ImageJ-like format.
        e.g. 't=3-, z=1-5', 't=1, z=-7' (this will not be interpreted as minus)
        """
        keylist = [key.strip() for key in string.split(",")]
        olist = [] # e.g. 'z', 't'
        vlist = [] # e.g. 5, 2:4
        for k in keylist:
            # e.g. k = "t = 4-7"
            o, v = [s.strip() for s in k.split("=")]
            olist.append(self.axisof(o))

            # set value or slice
            if "-" in v:
                start, end = [s.strip() for s in v.strip().split("-")]
                if (start == ""):
                    start = None
                else:
                    start = int(start) - 1
                    if (start < 0):
                        raise IndexError(f"out of range: {o}")
                if (end == ""):
                    end = None
                else:
                    end = int(end)

                vlist.append(slice(start, end, None))
            else:
                pos = int(v) - 1
                if (pos < 0):
                        raise IndexError(f"out of range: {o}")
                vlist.append(pos)
        
        input_keylist = []
        for i in range(len(self.axes)):
            if i in olist:
                j = olist.index(i)
                input_keylist.append(vlist[j])
            else:
                input_keylist.append(slice(None))

        return tuple(input_keylist)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Overloaded Numpy Functions to Inherit Attributes
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def __array_finalize__(self, obj):
        """
        Every time an np.ndarray object is made by numpy functions inherited to ImgArray,
        this function will be called to set essential attributes.
        Therefore, you can use such as img.copy() and img.astype("int") without problems (maybe...).
        """
        if (obj is None): return None
        self.dirpath = getattr(obj, "dirpath", None)
        self.name = getattr(obj, "name", None)
        self.history = getattr(obj, "history", [])

        try:
            self.axes = getattr(obj, "axes", None)
        except:
            self.axes = None
        if not self.axes.is_none() and len(self.axes) != self.ndim:
            self.axes = None
        
        self.metadata = getattr(obj, "metadata", {})

        try:
            self.lut = getattr(obj, "lut", None)
        except:
            self.lut = None


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Every time a numpy universal function (add, subtract, ...) is called,
        this function will be called to set/update essential attributes.
        """
        # convert to np.array
        def _replace_self(a):
            if (a is self): return a.view(np.ndarray)
            else: return a

        # call numpy function
        args = tuple(_replace_self(a) for a in inputs)

        if "out" in kwargs:
            kwargs["out"] = tuple(_replace_self(a) for a in kwargs["out"])

        result = getattr(ufunc, method)(*args, **kwargs)

        if result is NotImplemented:
            return NotImplemented
        
        result = result.view(self.__class__)
        
        # in the case result is such as np.float64
        if not isinstance(result, self.__class__):
            return result

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

        result.dirpath = dirpath
        result.name = name
        result.history = history
        result.metadata = metadata
        
        # set axes
        if axes is None:
            result.axes = None
            result.lut = None
        elif input_ndim == result.ndim:
            result.axes = axes
            result.lut = lut
        elif input_ndim > result.ndim:
            result.lut = None
            if "axis" in kwargs.keys() and not self.axes.is_none():
                axis = kwargs["axis"]
                result.axes = del_axis(axes, axis)
            else:
                result.axes = None
        else:
            result.axes = None
            result.lut = None

        return result
    

    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        self.dirpath = other.dirpath
        self.name = other.name
        self.metadata = other.metadata
        
        # if any function is on-going
        if hasattr(other, "ongoing"):
            self.ongoing = other.ongoing
        
        # set history
        if next_history is not None:
            self.history = other.history + [next_history]
        else:
            self.history = other.history.copy()
        
        # set axes
        if new_axes != "inherit":
            self.axes = new_axes
        else:
            self.axes = other.axes
        
        if hasattr(other, "rois") and not self.axes.is_none() and self.xyshape() == other.xyshape():
            self.rois = other.rois

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
        if (self.dtype == "uint8"):
            return self
        out = self.value
        if (self.dtype == "uint16"):
            out = out / 256
        elif(self.dtype == "bool"):
            pass
        elif (self.dtype in ("float16", "float32", "float64")):
            if (0 <= np.min(out) and np.max(out) < 1):
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
        if (self.dtype == "uint16"):
            return self
        out = self.value
        if (self.dtype == "uint8"):
            out = out * 256
        elif(self.dtype == "bool"):
            pass
        elif (self.dtype in ("float16", "float32", "float64")):
            if (0 <= np.min(out) and np.max(out) < 1):
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
        if (dtype == "uint16"):
            return self.as_uint16()
        elif (dtype == "uint8"):
            return self.as_uint8()
        elif (dtype in ("float", "f", "float32")):
            return self.astype("float32")
        elif (dtype == "bool"):
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

    def axisof(self, axisname):
        if (type(axisname) is int):
            return axisname
        else:
            return self.axes.find(axisname)
    
    def xyshape(self):
        return self.sizeof("x"), self.sizeof("y")
    
    def sizeof(self, axis:str):
        return self.shape[self.axes.find(axis)]

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
        if isinstance(axes, int):
            if axes == 2:
                axes = "ptzc"
            elif axes == 3:
                axes = "ptc"
            else:
                ValueError(f"dimension must be 2 or 3, but got {axes}")
                
        axes = "".join([a for a in axes if a in self.axes]) # update axes to existing ones
        iterlist = []
        total_repeat = 1
        for a in self.axes:
            if a in axes:
                iterlist.append(range(self.sizeof(a)))
                total_repeat *= self.sizeof(a)
            else:
                iterlist.append([slice(None)])
        selfview = self.value
        name = getattr(self, "ongoing", "iteration")
        
        timer = Timer()
        for i, sl in enumerate(itertools.product(*iterlist)):
            if total_repeat > 1 and showprogress:
                print(f"\r{name}: {i:>4}/{total_repeat:>4}", end="")
            yield sl, selfview[sl]
            
        timer.toc()
        if showprogress:
            print(f"\r{name}: {total_repeat:>4}/{total_repeat:>4} completed ({timer})")
    
    
    def parallel(self, func, axes, *args):
        """
        Multiprocessing tool.

        Parameters
        ----------
        func : callable
            Function applied to each image.
            sl, img = func(arg). arg must be packed into tuple or list.
        axes : str or int
            passed to iter()

        Returns
        -------
        ImgArray
        """
        out = np.zeros(self.shape)
        if self.__class__.n_cpu > 0:
            lmd = lambda x : (x[0], x[1], *args)
            name = getattr(self, "ongoing", "iteration")
            timer = Timer()
            print(f"{name} ...", end="")
            with multi.Pool(self.__class__.n_cpu) as p:
                results = p.map(func, map(lmd, self.as_uint16().iter(axes, False)))
            timer.toc()
            print(f"\r{name} completed ({timer})")
            
            for sl, imgf in results:
                out[sl] = imgf
        else:
            for sl, img in self.as_uint16().iter(axes):
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
    

