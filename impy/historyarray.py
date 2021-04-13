from .func import *
from .utilcls import *
from .metaarray import MetaArray


class HistoryArray(MetaArray):
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                history=None, metadata=None, lut=None):
        
        self = super().__new__(cls, obj, name, axes, dirpath, metadata)
        self.history = [] if history is None else history
        self.lut = lut
        return self

    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 history=None, metadata=None, lut=None):
        pass
    
    def __repr__(self):
        return f"\n"\
               f"    shape     : {self.shape_info}\n"\
               f"    dtype     : {self.dtype}\n"\
               f"  directory   : {self.dirpath}\n"\
               f"original image: {self.name}\n"\
               f"   history    : {'->'.join(self.history)}\n"
    
    def __getitem__(self, key):
        if isinstance(key, str):
            # img["t=2;z=4"] ... ImageJ-like method
            sl = self.str_to_slice(key)
            return self.__getitem__(sl)

        if isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
            # img[arr] ... where arr is 2-D boolean array
            key = add_axes(self.axes, self.shape, key)

        out = np.ndarray.__getitem__(self, key) # get item as np.ndarray
        keystr = key_repr(key)                 # write down key e.g. "0,*,*"
        if isinstance(out, self.__class__):   # cannot set attribution to such as numpy.int32 
            if hasattr(key, "__array__"):
                # fancy indexing will lose axes information
                new_axes = None
                
            elif "new" in keystr:
                # np.newaxis or None will add dimension
                new_axes = None
                
            elif self.axes:
                del_list = [i for i, s in enumerate(keystr.split(",")) if s != "*"]
                new_axes = del_axis(self.axes, del_list)
            else:
                new_axes = None
            
            new_history = f"getitem[{keystr}]"
            out._set_info(self, new_history, new_axes)
            
            if self.axes and hasattr(self, "labels"):
                label_sl = []
                if isinstance(key, tuple):
                    _keys = key
                else:
                    _keys = (key,)
                for i, a in enumerate(self.axes):
                    if a in self.labels.axes and i < len(_keys):
                        label_sl.append(_keys[i])
                if len(label_sl) == 0:
                    label_sl = (slice(None),)
                out.labels = self.labels[tuple(label_sl)]
                
        # TODO: Ellipsis
        return out
    
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)  # set item as np.ndarray
        keystr = key_repr(key)           # write down key e.g. "0,*,*"
        new_history = f"setitem[{keystr}]"
        
        self._set_info(self, new_history)
        
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
    
    def _inherit_meta(self, obj, ufunc, **kwargs):
        """
        Copy axis, history etc. from obj.
        This is called in __array_ufunc__(). Unlike _set_info(), keyword `axis` must be
        considered because it changes `ndim`.
        """
        if "axis" in kwargs.keys() and not obj.axes.is_none():
            axis = kwargs["axis"]
            new_axes = del_axis(obj.axes, axis)
        else:
            new_axes = "inherit"
        self._set_info(obj, ufunc.__name__, new_axes=new_axes)
        return self
    
    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        super()._set_info(other, new_axes)
        
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
    
    def split(self, axis=None) -> list:
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
            
        imgs = list(np.moveaxis(self, axisint, 0))
        for i, img in enumerate(imgs):
            img.history[-1] = f"axis({axis})={i}"
            img.axes = del_axis(self.axes, axisint)
            if axis == "c" and self.lut is not None:
                img.lut = [self.lut[i]]
            else:
                img.lut = None
        return imgs
    
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
    