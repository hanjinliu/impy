import numpy as np
from .axes import Axes
from .func import add_axes, del_axis, _key_repr

class MetaArray(np.ndarray):
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                metadata={}):
        if isinstance(obj, cls):
            return obj
        
        self = np.array(obj).view(cls)
        self.dirpath = "" if dirpath is None else dirpath
        self.name = "Image from impy" if name is None else name
        
        # MicroManager
        if self.name.endswith("_MMStack_Pos0.ome"):
            self.name = self.name[:-17]
        
        self.axes = axes
        self.metadata = {} if metadata is None else metadata
        return self
    
    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 metadata={}):
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
    def value(self):
        return np.asarray(self)
    
    def __repr__(self):
        if (self.axes.is_none()):
            shape_info = self.shape
        else:
            shape_info = ", ".join([f"{s}({o})" for s, o in zip(self.shape, self.axes)])

        return f"\n"\
               f"    shape     : {shape_info}\n"\
               f"    dtype     : {self.dtype}\n"\
               f"  directory   : {self.dirpath}\n"\
               f"original image: {self.name}\n"
    
    def __str__(self):
        return self.name
    

    def showinfo(self):
        print(repr(self))
        return None
    
    def _set_info(self, other, new_axes:str="inherit"):
        self.dirpath = other.dirpath
        self.name = other.name
        self.metadata = other.metadata
        
        # set axes
        if new_axes != "inherit":
            self.axes = new_axes
        else:
            self.axes = other.axes
        
        return None
    
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
            out._set_info(self, new_axes)
        
        return out
    
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
                    if start < 0:
                        raise IndexError(f"out of range: {o}")
                if end == "":
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
    
    # numpy functions that will change/discard order
    def transpose(self, axes):
        """
        change the order of image dimensions.
        'axes' will also be arranged.
        """
        out = super().transpose(axes)
        if (self.axes.is_none()):
            new_axes = None
        else:
            new_axes = "".join([self.axes[i] for i in list(axes)])
        out._set_info(self, new_axes=new_axes)
        return out
    
    def flatten(self):
        out = super().flatten()
        out._set_info(self, new_axes=None)
        return out
    
    def ravel(self):
        out = super().ravel()
        out._set_info(self, new_axes=None)
        return out