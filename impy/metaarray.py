import numpy as np
from .axes import Axes
from .func import add_axes, del_axis, _key_repr
import itertools

class MetaArray(np.ndarray):
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                metadata=None, dtype=None):
        if isinstance(obj, cls):
            return obj
        
        self = np.array(obj, dtype=dtype).view(cls)
        self.dirpath = "" if dirpath is None else dirpath
        self.name = "Image from impy" if name is None else name
        
        # MicroManager
        if self.name.endswith("_MMStack_Pos0.ome"):
            self.name = self.name[:-17]
        
        self.axes = axes
        self.metadata = {} if metadata is None else metadata
        return self
    
    def __init__(self, obj, name=None, axes=None, dirpath=None, 
                 metadata=None):
        pass
    
    @property
    def axes(self):
        return self._axes
    
    @axes.setter
    def axes(self, value):
        if value is None:
            self._axes = Axes()
        else:
            self._axes = Axes(value, self.ndim)
    
    @property
    def value(self):
        return np.asarray(self)
    
    def __repr__(self):
        if self.axes.is_none():
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

        if isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
            # img[arr] ... where arr is 2-D boolean array
            key = add_axes(self.axes, self.shape, key)

        out = super().__getitem__(key)          # get item as np.ndarray
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
                
            out._set_info(self, new_axes)
        
        return out
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            # img["t=2,z=4"] ... ImageJ-like method
            sl = self.str_to_slice(key)
            return self.__setitem__(sl, value)
        super().__setitem__(key, value)
    
    def __array_finalize__(self, obj):
        """
        Every time an np.ndarray object is made by numpy functions inherited to ImgArray,
        this function will be called to set essential attributes.
        Therefore, you can use such as img.copy() and img.astype("int") without problems (maybe...).
        """
        if obj is None: return None
        self.dirpath = getattr(obj, "dirpath", None)
        self.name = getattr(obj, "name", None)

        try:
            self.axes = getattr(obj, "axes", None)
        except:
            self.axes = None
        if not self.axes.is_none() and len(self.axes) != self.ndim:
            self.axes = None
        
        self.metadata = getattr(obj, "metadata", {})

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
        
        self._inherit_meta()
        
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
        
        return result._inherit_meta(ufunc, *inputs, **kwargs)
    
    def _inherit_meta(self, ufunc, *inputs, **kwargs):
        # set attributes for output
        name = "no name"
        dirpath = ""
        input_ndim = -1
        axes = None
        metadata = None
        for input_ in inputs:
            if isinstance(input_, self.__class__):
                name = input_.name
                dirpath = input_.dirpath
                axes = input_.axes
                input_ndim = input_.ndim
                metadata = input_.metadata.copy()
                break

        self.dirpath = dirpath
        self.name = name
        self.metadata = metadata
        
        # set axes
        if axes is None:
            self.axes = None
        elif input_ndim == self.ndim:
            self.axes = axes
        elif input_ndim > self.ndim:
            self.lut = None
            if "axis" in kwargs.keys() and not self.axes.is_none():
                axis = kwargs["axis"]
                self.axes = del_axis(axes, axis)
            else:
                self.axes = None
        else:
            self.axes = None

        return self
    
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
    
    def sort_axes(self):
        """
        Sort image dimensions to ptzcyx-order

        Returns
        -------
        MetaArray
            Sorted image
        """
        arr = np.array(self.axes.argsort())
        order = arr[arr]
        return self.transpose(order)
    
    def iter(self, axes):
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
        for a in self.axes:
            if a in axes:
                iterlist.append(range(self.sizeof(a)))
            else:
                iterlist.append([slice(None)])
                
        selfview = self.value
        
        for sl in itertools.product(*iterlist):
            yield sl, selfview[sl]
            
        
            
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
    
    def reshape(self, *args, **kwargs):
        raise NotImplementedError("Cannot reshape ImgArray")
    
    
    def axisof(self, axisname):
        if (type(axisname) is int):
            return axisname
        else:
            return self.axes.find(axisname)
    
    def xyshape(self):
        return self.sizeof("x"), self.sizeof("y")
    
    def sizeof(self, axis:str):
        return self.shape[self.axes.find(axis)]