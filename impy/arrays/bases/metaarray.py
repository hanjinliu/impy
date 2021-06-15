from __future__ import annotations
import numpy as np
from ...axes import Axes, ImageAxesError
from ...func import *
import itertools

class MetaArray(np.ndarray):
    additional_props = ["dirpath", "metadata", "name"]
    NP_DISPATCH = {}
    
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                metadata=None, dtype=None):
        if isinstance(obj, cls):
            return obj
        
        self = np.array(obj, dtype=dtype).view(cls)
        self.dirpath = dirpath
        self.name = name
        
        # MicroManager
        if isinstance(self.name, str) and self.name.endswith("_MMStack_Pos0.ome"):
            self.name = self.name[:-17]
        
        self.axes = axes
        self.metadata = metadata
        return self
    
    @property
    def axes(self):
        return self._axes
    
    @axes.setter
    def axes(self, value):
        if value is None:
            self._axes = Axes()
        else:
            self._axes = Axes(value)
            if self.ndim != len(self._axes):
                raise ImageAxesError("Inconpatible dimensions: "
                                    f"image (ndim={self.ndim}) and axes ({value})")
    
    @property
    def value(self):
        return np.asarray(self)
    
    @property
    def spatial_shape(self):
        return tuple(self.sizeof(a) for a in "zyx" if a in self.axes)
    
    
    @property
    def shape_info(self):
        if self.axes.is_none():
            shape_info = self.shape
        else:
            shape_info = ", ".join([f"{s}({o})" for s, o in zip(self.shape, self.axes)])
        return shape_info
    
    @property
    def scale(self):
        return self.axes.scale
    
    @property
    def scale_unit(self):
        try:
            unit = self.metadata["unit"]
            if unit.startswith(r"\u"):
                unit = "u" + unit[6:]
        except Exception:
            unit = None
        return unit
    
    @scale_unit.setter
    def scale_unit(self, unit):
        if not isinstance(unit, str):
            raise TypeError("Can only set str to scale unit.")
        if isinstance(self.metadata, dict):
            self.metadata["unit"] = unit
        else:
            self.metadata = {"unit": unit}
    
    def set_scale(self, other=None, **kwargs) -> None:
        """
        Set scales of each axis.

        Parameters
        ----------
        other : dict or MetaArray, optional
            New scales. If dict, it should be like {"x": 0.1, "y": 0.1}. If MetaArray, only
            scales of common axes are copied.
        kwargs : 
            This enables function call like set_scale(x=0.1, y=0.1).

        """        
        if self.axes.is_none():
            raise ImageAxesError("Image does not have axes.")
        
        elif isinstance(other, dict):
            # lateral-scale can be set with one keyword.
            if "yx" in other:
                yxscale = other.pop("yx")
                other["x"] = other["y"] = yxscale
            if "xy" in other:
                yxscale = other.pop("xy")
                other["x"] = other["y"] = yxscale
            # check if all the keys are contained in axes.
            for a, val in other.items():
                if a not in self.axes:
                    raise ImageAxesError(f"Image does not have axis {a}.")    
                elif not np.isscalar(val):
                    raise TypeError(f"Cannot set non-numeric value as scales.")
            self.axes.scale.update(other)
            
        elif isinstance(other, MetaArray):
            # Here should not be `self.__class__` because sometimes scales are copied from
            # an object in different branches of subclasses.
            self.set_scale({a: s for a, s in other.scale.items() if a in self.axes})
            
        elif kwargs:
            self.set_scale(dict(kwargs))
            
        else:
            raise TypeError(f"'other' must be str or MetaArray, but got {type(other)}")
        
        return None
    
    
    def __repr__(self):
        return f"\n"\
               f"    shape     : {self.shape_info}\n"\
               f"    dtype     : {self.dtype}\n"\
               f"  directory   : {self.dirpath}\n"\
               f"original image: {self.name}\n"
    
    def __str__(self):
        return self.name
    

    def showinfo(self):
        print(repr(self))
        return None
    
    def _set_additional_props(self, other):
        # set additional properties
        # If `other` does not have it and `self` has, then the property will be inherited.
        for p in self.__class__.additional_props:
            setattr(self, p, getattr(other, p, 
                                     getattr(self, p, 
                                             None)))
    
    def _set_info(self, other, new_axes:str="inherit"):
        self._set_additional_props(other)
        # set axes
        try:
            if new_axes != "inherit":
                self.axes = new_axes
                self.set_scale(other)
            else:
                self.axes = other.axes.copy()
        except ImageAxesError:
            self.axes = None
        
        return None
    
    def __getitem__(self, key):
        if isinstance(key, str):
            # img["t=2;z=4"] ... ImageJ-like, axis-targeted slicing
            sl = self._str_to_slice(key)
            return self.__getitem__(sl)

        if isinstance(key, np.ndarray):
            key = self._broadcast(key)
        
        out = super().__getitem__(key)         # get item as np.ndarray
        keystr = key_repr(key)                 # write down key e.g. "0,*,*"
        
        if isinstance(out, self.__class__):   # cannot set attribution to such as numpy.int32 
            if hasattr(key, "__array__"):
                # fancy indexing will lose axes information
                new_axes = None
                
            elif "new" in keystr:
                # np.newaxis or None will add dimension
                new_axes = None
                
            elif self.axes:
                del_list = [i for i, s in enumerate(keystr.split(",")) if s not in ("*", "")]
                new_axes = del_axis(self.axes, del_list)
            else:
                new_axes = None
                
            out._getitem_additional_set_info(self, keystr=keystr,
                                             new_axes=new_axes, key=key)
        
        return out
    
    def _getitem_additional_set_info(self, other, **kwargs):
        self._set_info(other, kwargs["new_axes"])
        return None
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            # img["t=2;z=4"] ... ImageJ-like method
            sl = self._str_to_slice(key)
            return self.__setitem__(sl, value)
        
        if isinstance(key, MetaArray) and key.dtype == bool and not key.axes.is_none():
            key = add_axes(self.axes, self.shape, key, key.axes)
            
        elif isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
            # img[arr] ... where arr is 2-D boolean array
            key = add_axes(self.axes, self.shape, key)

        super().__setitem__(key, value)
    
    
    def __array_finalize__(self, obj):
        """
        Every time an np.ndarray object is made by numpy functions inherited to ImgArray,
        this function will be called to set essential attributes. Therefore, you can use
        such as img.copy() and img.astype("int") without problems (maybe...).
        """
        if obj is None: return None
        self._set_additional_props(obj)

        try:
            self.axes = getattr(obj, "axes", None)
        except Exception:
            self.axes = None
        if not self.axes.is_none() and len(self.axes) != self.ndim:
            self.axes = None
        
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Every time a numpy universal function (add, subtract, ...) is called,
        this function will be called to set/update essential attributes.
        """
        args_, _ = replace_inputs(self, args, kwargs)

        result = getattr(ufunc, method)(*args_, **kwargs)

        if result is NotImplemented:
            return NotImplemented
        
        result = result.view(self.__class__)
        
        # in the case result is such as np.float64
        if not isinstance(result, self.__class__):
            return result
        
        result._process_output(ufunc, args, kwargs)
        
        return result
    
    def _inherit_meta(self, obj, ufunc, **kwargs):
        """
        Copy axis, history etc. from obj.
        This is called in __array_ufunc__(). Unlike _set_info(), keyword `axis` must be
        considered because it changes `ndim`.
        """
        if "axis" in kwargs.keys() and not obj.axes.is_none():
            new_axes = del_axis(obj.axes, kwargs["axis"])
        else:
            new_axes = "inherit"
        self._set_info(obj, new_axes=new_axes)
        return self
    
    def __array_function__(self, func, types, args, kwargs):
        """
        Every time a numpy function (np.mean...) is called, this function will be called. Essentially numpy
        function can be overloaded with this method.
        """
        if (func in self.__class__.NP_DISPATCH and 
            all(issubclass(t, MetaArray) for t in types)):
            return self.__class__.NP_DISPATCH[func](*args, **kwargs)
        
        args_, _ = replace_inputs(self, args, kwargs)

        result = func(*args_, **kwargs)

        if result is NotImplemented:
            return NotImplemented
        
        if isinstance(result, (tuple, list)):
            _as_meta_array = lambda a: a.view(self.__class__)._process_output(func, args, kwargs) \
                if type(a) is np.ndarray else a
            result = type(result)(_as_meta_array(r) for r in result)
            
        else:
            result = result.view(self.__class__)
            # in the case result is such as np.float64
            if isinstance(result, self.__class__):
                result._process_output(func, args, kwargs)
        
        return result
    
    def _process_output(self, func, args, kwargs):
        # find the largest MetaArray. Largest because of broadcasting.
        arr = None
        for arg in args:
            if isinstance(arg, self.__class__):
                if arr is None or arr.ndim < arg.ndim:
                    arr = arg
                    
        if isinstance(arr, self.__class__):
            self._inherit_meta(arr, func, **kwargs)
        
        return self
        
    
    @classmethod
    def implements(cls, numpy_function):
        """
        Add functions to NP_DISPATCH so that numpy functions can be overloaded.
        """        
        def decorator(func):
            cls.NP_DISPATCH[numpy_function] = func
            return func
        return decorator
    
    def _str_to_slice(self, string:str):
        """
        get subslices using ImageJ-like format.
        e.g. 't=3:, z=1:5', 't=1, z=:7'
        """
        keylist = string.split(";")
        sl_list = [slice(None)]*self.ndim
        
        for k in keylist:
            # e.g. k = "t=4:7"
            axis, sl_str = k.split("=")
            sl_list[self.axisof(axis)] = str_to_slice(sl_str)

        return tuple(sl_list)
    
    def sort_axes(self):
        """
        Sort image dimensions to ptzcyx-order

        Returns
        -------
        MetaArray
            Sorted image
        """
        order = self.axes.argsort()
        return self.transpose(order)
    
    
    def iter(self, axes, israw=False, exclude=""):
        """
        Iteration along axes. Unlike self.iter(axes), this function yields subclass objects
        so that this function is slower but accessible to attributes such as labels.

        Parameters
        ----------
        axes : str or int
            On which axes iteration is performed. Or the number of spatial dimension.
        israw : bool, default is False
            If True, MetaArray will be returned. If False, np.ndarray will be returned.
        exclude : str, optional
            Which axes will be excluded in output. For example, self.axes="tcyx" and 
            exclude="c" then the axes of output will be "tyx" and slice is also correctly 
            arranged.
            
        Yields
        -------
        slice and (np.ndarray or MetaArray)
            slice and a subimage=self[sl]
        """     
        iterlist = self._get_iterlist(axes)
        selfview = self if israw else self.value
        it = itertools.product(*iterlist)
        c = 0 # counter
        for sl in it:
            if len(exclude) == 0:
                outsl = sl
            else:
                outsl = tuple(s for i, s in enumerate(sl) 
                              if self.axes[i] not in exclude)
            yield outsl, selfview[sl]
            c += 1
            
        # if iterlist = []
        if c == 0:
            outsl = (slice(None),) * (self.ndim - len(exclude))
            yield outsl, selfview
            
    
    def _get_iterlist(self, axes):
        """
        If axes="tzc", then equivalent to following pseudo code:
        for t in all_t:
            for z in all_z:
                for c in all_c:
                    yield self[t, z, c, ...]
        """        
                
        iterlist = []
        for a in self.axes:
            if a in axes:
                iterlist.append(range(self.sizeof(a)))
            else:
                iterlist.append([slice(None)])
        return iterlist
            
    
    def transpose(self, axes):
        """
        change the order of image dimensions.
        'axes' will also be arranged.
        """
        out = super().transpose(axes)
        if self.axes.is_none():
            new_axes = None
        else:
            new_axes = "".join([self.axes[i] for i in list(axes)])
        out._set_info(self, new_axes=new_axes)
        return out
    
    
    def axisof(self, axisname):
        if type(axisname) is int:
            return axisname
        else:
            return self.axes.find(axisname)
    
    def sizeof(self, axis:str):
        return self.shape[self.axes.find(axis)]
    
    def sizesof(self, axes:str):
        return tuple(self.sizeof(a) for a in axes)

    def _broadcast(self, value):
        """
        More flexible broadcasting. If `self` has "zcyx"-axes and `value` has "zyx"-axes, then
        they should be broadcasted by stacking `value` along "c"-axes
        """        
        if isinstance(value, MetaArray) and not value.axes.is_none():
            value = add_axes(self.axes, self.shape, value, value.axes)
        elif isinstance(value, np.ndarray):
            try:
                if self.sizesof("yx") == value.shape:
                    value = add_axes(self.axes, self.shape, value)
            except ImageAxesError:
                pass
        return value
    
    def __add__(self, value):
        value = self._broadcast(value)
        return super().__add__(value)
    
    def __sub__(self, value):
        value = self._broadcast(value)
        return super().__sub__(value)
    
    def __mul__(self, value):
        value = self._broadcast(value)
        return super().__mul__(value)
    
    def __truediv__(self, value):
        value = self._broadcast(value)
        return super().__truediv__(value)