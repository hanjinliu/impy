from ..func import *
from dask import array as da
from ..axes import Axes, ImageAxesError
from .imgarray import ImgArray

class LazyImgArray:
    max_byte = 2e9
    def __init__(self, obj: da.core.Array, name=None, axes=None, dirpath=None, history=None, metadata=None):
        if not isinstance(obj, da.core.Array):
            raise TypeError("obj must be dask array")
        self.img = obj
        self.dirpath = dirpath
        self.name = name
        
        # MicroManager
        if isinstance(self.name, str) and self.name.endswith("_MMStack_Pos0.ome"):
            self.name = self.name[:-17]
        
        self.axes = axes
        self.metadata = metadata
        self.history = history
        
    @property
    def ndim(self):
        return self.img.ndim
    
    @property
    def shape(self):
        return self.img.shape
    
    @property
    def dtype(self):
        return self.img.dtype
    
    @property
    def size(self):
        return self.img.size
    
    @property
    def axes(self):
        return self._axes
    
    @property
    def itemsize(self):
        return self.img.itemsize
    
    @axes.setter
    def axes(self, value):
        if value is None:
            self._axes = Axes()
        else:
            self._axes = Axes(value)
            if self.ndim != len(self._axes):
                raise ImageAxesError("Inconpatible dimensions: "
                                    f"image (ndim={self.ndim}) and axes ({value})")
    
    def __getitem__(self, key):
        if isinstance(key, str):
            key = axis_targeted_slicing(self.img, self.axes, key)
        return self.img[key]
    
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
            
        elif kwargs:
            self.set_scale(dict(kwargs))
            
        else:
            raise TypeError(f"'other' must be str or MetaArray, but got {type(other)}")
        
        return None
    
    
    def _repr_dict_(self):
        return {"    shape     ": self.shape_info,
                "    dtype     ": self.dtype,
                "  directory   ": self.dirpath,
                "original image": self.name}
    
    def __repr__(self):
        return "\n" + "\n".join(f"{k}: {v}" for k, v in self._repr_dict_().items()) + "\n"
    
    @property
    def data(self) -> ImgArray:
        total_byte = self.size * self.itemsize
        if total_byte > self.__class__.max_byte:
            raise RuntimeError(f"Too large: {total_byte*1e-9:.2f} GB")
        img = self.img.compute().compute().view(ImgArray)
        for attr in ["name", "dirpath", "axes", "metadata", "history"]:
            setattr(img, attr, getattr(self, attr, None))
        return img
    
    def axisof(self, axisname):
        if type(axisname) is int:
            return axisname
        else:
            return self.axes.find(axisname)