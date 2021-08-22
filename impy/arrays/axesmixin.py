from ..axes import Axes, ImageAxesError
import numpy as np
import re

class AxesMixin:
    """
    Abstract class with shape, ndim and axes are defined.
    """    
    @property
    def shape_info(self):
        if self.axes.is_none():
            shape_info = self.shape
        else:
            shape_info = ", ".join([f"{s}({o})" for s, o in zip(self.shape, self.axes)])
        return shape_info
    
    @property
    def spatial_shape(self):
        return tuple(self.sizeof(a) for a in "zyx" if a in self.axes)
    
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
                                    f"array (ndim={self.ndim}) and axes ({value})")
        
    @property
    def scale(self):
        return self.axes._scale
    
    @scale.setter
    def scale(self, value:dict):
        if not isinstance(value, dict):
            raise TypeError(f"Cannot set scale using {type(value)}.")
        return self.set_scale(value)
    
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
    
    def __repr__(self):
        return "\n" + "\n".join(f"{k}: {v}" for k, v in self._repr_dict_().items()) + "\n"
    
    def _repr_html_(self):
        strs = []
        for k, v in self._repr_dict_().items():
            v = re.sub("->", "<br>&rarr; ", str(v))
            strs.append(f"<tr><td width=\"100\" >{k}</td><td>{v}</td></tr>")
        main = "<table border=\"1\">" + "".join(strs) + "</table>"
        html = f"""
        <head>
            <style>
                #wrapper {{
                    height: 140px;
                    width: 500px;
                    overflow-y: scroll;
                }}
            </style>
        </head>

        <body>
            <div id="wrapper">
                {main}
            </div>
        </body>
        """
        return html
        
    def axisof(self, symbol):
        if type(symbol) is int:
            return symbol
        else:
            return self.axes.find(symbol)
    
    
    def sizeof(self, axis:str):
        return self.shape[self.axes.find(axis)]
    
    def sizesof(self, axes:str):
        return tuple(self.sizeof(a) for a in axes)
    

    def set_scale(self, other=None, **kwargs) -> None:
        """
        Set scales of each axis.

        Parameters
        ----------
        other : dict or object with axes
            New scales. If dict, it should be like {"x": 0.1, "y": 0.1}. If MetaArray, only
            scales of common axes are copied.
        kwargs : 
            This enables function call like set_scale(x=0.1, y=0.1).

        """        
        if self.axes.is_none():
            raise ImageAxesError("Image does not have axes.")
        
        elif isinstance(other, dict):
            # voxel-scale can be set with one keyword.
            if "zyx" in other:
                zyxscale = other.pop("zyx")
                other["x"] = other["y"] = other["z"] = zyxscale
            if "xyz" in other:
                zyxscale = other.pop("xyz")
                other["x"] = other["y"] = other["z"] = zyxscale
            # lateral-scale can be set with one keyword.
            if "yx" in other:
                yxscale = other.pop("yx")
                other["x"] = other["y"] = yxscale
            elif "xy" in other:
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
        
        elif hasattr(other, "scale"):
            self.set_scale({a: s for a, s in other.scale.items() if a in self.axes})
        
        else:
            raise TypeError(f"'other' must be str or LazyImgArray, but got {type(other)}")
        
        return None