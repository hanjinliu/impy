from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    overload,
    MutableMapping,
)
import numpy as np
import itertools
import re
from warnings import warn
from collections import namedtuple

from ..utils.axesop import switch_slice
from ..axes import Axes, ImageAxesError, ScaleView, AxisLike
from .._types import Slices, Dims

if TYPE_CHECKING:
    from typing_extensions import Self, Literal

class AxesMixin:
    """Abstract class with shape, ndim and axes are defined."""
    
    _INHERIT = object()
    _axes: Axes
    ndim: int
    shape: tuple[int, ...]
    value: Any
    
    @property
    def shape_info(self) -> str:
        shape_info = ", ".join([f"{s}({o})" for s, o in zip(self.shape, self.axes)])
        return shape_info
    
    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.sizeof(a) for a in "zyx" if a in self.axes)
    
    @property
    def axes(self) -> Axes:
        """Axes of the array."""
        return self._axes
    
    @axes.setter
    def axes(self, value: Iterable[AxisLike] | None):
        if value is None:
            self._axes = Axes.undef(self.ndim)
        else:
            axes = Axes(value)
            if self.ndim != len(axes):
                raise ImageAxesError(
                    "Inconpatible dimensions: "
                    f"array (ndim={self.ndim}) and axes ({value})"
                )
            self._axes = axes
    
    @property
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError()
    
    @property
    def scale(self) -> ScaleView:
        return self.axes.scale
    
    @scale.setter
    def scale(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"Cannot set scale using {type(value)}.")
        return self.set_scale(value)
    
    @property
    def scale_unit(self) -> str:
        units = set(a.unit for a in self.axes if str(a) in "zyx")
        if len(units) == 0:
            return self.axes[-1].unit
        elif len(units) == 1:
            return list(units)[0]
        else:
            warn(f"Inconsistent spatial unit: {units}.")
            return list(units)[-1]
    
    @scale_unit.setter
    def scale_unit(self, unit) -> None:
        unit = str(unit)
        for a in self.axes:
            if str(a) in ["z", "y", "x"]:
                a.unit = unit
    
    def _repr_dict_(self) -> dict[str, Any]:
        raise NotImplementedError()
        
    def __repr__(self) -> str:
        info = "\n".join(f"{k:^16}: {v}" for k, v in self._repr_dict_().items())
        return f"{self.__class__.__name__} of\n{info}\n"
    
    def _repr_html_(self) -> str:
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
        
    def axisof(self, symbol) -> int:
        if type(symbol) is int:
            return symbol
        else:
            return self.axes.find(symbol)
    
    
    def sizeof(self, axis: str) -> int:
        return getattr(self.shape, axis)
    
    def sizesof(self, axes: str) -> tuple[int, ...]:
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
        
        if isinstance(other, MutableMapping):
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

            for k, v in other.items():
                self.axes[k].scale = v
            
        elif kwargs:
            self.set_scale(dict(kwargs))
        
        elif hasattr(other, "scale"):
            self.set_scale(
                {a: s for a, s in other.scale.items() if a in self.axes}  # type: ignore
            )
        
        else:
            raise TypeError(
                f"'other' must be str or axes supported object, but got {type(other)}"
            )
        
        return None
    
    def set_axis_labels(self, _dict: MutableMapping[str, float] = None, **kwargs) -> str:
        if _dict is None:
            _dict = kwargs
        for k, v in _dict.items():
            if self.sizeof(k) != len(v):
                raise ValueError(f"Lengths of axis {k} and labels {v} don't match.")
        for k, v in _dict.items():
            self.axes[k].labels = v
        return None
    
    @overload
    def iter(
        self,
        axes: str,
        israw: Literal[False] = False, 
        exclude: Dims = "",
    ) -> Iterator[tuple[Slices, np.ndarray]]:
        ...
    
    @overload
    def iter(
        self,
        axes: str,
        israw: Literal[True] = False, 
        exclude: Dims = "",
    ) -> Iterator[tuple[Slices, Self]]:
        ...
    
    def iter(self, axes, israw = False, exclude = ""):
        """
        Iteration along axes. If axes="tzc", then equivalent to following pseudo code:
        
            .. code-block::
            
                for t in all_t:
                    for z in all_z:
                        for c in all_c:
                            yield self[t, z, c, ...]

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
        slice and (np.ndarray or AxesMixin)
            slice and a subimage=self[sl]
        """     
        iterlist = switch_slice(
            axes=axes, 
            all_axes=self.axes,
            ifin=[range(s) for s in self.shape], 
            ifnot=[(slice(None),)] * self.ndim
        )
        exclude = list(exclude)
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

_AxesShapes: dict[str, tuple] = {}

def get_axes_tuple(self: AxesMixin):
    axes = self.axes
    try:
        return _AxesShapes[axes]
    except KeyError:
        fields = []
        for i, a in enumerate(self.axes):
            s = str(a)
            if s.isidentifier():
                fields.append(s)
            else:
                fields.append(f"axis_{i}")
        tup = namedtuple("AxesShape", fields)
        _AxesShapes[axes] = tup
        return tup