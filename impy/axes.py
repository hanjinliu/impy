from __future__ import annotations
from collections import defaultdict, OrderedDict
from copy import copy
from typing import Any, MutableSequence, Union, Iterable, Sequence, overload, TYPE_CHECKING
import numpy as np
from numbers import Real

if TYPE_CHECKING:
    from typing_extensions import Self

ORDER = defaultdict(int, {"p": 1, "t": 2, "z": 3, "c": 4, "y": 5, "x": 6})

class ImageAxesError(RuntimeError):
    """This error is raised when axes is defined in a wrong way."""


class Axis:
    """
    An axis object.
    
    This object behaves like a length-1 string as much as possible.
    
    Parameters 
    ----------
    name : str
        Name of axis.
    """
    
    def __init__(self, name: str):
        self._name = str(name)
    
    def __str__(self) -> str:
        """String representation of the axis."""
        return self._name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._name!r}]"
    
    def __hash__(self) -> int:
        """Hash as a string."""
        return hash(str(self))
    
    def __eq__(self, other) -> bool:
        return str(self) == other
    
    def __copy__(self) -> Self:
        return self.__class__(self._name)
    
    def __add__(self, other: str) -> str:
        return self._name + other
    
    def __radd__(self, other: str) -> str:
        return other + self._name
    
    def __lt__(self, other) -> bool:
        """To support alphabetic ordering."""
        return str(self) < str(other)
    
    def __len__(self) -> int:
        return len(str(self))

AxisLike = Union[str, Axis]
    
class AnnotatedAxis(Axis):
    def __init__(self, name: str, metadata: dict[str, Any] = {}):
        super().__init__(name)
        self._metadata = metadata.copy()
    
    def __copy__(self) -> Self:
        return self.__class__(self._name, self._metadata.copy())

class UndefAxis(Axis):
    """Undefined axis object."""
    def __init__(self, name: str = "#"):
        super().__init__(name)
    
    def __repr__(self) -> str:
        return "#undef"
    
    def __hash__(self) -> str:
        return id(self)
    
    def __eq__(self, other) -> bool:
        return False

class ScaleDict(OrderedDict[str, float]):
    def __init__(self, d: dict[str, float] = {}):
        for k, v in d.items():
            if v <= 0:
                raise ValueError(f"Cannot set negative scale: {k}={v}.")
            super().__setitem__(k, v)
        
    def __getattr__(self, key: str) -> float:
        """
        To enable such as scale.x or scale.y. Simply this can be achieved by
        
            .. code-block:: python
                
                __getattr__ = dict.__getitem__
        
        However, we also want to convert it to np.ndarray for compatibility with napari's 
        "scale" arguments. Because __getattr__ is called inside np.ndarray, it expected to 
        raise AttributeError rather than KeyError.
        """        
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Image does not have {key} axes.")
    
    @overload
    def __getitem__(self, key: str | int) -> float:
        ...
    
    @overload
    def __getitem__(self, key: slice) -> list[float]:
        ...
    
    def __getitem__(self, key: str | int | slice) -> float:
        if isinstance(key, (str, Axis)):
            return super().__getitem__(str(key))
        elif isinstance(key, (int, slice)):
            return list(self.values())[key]
        else:
            raise TypeError(f"Cannot slice {type(self)} with {type(key)}.")
    
    def __setitem__(self, key: str, value: Real) -> None:
        value = float(value)
        if key not in self.keys():
            raise ImageAxesError(f"Image does not have {key} axes.")
        elif value <= 0:
            raise ValueError(f"Cannot set negative scale: {key}={value}.")
        return super().__setitem__(key, value)
    
    __setattr__ = __setitem__
    
    def __list__(self) -> list[Real]:
        axes = sorted(self.keys(), key=lambda a: ORDER[a])
        return [self[a] for a in axes]
    
    def __array__(self, dtype=None):
        return np.array(self.__list__(), dtype=dtype)
    
    def __repr__(self) -> str:
        kwargs = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({kwargs})"
    
    def copy(self) -> ScaleDict:
        return self.__class__(self)
    
    def keys(self):
        yield from map(str, super().keys())
    
    def replace(self, old: AxisLike, new: AxisLike):
        d = {}
        for k, v in self.items():
            if k == old:
                k = new
            d[k] = v
        return self.__class__(d)


def as_axis(obj: Any) -> Axis:
    if isinstance(obj, str):
        if obj == "#":
            axis = UndefAxis()
        else:
            axis = Axis(obj)
    elif isinstance(obj, Axis):
        axis = copy(obj)
    else:
        raise TypeError(f"Cannot use {type(obj)} as an axis.")
    return axis

class Axes(MutableSequence[Axis]):
    """
    A sequence of axes.
    
    This object behaves like a string as much as possible.
    """
    def __init__(self, value: Iterable[AxisLike]) -> None:
        if not isinstance(value, self.__class__):
            inputs = list(map(as_axis, value))
            ndim = len(inputs)
            
            # check duplication
            if ndim > len(set(inputs)):
                raise ImageAxesError(f"Duplicated axes found: {inputs}.")
            
            self._axis_list = inputs
            self.scale = {a: 1.0 for a in self._axis_list}
            
        else:
            self._axis_list = value._axis_list.copy()
            self.scale = value.scale
            
    @classmethod
    def undef(cls, ndim: int):
        """Construct an Axes object initialized with undefined axes."""
        return cls([UndefAxis() for _ in range(ndim)])
        
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        if value is None:
            self._scale = None
        else:
            self._scale = ScaleDict(value)
        
    def __str__(self):
        return "".join(map(str, self._axis_list))
    
    def __len__(self):
        return len(self._axis_list)

    def __getitem__(self, key):
        """Get an axis."""
        return self._axis_list[key]
    
    def __setitem__(self, key: int, value) -> None:
        self.replace(self[key], value)
        return None
    
    def __delitem__(self, key: int) -> None:
        self.drop(self[key])
        return None
    
    def __iter__(self):
        return iter(self._axis_list)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, self.__class__):
            return other._axis_list == self._axis_list
        return self._axis_list == other

    def __contains__(self, other):
        return other in self._axis_list
    
    def __repr__(self):
        s = ", ".join(map(repr, self))
        return f"{self.__class__.__name__}[{s}]"

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __add__(self, other: Iterable[AxisLike]) -> Axes:
        return self.__class__(self._axis_list + list(other))
    
    def __radd__(self, other: Iterable[AxisLike]) -> Axes:
        return self.__class__(list(other) + self._axis_list)
    
    def is_sorted(self) -> bool:
        return self == self.sorted()
    
    def check_is_sorted(self):
        if not self.is_sorted():
            raise ImageAxesError(f"Axes must in tzcxy order, but got {self._axis_list}")
    
    def find(self, axis: str) -> int:
        i = self._axis_list.index(axis)
        if i < 0:
            raise ImageAxesError(f"Image does not have {axis}-axis: {self._axis_list}")
        else:
            return i
    
    def sort(self) -> None:
        self._axis_list = self.sorted()
        return None
    
    def sorted(self)-> list[Axis]:
        return [self._axis_list[i] for i in self.argsort()]
    
    def argsort(self):
        return np.argsort([ORDER.get(k, 0) for k in self._axis_list])
    
    def has_undef(self) -> bool:
        """True if the object has at least one undefined axis."""
        return any(isinstance(a, UndefAxis) for a in self)
    
    def copy(self):
        """Make a copy of Axes object."""
        return self.__class__(self)

    def replace(self, old: AxisLike, new: AxisLike):
        """
        Replace axis symbol. To avoid unexpected effect between images, new scale attribute
        will be copied.

        Parameters
        ----------
        old : str
            Old symbol.
        new : str
            New symbol.
        """        
        i = self.index(old)
        if new in self._axis_list:
            raise ImageAxesError(f"Axes {new} already exists: {self}")
        
        scale = self.scale.replace(old, new)
        self._axis_list[i] = new
        self.scale = scale
        return None
    
    def contains(self, chars: Iterable[AxisLike]) -> bool:
        """True if self contains all the characters in ``chars``."""
        return all(a in self._axis_list for a in chars)
    
    def drop(self, axes) -> Axes:
        """Drop an axis or a list of axes."""
        if not isinstance(axes, (list, tuple, str)):
            axes = (axes,)
        
        drop_list = []
        for a in axes:
            if isinstance(a, int):
                drop_list.append(self._axis_list[a])
            else:
                drop_list.append(a)
        
        return Axes(a for a in self._axis_list if a not in drop_list)
    
    def insert(self, idx: int = -1, axis: AxisLike = "#"):
        _list = self._axis_list
        _list.insert(idx, as_axis(axis))
        return self.__class__(_list)
    
    def extend(self, axes: Iterable[AxisLike]) -> Axes:
        return self + axes
