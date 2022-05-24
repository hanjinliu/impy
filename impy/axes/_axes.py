from __future__ import annotations
from typing import Sequence, Iterable, overload, MutableMapping
import weakref
import numpy as np

from ._axis import Axis, AxisLike, as_axis, UndefAxis

ORDER = {"p": 1, "t": 2, "z": 3, "c": 4, "y": 5, "x": 6}


class ImageAxesError(RuntimeError):
    """This error is raised when axes is defined in a wrong way."""


class ScaleView(MutableMapping[str, float]):
    _axes_ref: weakref.ReferenceType["Axes"]
    
    def __init__(self, axes: "Axes"):
        super().__setattr__("_axes_ref", weakref.ref(axes))
    
    @property
    def axes(self) -> "Axes":
        _axes = self._axes_ref()
        if _axes is None:
            raise RuntimeError("Axes object is deleted.")
        return _axes
        
    def __getattr__(self, key: str) -> float:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Image does not have {key} axes.")
    
    def __iter__(self) -> Iterable[str]:
        return iter(self.axes)
    
    def values(self) -> Iterable[float]:
        return map(lambda a: a.scale, self.axes)
    
    def __len__(self) -> int:
        return len(self.axes)
    
    @overload
    def __getitem__(self, key: str | int) -> float:
        ...
    
    @overload
    def __getitem__(self, key: slice) -> list[float]:
        ...
    
    def __getitem__(self, key):
        if isinstance(key, (str, Axis)):
            if key not in self.axes:
                raise KeyError(key)
            return self.axes[key].scale
        elif isinstance(key, (int, slice)):
            return [a.scale for a in self.axes][key]
        else:
            raise TypeError(f"Cannot slice {type(self)} with {type(key)}.")
    
    def __setitem__(self, key: str, value: float) -> None:
        self.axes[key].scale = value
    
    def __delitem__(self, key: str) -> None:
        raise AttributeError("Cannot delete item.")
    
    __setattr__ = __setitem__
    
    def __list__(self) -> list[float]:
        axes = sorted(self.keys(), key=lambda a: ORDER.get(a, 0))
        return [self[a] for a in axes]
    
    def __array__(self, dtype=None):
        return np.array(self.__list__(), dtype=dtype)
    
    def __repr__(self) -> str:
        kwargs = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({kwargs})"
    
    def copy(self) -> ScaleView:
        return self.__class__(self.axes)

class Axes(Sequence[Axis]):
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
            
        else:
            self._axis_list = [a.__copy__() for a in value._axis_list]
            
    @classmethod
    def undef(cls, ndim: int):
        """Construct an Axes object initialized with undefined axes."""
        return cls([UndefAxis() for _ in range(ndim)])
        
    @property
    def scale(self) -> ScaleView:
        return ScaleView(self)
    
    @scale.setter
    def scale(self, value: dict[str, float]) -> None:
        for k, v in value:
            self[k].scale = v
        
    def __str__(self):
        return "".join(map(str, self._axis_list))
    
    def __len__(self):
        return len(self._axis_list)

    @overload
    def __getitem__(self, key: int | str | Axis) -> Axis:
        ...
        
    @overload
    def __getitem__(self, key: slice) -> Axes:
        ...
        
    def __getitem__(self, key):
        """Get an axis."""
        if isinstance(key, (str, Axis)):
            return self._axis_list[self.find(key)]
        elif isinstance(key, slice):
            l = self._axis_list[key]
            return self.__class__(l)
        else:
            return self._axis_list[key]
    
    def __setitem__(self, key: int | str, value) -> None:
        if isinstance(key, str):
            old = key
        else:
            old = self[key]
        self.replace(old, value)
        return None
    
    def __delitem__(self, key: int | str) -> None:
        if isinstance(key, str):
            old = key
        else:
            old = self[key]
        self.drop(old)
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
        """Hash as a tuple of strings."""
        return hash(tuple(map(str, self._axis_list)))
    
    def __add__(self, other: Iterable[AxisLike]) -> Axes:
        return self.__class__(self._axis_list + list(other))
    
    def __radd__(self, other: Iterable[AxisLike]) -> Axes:
        return self.__class__(list(other) + self._axis_list)
    
    def is_sorted(self) -> bool:
        return self == self.sorted()
    
    def find(self, axis: str) -> int:
        try:
            return self._axis_list.index(axis)
        except ValueError:
            _axes = tuple(str(a) for a in self._axis_list)
            raise ImageAxesError(
                f"Image does not have {axis}-axis: {_axes}."
            ) from None
    
    def sort(self) -> None:
        self._axis_list = self.sorted()
        return None
    
    def sorted(self)-> list[Axis]:
        return [self._axis_list[i] for i in self.argsort()]
    
    def argsort(self):
        return np.argsort([ORDER.get(k, 0) for k in self._axis_list])
    
    def copy(self):
        """Make a copy of Axes object."""
        return self.__class__(self)

    def replace(self, old: AxisLike, new: AxisLike):
        """
        Replace axis symbol. To avoid unexpected effect between images, new scale
        attribute will be copied.

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
        
        if isinstance(new, str):
            new_axis = Axis(new, metadata=self[i].metadata.copy())
        else:
            new_axis = new
        self._axis_list[i] = new_axis
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
    
    def insert(self, idx: int = -1, axis: AxisLike = "#") -> Axes:
        _list = self._axis_list
        _list.insert(idx, as_axis(axis))
        return self.__class__(_list)
    
    def extend(self, axes: Iterable[AxisLike]) -> Axes:
        return self + axes
