from __future__ import annotations
from typing import (
    Any, Mapping, Sequence, Iterable, overload, MutableMapping, TypeVar, TYPE_CHECKING
)
import weakref
import numpy as np

from ._axis import Axis, AxisLike, as_axis, UndefAxis
from ._slicer import Slicer

if TYPE_CHECKING:
    from ._axes_tuple import AxesTuple

ORDER = {"p": 1, "t": 2, "z": 3, "c": 4, "y": 5, "x": 6}
_T = TypeVar("_T")
AxesLike = Iterable[AxisLike]

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
    def __init__(self, value: AxesLike) -> None:
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
    
    def __getattr__(self, key: str) -> Axis:
        """Return an axis with name `key`."""
        try:
            idx = self._axis_list.index(key)
        except:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}.")
        return self._axis_list[idx]
    
    def __iter__(self):
        return iter(self._axis_list)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, self.__class__):
            return other._axis_list == self._axis_list
        return self._axis_list == other

    def __contains__(self, other: AxisLike) -> bool:
        return other in self._axis_list
    
    def __repr__(self):
        s = ", ".join(map(lambda x: repr(str(x)), self))
        return f"{self.__class__.__name__}[{s}]"

    def __hash__(self) -> int:
        """Hash as a tuple of strings."""
        return hash(tuple(map(str, self._axis_list)))
    
    def __add__(self, other: AxesLike) -> Axes:
        return self.__class__(self._axis_list + list(other))
    
    def __radd__(self, other: AxesLike) -> Axes:
        return self.__class__(list(other) + self._axis_list)
    
    def is_sorted(self) -> bool:
        return self == self.sorted()
    
    @overload
    def find(self, axis: str | Axis) -> int:
        ...
    
    @overload
    def find(self, axis: str | Axis, default: _T) -> _T:
        ...
        
    def find(self, axis: str | Axis, *args) -> int:
        """Find the index of an axis."""
        if len(args) > 1:
            raise TypeError(f"Expected 2 or 3 arguments but got {len(args) + 2}.")
        try:
            return self._axis_list.index(axis)
        except ValueError:
            if args:
                return args[0]
            _axes = tuple(str(a) for a in self._axis_list)
            raise ImageAxesError(
                f"Image does not have {axis}-axis: {_axes}."
            ) from None
    
    def sorted(self)-> Axes:
        return self.__class__([self._axis_list[i] for i in self.argsort()])
    
    def argsort(self):
        return np.argsort([ORDER.get(k, 0) for k in self._axis_list])
    
    def has_undef(self) -> bool:
        return any(isinstance(a, UndefAxis) for a in self._axis_list)

    def copy(self):
        """Make a copy of Axes object."""
        return self.__class__(self)

    def replace(self, old: AxisLike, new: AxisLike) -> Axes:
        """
        Create a new Axes object with `old` axis replaced by `new`.
        
        To avoid unexpected effect between images, new scale attribute will be copied.

        Parameters
        ----------
        old : str
            Old symbol.
        new : str
            New symbol.
        """        
        i = self.index(old)
        if new in self._axis_list and old != new:
            raise ImageAxesError(f"Axes {new} already exists: {self}")
        
        if isinstance(new, str):
            new_axis = Axis(new, metadata=self[i].metadata.copy())
        else:
            new_axis = new
        axis_list = self._axis_list.copy()
        axis_list[i] = new_axis
        return self.__class__(axis_list)
    
    def contains(self, chars: AxesLike, *, ignore_undef: bool = False) -> bool:
        """True if self contains all the characters in ``chars``."""
        if ignore_undef:
            return all(a in self._axis_list for a in chars if not isinstance(a, UndefAxis))
        return all(a in self._axis_list for a in chars)
    
    def drop(self, axes: AxisLike | AxesLike | int | Iterable[int]) -> Axes:
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
        """Insert axis at a position."""
        _list = self._axis_list
        _list.insert(idx, as_axis(axis))
        return self.__class__(_list)
    
    def extend(self, axes: AxesLike) -> Axes:
        """Extend axes with given axes."""
        return self + axes

    @overload
    def create_slice(self, sl: Mapping[str, Any] | Slicer) -> tuple[Any, ...]:
        ...
    
    @overload
    def create_slice(self, **kwargs: dict[str, Any]) -> tuple[Any, ...]:
        ...
    
    def create_slice(self, sl = None, /, **kwargs):
        if sl is None:
            sl = kwargs
        elif isinstance(sl, Slicer):
            sl = sl._dict
            
        if not sl:
            raise TypeError("Slice not given.")
        
        sl_list = [slice(None)] * len(self)
    
        for k, v in sl.items():
            idx = self.index(k)
            sl_list[idx] = v
        
        return tuple(sl_list)

    def tuple(self, iterable: Iterable[_T], /) -> AxesTuple[_T] | tuple[_T, ...]:
        from ._axes_tuple import get_axes_tuple
        try:
            out = get_axes_tuple(self)(*iterable)
        except ImageAxesError:
            out = tuple(iterable)
        return out


def _broadcast_two(axes0: AxesLike, axes1: AxesLike) -> Axes:
    if not isinstance(axes0, Axes):
        axes0 = Axes(axes0)
    if not isinstance(axes1, Axes):
        axes1 = Axes(axes1)
    
    arg_idx: list[int] = []
    out = list(axes0)
    for a in axes1:
        if type(a) is UndefAxis:
            raise TypeError("Cannot broadcast Axes with UndefAxis.")
        arg_idx.append(axes0.find(a, -1))
    
    stack = []
    n_insert = 0
    iter = enumerate(arg_idx.copy())
    for i, idx in iter:
        if idx < 0:
            stack.append(i)
        else:
            for j in stack:
                out.insert(idx + n_insert, axes1[j])
                n_insert += 1
            stack.clear()
    for j in stack:
        out.append(axes1[j])
        
    return Axes(out)

def broadcast(*axes_objects: AxesLike) -> Axes:
    """
    Broadcast two or more axes objects and returns their consensus.
    
    This function is designed for more flexible ``numpy`` broadcasting using axes.
    
    Examples
    --------
    >>> broadcast("zyx", "tzyx")  # Axes "tzyx"
    >>> broadcast("tzyx", "tcyx")  # Axes "tzcyx"
    >>> broadcast("yx", "xy")  # Axes "yx"
    """
    n_axes = len(axes_objects)
    
    if n_axes == 2:
        return _broadcast_two(*axes_objects)
    elif n_axes < 2:
        raise TypeError("Less than two axes objects were given.")
    
    it = iter(axes_objects)
    axes0 = next(it)
    for axes1 in it:
        axes0 = _broadcast_two(axes0, axes1)
    return axes0
