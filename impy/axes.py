from __future__ import annotations
from collections import defaultdict, OrderedDict
from typing import Hashable, Iterable, Sequence, overload
import numpy as np
from numbers import Real

ORDER = defaultdict(int, {"p": 1, "t": 2, "z": 3, "c": 4, "y": 5, "x": 6})

class ImageAxesError(RuntimeError):
    """This error is raised when axes is defined in a wrong way."""

class UndefAxis:
    def __str__(self) -> str:
        return "#"
    
    def __hash__(self) -> str:
        return id(self)

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
        if isinstance(key, str):
            return super().__getitem__(key)
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
    
    def replace(self, old: Hashable, new: Hashable):
        d = {}
        for k, v in self.items():
            if k == old:
                k = new
            d[k] = v
        return self.__class__(d)


class Axes(Sequence[Hashable]):
    def __init__(self, value: Iterable[Hashable]) -> None:
        if not isinstance(value, self.__class__):
            inputs = list(value)
            ndim = len(inputs)
            
            # replace undef.
            if "#" in inputs:
                for i in range(ndim):
                    if inputs[i] == "#":
                        inputs[i] = UndefAxis()
            
            # check duplication
            if ndim > len(set(inputs)):
                raise ImageAxesError(f"Duplicated axes found: {inputs}.")
            
            self._axis_list = list(value)
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
        return self._axis_list[key]
    
    def __iter__(self):
        return iter(self._axis_list)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self._axis_list == list(other)
        elif isinstance(other, self.__class__):
            return other._axis_list == self._axis_list
        return False

    def __contains__(self, other):
        return other in self._axis_list
    
    def __repr__(self):
        return f"{self.__class__.__name__}['{self}']"

    def __hash__(self) -> int:
        return hash(str(self))
    
    def is_sorted(self) -> bool:
        return str(self) == self.sorted()
    
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
        self._axis_list = list(self.sorted())
        return None
    
    def sorted(self)-> str:
        return "".join([self._axis_list[i] for i in self.argsort()])
    
    def argsort(self):
        return np.argsort([ORDER.get(k, 0) for k in self._axis_list])
    
    def has_undef(self) -> bool:
        return any(isinstance(a, UndefAxis) for a in self)
    
    def copy(self):
        return self.__class__(self)

    def replace(self, old: str, new: str):
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
    
    def contains(self, chars: Iterable[str]) -> bool:
        """True if self contains all the characters in ``chars``."""
        return all(a in self._axis_list for a in chars)