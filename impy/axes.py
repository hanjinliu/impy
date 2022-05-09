from __future__ import annotations
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from numbers import Real

ORDER = defaultdict(int, {"p": 1, "t": 2, "z": 3, "c": 4, "y": 5, "x": 6})

class ImageAxesError(RuntimeError):
    """This error is raised when axes is defined in a wrong way."""

class NoneAxes:
    def __bool__(self):
        return False

    def __contains__(self, other):
        return None
    
    def __iter__(self):
        raise StopIteration

NONE = NoneAxes()

class ScaleDict(OrderedDict):
    def __init__(self, d: dict = {}):
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
    
    def __getitem__(self, key: str) -> float:
        return super().__getitem__(key)
    
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
    
    def replace(self, old: str, new: str):
        d = {}
        for k, v in self.items():
            if k == old:
                k = new
            d[k] = v
        return self.__class__(d)


def check_none(func):
    def checked(self: Axes, *args, **kwargs):
        if self.is_none():
            raise ImageAxesError("Axes not defined.")
        return func(self, *args, **kwargs)
    return checked


class Axes:
    def __init__(self, value=None) -> None:
        if value == NONE or value is None:
            self._axes_str: str = NONE
            self.scale: ScaleDict[str, float] = None
            
        elif isinstance(value, str):
            value = value.lower()
            c = Counter(value)
            twice = [a for a, v in c.items() if v > 1]
            if len(twice) > 0:
                raise ImageAxesError(f"{', '.join(twice)} appeared twice")
            self._axes_str = value
            self.scale = {a: 1.0 for a in self._axes_str}
            
        elif isinstance(value, self.__class__):
            self._axes_str = value._axes_str
            self.scale = value.scale
            
        elif isinstance(value, dict):
            if any(len(v) != 1 for v in value.keys()):
                raise ImageAxesError("Only one-character str can be an axis symbol.")
            self._axes_str = "".join(value.keys())
            for k, v in value.items():
                try:
                    self.scale[k] = float(v)
                except ValueError:
                    raise TypeError(f"Cannot set {type(v)} information to axes from a dict.")
            
        else:
            raise ImageAxesError(f"Cannot set {type(value)} to axes.")
    
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        if value is None:
            self._scale = None
        else:
            self._scale = ScaleDict(value)
        
    @check_none
    def __str__(self):
        return self._axes_str
    
    @check_none
    def __len__(self):
        return len(self._axes_str)

    @check_none
    def __getitem__(self, key):
        return self._axes_str[key]
    
    @check_none
    def __iter__(self):
        return self._axes_str.__iter__()
    
    @check_none
    def __next__(self):
        return self._axes_str.__next__()
    
    @check_none
    def __eq__(self, other):
        if isinstance(other, str):
            return self._axes_str == other
        elif isinstance(other, self.__class__):
            return other == self._axes_str

    def __contains__(self, other):
        return other in self._axes_str
    
    def __bool__(self):
        return not self.is_none()
    
    def __repr__(self):
        if self.is_none():
            return "No axes defined"
        else:
            return self._axes_str

    def __hash__(self) -> int:
        return hash(self._axes_str)
    
    def is_none(self):
        return isinstance(self._axes_str, NoneAxes)
    
    @check_none
    def is_sorted(self) -> bool:
        return self._axes_str == self.sorted()
    
    def check_is_sorted(self):
        if self.is_sorted():
            pass
        else:
            raise ImageAxesError(f"Axes must in tzcxy order, but got {self._axes_str}")
    
    @check_none
    def find(self, axis) -> int:
        i = self._axes_str.find(axis)
        if i < 0:
            raise ImageAxesError(f"Image does not have {axis}-axis: {self._axes_str}")
        else:
            return i
    
    @check_none
    def sort(self) -> None:
        self._axes_str = self.sorted()
        return None
    
    def sorted(self)-> str:
        return "".join([self._axes_str[i] for i in self.argsort()])
    
    @check_none
    def argsort(self):
        return np.argsort([ORDER[k] for k in self._axes_str])
    
    def copy(self):
        return self.__class__(self)

    def replace(self, old:str, new:str):
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
        if len(old) != 1 or len(new) != 1:
            raise ValueError("Both `old` and `new` must be single character.")
        if old not in self._axes_str:
            raise ImageAxesError(f"Axes {old} does not exist: {self._axes_str}")
        if new in self._axes_str:
            raise ImageAxesError(f"Axes {new} already exists: {self._axes_str}")
        
        self._axes_str = self._axes_str.replace(old, new)
        scale = self.scale.replace(old, new)
        self.scale = scale
        return None