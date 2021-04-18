from collections import defaultdict, Counter
import numpy as np

ORDER = defaultdict(int, {"p": 1, "t": 2, "z": 3, "c": 4, "y": 5, "x": 6})

class ImageAxesError(Exception):
    pass

class NoneAxes:
    def __bool__(self):
        return False

    def __contains__(self, other):
        return None
    
    def __iter__(self):
        raise StopIteration

NONE = NoneAxes()


def check_none(func):
    def checked(self, *args, **kwargs):
        if self.is_none():
            raise ImageAxesError("Axes not defined.")
        return func(self, *args, **kwargs)
    return checked


class Axes:
    def __init__(self, value=None) -> None:
        if value == NONE or value is None:
            self.axes = NONE
            self.tag = None
        elif isinstance(value, str):
            value = value.lower()
            c = Counter(value)
            twice = [a for a, v in c.items() if v > 1]
            if len(twice) > 0:
                raise ImageAxesError(f"{', '.join(twice)} appeared twice")
            self.axes = value
            self.tag = {a: 1.0 for a in self.axes}
            
        elif isinstance(value, self.__class__):
            self.axes = value.axes
            self.tag = value.tag
            
        elif isinstance(value, dict):
            if any(len(v)!=1 for v in value.keys()):
                raise ImageAxesError("Only one-character str can be an axis symbol.")
            self.axes = "".join(value.keys())
            self.tag = value
            
        else:
            raise ImageAxesError(f"Cannot set {type(value)} to axes.")
    
        
    @check_none
    def __str__(self):
        return self.axes
    
    @check_none
    def __len__(self):
        return len(self.axes)

    @check_none
    def __getitem__(self, key):
        return self.axes[key]
    
    @check_none
    def __iter__(self):
        return self.axes.__iter__()
    
    @check_none
    def items(self):
        for a in self.axes:
            yield a, self.tag[a]
    
    @check_none
    def __next__(self):
        return self.axes.__next__()
    
    @check_none
    def __eq__(self, other):
        if isinstance(other, str):
            return self.axes == other
        elif isinstance(other, self.__class__):
            return other == self.axes

    def __contains__(self, other):
        return other in self.axes
    
    def __bool__(self):
        return not self.is_none()
    
    def __repr__(self):
        if self.is_none():
            return "No axes defined"
        else:
            return self.axes

    def is_none(self):
        return isinstance(self.axes, NoneAxes)

    def to_none(self):
        self.axes = NONE
        return None
    
    @check_none
    def is_sorted(self) -> bool:
        return self.axes == self.sorted()
    
    def check_is_sorted(self):
        if self.is_sorted():
            pass
        else:
            raise ImageAxesError(f"Axes must in tzcxy order, but got {self.axes}")
    
    @check_none
    def find(self, axis) -> int:
        i = self.axes.find(axis)
        if i < 0:
            raise ImageAxesError(f"Image does not have {axis}-axis: {self.axes}")
        else:
            return i
    
    @check_none
    def sort(self) -> None:
        self.axes = self.sorted()
        return None
    
    def sorted(self)-> str:
        return "".join([self.axes[i] for i in self.argsort()])
    
    @check_none
    def argsort(self):
        return np.argsort([ORDER[k] for k in self.axes])
    
    def copy(self):
        return self.__class__(self)
