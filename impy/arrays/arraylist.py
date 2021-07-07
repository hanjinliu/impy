from .bases import MetaArray
from ..utilcls import Progress

class ArrayList:
    def __init__(self, iterable=None):
        if iterable is not None:
            self._arrays = list(iterable)
            self.type = type(self._arrays[0])
            self._check()
        else:
            self._arrays = []
            self.type = None
        
    @property
    def arrays(self):
        return self._arrays
    
    def _check(self):
        if not isinstance(self._arrays[0], MetaArray):
            raise TypeError("ArrayList cannot contain objects other than MetaArray.")
        return all((type(arr) is self.type) for arr in self._arrays)
    
    def append(self, arr):
        if self.type is None:
            self._arrays = [arr]
            self.type = type(arr)
        elif type(arr) is self.type:
            self._arrays.append(arr)
        else:
            raise TypeError(f"Cannot add {type(arr)} because ArrayList is composed of {self.type}.")
        
    def __repr__(self):
        return f"ArrayList[{self._arrays[0].__class__.__name__}]"
    
    def __len__(self):
        return len(self._arrays)
    
    def __iter__(self):
        return iter(self._arrays)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._arrays[key]
        out = self.__class__()
        out._arrays = self._arrays[key]
        out.type = type(out._arrays[0])
        return out
    
    def __getattr__(self, name: str):
        f = getattr(self._arrays[0], name) # raise AttributeError here if it should be raised
        if not callable(f):
            raise TypeError("Only methods can be called from ArrayList.")
        def _run(*args, **kwargs):
            out = self.__class__()
            with Progress(name):
                out._arrays = list(getattr(a, name)(*args, **kwargs) for a in self._arrays)
            return out
        return _run
    
    @property
    def axes(self):
        axes = self._arrays[0].axes
        if all(arr.axes == axes for arr in self._arrays):
            return axes
        else:
            raise ValueError("No all the axes are same.")
        