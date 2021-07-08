from .utilcls import Progress

class DataList:
    """
    List-like class that can call same method for every object containded in it. Accordingly, DataList
    cannot have objects with different types. It is checked in `__init__()` by calling `_check()`.
    
    Examples
    --------
    (1) Run Gaussian filter for every ImgArray.
    >>> imgs = DataList([img1, img2, ...])
    >>> out = imgs.gaussian_filter()   # getattr is called for every image here.
    
    (2) Find single molecules for every ImgArray.
    >>> imgs = DataList([img1, img2, ...])
    >>> out = imgs.find_sm()
    """    
    def __init__(self, iterable=None):
        if iterable is not None:
            self._arrays = list(iterable)
            self._type = type(self._arrays[0])
            self._check()
        else:
            self._arrays = []
            self._type = None
        
    @property
    def arrays(self):
        return self._arrays
    
    def _check(self):
        return all((type(arr) is self._type) for arr in self._arrays)
    
    def _append(self, arr):
        if self._type is None:
            self._arrays = [arr]
            self._type = type(arr)
        elif type(arr) is self._type:
            self._arrays.append(arr)
        else:
            raise TypeError(f"Cannot add {type(arr)} because ArrayList is composed of {self._type}.")
    
    # several repr function should be defined because in IPython kernel these functions may be called for rich
    # print and by definition __getattr__ will be called every time.
        
    def _repr_(self):
        return f"DataList[{self._type.__name__}]"
    
    __repr__ = _repr_
    _repr_html_ = _repr_
    _repr_latex_ = _repr_
    
    def __len__(self):
        return len(self._arrays)
    
    def __iter__(self):
        return iter(self._arrays)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._arrays[key]
        out = self.__class__()
        out._arrays = self._arrays[key]
        out._type = type(out._arrays[0])
        return out
    
    def __getattr__(self, name: str):
        f = getattr(self._arrays[0], name) # raise AttributeError here if it should be raised
        if not callable(f):
            raise TypeError("Only methods can be called from ArrayList.")
        def _run(*args, **kwargs):
            out = self.__class__()
            with Progress(name):
                out._arrays = list(getattr(a, name)(*args, **kwargs) for a in self._arrays)
                out._type = type(out._arrays[0])
            return out
        return _run
    
        