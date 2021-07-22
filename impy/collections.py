from __future__ import annotations
from .utils.utilcls import Progress
from collections import UserList, UserDict
import numpy as np

__all__ = ["DataList", "DataDict"]

class CollectionBase:
    @property
    def _such_as(self):
        raise NotImplementedError
    
    def _repr_(self):
        if len(self) == 1:
            return \
        f"""{self.__class__.__name__}[{self._type.__name__}] with a component:
        {repr(self._such_as)}
        """
        elif len(self) > 1:
            return \
        f"""{self.__class__.__name__}[{self._type.__name__}] with components such as:
        {repr(self._such_as)}
        """
        else:
            return f"{self.__class__.__name__} with no component"
    # several repr function should be defined because in IPython kernel these functions may be called 
    # for rich print and by definition __getattr__ will be called every time.
        
    __repr__ = _repr_
    _repr_html_ = _repr_
    _repr_latex_ = _repr_
    
    
class DataList(CollectionBase, UserList):
    """
    List-like class that can call same method for every object containded in it. Accordingly, DataList
    cannot have objects with different types. It is checked every time constructor or `append` method is
    called.
    
    Examples
    --------
    (1) Run Gaussian filter for every ImgArray.
    >>> imgs = DataList([img1, img2, ...])
    >>> out = imgs.gaussian_filter()   # getattr is called for every image here.
    
    (2) Find single molecules for every ImgArray.
    >>> imgs = DataList([img1, img2, ...])
    >>> out = imgs.find_sm()
    """ 
    def __init__(self, iterable=()):
        super().__init__(iterable)
        try:
            self._type = type(self._such_as)
        except IndexError:
            self._type = None
        else:
            if any((type(arr) is not self._type) for arr in self):
                raise TypeError("All the components must be the same type.")
    
    @property
    def _such_as(self):
        return self[0]
    
    def append(self, component):
        if self._type is None:
            super().append(component)
            self._type = type(component)
        elif type(component) is self._type:
            super().append(component)
        else:
            raise TypeError(f"Cannot add {type(component)} because {self.__class__.__name__} is composed of "
                            f"{self._type}.")
    
    def __getattr__(self, name: str):
        f = getattr(self._such_as, name) # raise AttributeError here if it should be raised
        if not callable(f):
            return self.__class__(getattr(a, name) for a in self)
        
        def _run(*args, **kwargs):
            if not name.startswith("_"):
                with Progress(name):
                    out = self.__class__(getattr(a, name)(*args, **kwargs) for a in self)
            else:
                out = self.__class__(getattr(a, name)(*args, **kwargs) for a in self)
            return out
        return _run
    
    def apply(self, func, *args, **kwargs) -> DataList:
        """
        Apply same function to each components. It can be any callable objects or any method of the components.

        Parameters
        ----------
        func : Callable or str
            Function to be applied to each components.
        args
            Other arguments of `func`.
        kwargs
            Other keyword arguments of `func`.

        Returns
        -------
        DataList
            This list is composed of [func(data[0]), func(data[1]), ...]
        """        
        if isinstance(func, str):
            return self.__class__(getattr(data, func)(*args, **kwargs) 
                                  for data in self)
        else:
            return self.__class__(func(data, *args, **kwargs) 
                                  for data in self)
            
    
class DataDict(CollectionBase, UserDict):
    """
    Dictionary-like class that can call same method for every object containded in the values. Accordingly, 
    DataDict cannot have objects with different types as values. It is checked every time constructor or 
    `__setitem__` method is called.
    
    Examples
    --------
    (1) Run Gaussian filter for every ImgArray.
    >>> imgs = DataDict(first=img1, second=img2)
    >>> out = imgs.gaussian_filter()   # getattr is called for every image here.
    >>> out.first # return the first one.
    
    (2) Find single molecules for every ImgArray.
    >>> imgs = DataDict([img1, img2, ...])
    >>> out = imgs.find_sm()
    """ 
    def __init__(self, d=None, **kwargs):
        if isinstance(d, dict):
            kwargs = d
        
        self._type = None    
        super().__init__(**kwargs)
        
        try:
            self._type = type(self._such_as)
        except StopIteration:
            self._type = None
        else:
            if any((type(arr) is not self._type) for arr in self.values()):
                raise TypeError("All the components must be the same type.")
    
    @property
    def _such_as(self):
        return next(iter(self.data.values()))
    
    
    def __setitem__(self, name:str, component):
        if self._type is None:
            super().__setitem__(name, component)
            self._type = type(component)
        elif type(component) is self._type:
            super().__setitem__(name, component)
        else:
            raise TypeError(f"Cannot set {type(component)} because {self.__class__.__name__} is composed of "
                            f"{self._type}.")

    def __getattr__(self, name: str):
        if name in self.keys():
            return self[name]
        f = getattr(self._such_as, name) # raise AttributeError here if it should be raised
        if not callable(f):
            return self.__class__({k: getattr(a, name) for k, a in self.items()})
        
        def _run(*args, **kwargs):
            if not name.startswith("_"):
                with Progress(name):
                    out = self.__class__({k: getattr(a, name)(*args, **kwargs) 
                                          for k, a in self.items()})
            else:
                out = self.__class__({k: getattr(a, name)(*args, **kwargs) 
                                      for k, a in self.items()})
            return out
        return _run
    
    def apply(self, func, *args, **kwargs):
        """
        Apply same function to each components. It can be any callable objects or any method of the components.

        Parameters
        ----------
        func : Callable or str
            Function to be applied to each components.
        args
            Other arguments of `func`.
        kwargs
            Other keyword arguments of `func`.

        Returns
        -------
        DataList
            This list is composed of {"name0": func(data[0]), "name1": func(data[1]), ...}
        """        
        if isinstance(func, str):
            return self.__class__({k: getattr(data, func)(*args, **kwargs) 
                                   for k, data in self.items()})
        else:
            return self.__class__({k: func(data, *args, **kwargs) 
                                   for k, data in self.items()})