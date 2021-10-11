from __future__ import annotations
from .utils.utilcls import Progress
from collections import UserList, UserDict
from typing import Any, Callable, TypeVar

__all__ = ["DataList", "DataDict"]

_T = TypeVar("_T")

class CollectionBase:
    _type: _T
    
    @property
    def _such_as(self) -> _T:
        raise NotImplementedError
    
    def _repr_(self, _repr_: str = None) -> str:
        if len(self) == 1:
            return \
        f"""{self.__class__.__name__}[{self._type.__name__}] with a component:
        {getattr(self._such_as, _repr_, self.__repr__)()}
        """
        elif len(self) > 1:
            return \
        f"""{self.__class__.__name__}[{self._type.__name__}] with {len(self)} components such as:
        {getattr(self._such_as, _repr_, self.__repr__)()}
        """
        else:
            return f"{self.__class__.__name__} with no component"
    
    def __repr__(self) -> str:
        return self._repr_("__repr__")
    
    def _repr_html_(self) -> str:
        return self._repr_("_repr_html_")
    
    def _repr_latex_(self) -> str:
        return self._repr_("_repr_latex_")
        
    
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
    def _such_as(self) -> _T:
        return self[0]
    
    def append(self, component: _T) -> None:
        if self._type is None:
            super().append(component)
            self._type = type(component)
        elif type(component) is self._type:
            super().append(component)
        else:
            raise TypeError(f"Cannot append {type(component)} because {self.__class__.__name__} is composed of "
                            f"{self._type}.")
    
    def __add__(self, other: DataList):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {type(other)}.")
        elif self._type is other._type or self._type is None or other._type is None:
            return super().__add__(other)
        else:
            raise TypeError("Cannot add two lists composed of different type of objects: "
                           f"{self._type} and {other._type}.")
    
    def __iadd__(self, other: DataList) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {type(other)}.")
        elif self._type is other._type or self._type is None or other._type is None:
            self._type = self._type if other._type is None else other._type
            return super().__iadd__(other)
        else:
            raise TypeError("Cannot add two lists composed of different type of objects: "
                           f"{self._type} and {other._type}.")
    
    def extend(self, other: DataList) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot extend DataList with {type(other)}.")
        elif self._type is other._type or self._type is None or other._type is None:
            self._type = self._type if other._type is None else other._type
            return super().extend(other)
        else:
            raise TypeError("Cannot add two lists composed of different type of objects: "
                           f"{self._type} and {other._type}.")
        
    def __getattr__(self, name: str) -> Any:
        f = getattr(self._such_as, name) # raise AttributeError here if it should be raised
        if not callable(f):
            return self.__class__(getattr(a, name) for a in self)
        
        def _run(*args, **kwargs):
            out = None if name.startswith("_") else "stdout"
            with Progress(name):
                out = self.__class__(getattr(a, name)(*args, **kwargs) for a in self)
            return out
        return _run
    
    def apply(self, func: Callable|str, *args, **kwargs) -> DataList:
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
    def _such_as(self) -> _T:
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
            
            out = None if name.startswith("_") else "stdout"
            with Progress(name):
                out = self.__class__({k: getattr(a, name)(*args, **kwargs) 
                                      for k, a in self.items()})
            return out
        return _run
    
    def apply(self, func: Callable|str, *args, **kwargs):
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

