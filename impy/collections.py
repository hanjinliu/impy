from __future__ import annotations
from typing import (
    Any,
    Callable,
    Iterable,
    TypeVar,
    Hashable,
    MutableSequence,
    MutableMapping,
    overload,
)

__all__ = ["DataList", "DataDict"]

_T = TypeVar("_T")

class CollectionBase:
    _type: type
    
    @property
    def _such_as(self):
        raise NotImplementedError
    
    def _repr_(self, _repr_: str = None) -> str:
        if len(self) == 1:
            return (
            f"{self.__class__.__name__}[{self._type.__name__}] with a component:\n"
            f"{getattr(self._such_as, _repr_, self.__repr__)()}"
            )
        elif len(self) > 1:
            return (
            f"{self.__class__.__name__}[{self._type.__name__}] with {len(self)} "
            f"components such as:\n{getattr(self._such_as, _repr_, self.__repr__)()}"
            )
        else:
            return f"{self.__class__.__name__} with no component"
    
    def __repr__(self) -> str:
        return self._repr_("__repr__")
    
    def _repr_html_(self) -> str:
        return self._repr_("_repr_html_")
    
    def _repr_latex_(self) -> str:
        return self._repr_("_repr_latex_")
        
    
class DataList(CollectionBase, MutableSequence[_T]):
    """
    List-like class that can call same method for every object containded in it. 
    Accordingly, DataList cannot have objects with different types. It is checked
    every time constructor or `append` method is called.
    
    Examples
    --------
    (1) Run Gaussian filter for every ImgArray.
        >>> imgs = DataList([img1, img2, ...])
        >>> out = imgs.gaussian_filter()   # getattr is called for every image here.
    
    (2) Find single molecules for every ImgArray.
        >>> imgs = DataList([img1, img2, ...])
        >>> out = imgs.find_sm()
    """ 
    
    def __init__(self, iterable: Iterable[_T] = ()):
        self._list = list(iterable)
        try:
            self._type = type(self._such_as)
        except IndexError:
            self._type = None
        else:
            if any((type(arr) is not self._type) for arr in self):
                raise TypeError("All the components must be the same type.")
    
    @overload
    def __getitem__(self, key: int) -> _T:
        ...
    
    @overload
    def __getitem__(self, key: slice) -> DataList[_T]:
        ...
    
    @overload
    def __getitem__(self, key: list[int]) -> DataList[_T]:
        ...
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, slice):
            return DataList(self._list[key])
        elif isinstance(key, list):
            return DataList([self._list[i] for i in key])
        raise TypeError("Only int, slice and list can be used for slicing.")
    
    def __setitem__(self, key, value: _T):
        if type(value) is not self._type:
            raise TypeError(
                f"Cannot set {type(value)} to {self.__class__.__name__} with type "
                f"{self._type}."
            )
        self._list[key] = value
    
    def __delitem__(self, key: int | slice):
        del self._list[key]
        
    @property
    def _such_as(self) -> _T:
        return self._list[0]
    
    def insert(self, key: int, component: _T) -> None:
        if self._type is None:
            self._list.insert(key, component)
            self._type = type(component)
        elif type(component) is self._type:
            self._list.insert(key, component)
        else:
            raise TypeError(
                f"Cannot insert {type(component)} because {self.__class__.__name__} is "
                f"composed of {self._type}."
            )
    
    def __add__(self, other: Iterable[_T]):
        l = DataList(other)
        if self._type is None:
            return l  # self is an empty list
        else:
            if l._type is None:
                return DataList(self._list)
            else:
                if self._type is not l._type:
                    raise TypeError(
                        f"Cannot add {self.__class__.__name__} of type {self._type} and "
                        f"that of {l._type}."
                    )
                l._list = self._list + l._list
                return l
    
    def __iadd__(self, values: Iterable[_T]) -> DataList:
        return self + values
    
    def __len__(self) -> int:
        return len(self._list)
    
    def __getattr__(self, name: str) -> Callable[..., DataList[Any]]:
        f = getattr(self._such_as, name)  # raise AttributeError here if it should be raised
        if not callable(f):
            return self.__class__(getattr(a, name) for a in self)
        
        def _run(*args, **kwargs):
            out = self.__class__(getattr(a, name)(*args, **kwargs) for a in self)
            return out
        return _run
    
    def apply(self, func: Callable | str, *args, **kwargs) -> DataList:
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
            return self.__class__(
                getattr(data, func)(*args, **kwargs) for data in self
            )
        else:
            return self.__class__(
                func(data, *args, **kwargs) for data in self
            )
            

_K = TypeVar("_K", bound=Hashable)

class DataDict(CollectionBase, MutableMapping[_K, _T]):
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
    def __init__(self, d: dict[_K, _T] | None = None, **kwargs: dict[_K, _T]):
        if isinstance(d, dict):
            kwargs = d
        
        self._type = None    
        self._dict = kwargs
        
        try:
            self._type = type(self._such_as)
        except StopIteration:
            self._type = None
        else:
            if any((type(arr) is not self._type) for arr in self.values()):
                raise TypeError("All the components must be the same type.")
    
    @property
    def _such_as(self) -> _T:
        return next(iter(self._dict.values()))
    
    def __getitem__(self, key: _K) -> _T:
        return self._dict[key]
        
    def __setitem__(self, key: _K, value: _T) -> None:
        if self._type is None:
            self._dict[key] = value
            self._type = type(value)
        elif type(value) is self._type:
            self._dict[key] = value
        else:
            raise TypeError(
                f"Cannot set {type(value)} because {self.__class__.__name__} is "
                f"composed of {self._type}."
            )
    
    def __delitem__(self, key: _K) -> None:
        del self._dict[key]
    
    def __iter__(self):
        return iter(self._dict)
    
    def __len__(self) -> int:
        return len(self._dict)
    
    def __getattr__(self, name: str) -> Callable[..., DataDict[_K, Any]]:
        if name in self.keys():
            return self[name]
        f = getattr(self._such_as, name) # raise AttributeError here if it should be raised
        if not callable(f):
            return self.__class__({k: getattr(a, name) for k, a in self.items()})
        
        def _run(*args, **kwargs):
            out = self.__class__(
                {k: getattr(a, name)(*args, **kwargs) for k, a in self.items()}
            )
            return out
        return _run
    
    def apply(self, func: Callable | str, *args, **kwargs):
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
        DataDict
            This list is composed of {"name0": func(data[0]), "name1": func(data[1]), ...}
        """        
        if isinstance(func, str):
            return self.__class__(
                {k: getattr(data, func)(*args, **kwargs) for k, data in self.items()}
            )
        else:
            return self.__class__(
                {k: func(data, *args, **kwargs) for k, data in self.items()}
            )

