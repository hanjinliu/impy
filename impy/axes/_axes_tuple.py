from __future__ import annotations
from typing import NamedTuple, overload, TYPE_CHECKING, TypeVar
from collections import namedtuple
from ._axis import Axis

if TYPE_CHECKING:
    from ._axes import Axes
    _T = TypeVar("_T")
    
    class AxesTuple(NamedTuple[_T, ...]):
        def __getattr__(self, axis: str) -> Axis:
            ...
        
        @overload
        def __getitem__(self, key: int | str | Axis) -> _T:
            ...

        @overload
        def __getitem__(self, key: slice) -> AxesTuple[_T, ...]:
            ...        
        
else:
    AxesTuple = NamedTuple
    
_AxesShapes: dict[str, AxesTuple] = {}  # cache

def get_axes_tuple(axes: Axes):
    try:
        return _AxesShapes[axes]
    except KeyError:
        fields = []
        for i, a in enumerate(axes):
            s = str(a)
            if s.isidentifier():
                fields.append(s)
            else:
                fields.append(f"axis_{i}")
        tup = namedtuple("AxesShape", fields)
        tup.__getitem__ = _getitem
        _AxesShapes[axes] = tup
        return tup

@overload
def _getitem(self: AxesTuple, key: int | str | Axis) -> int:
    ...

@overload
def _getitem(self: AxesTuple, key: slice) -> tuple[int, ...]:
    ...

def _getitem(self: AxesTuple, key, /):
    if isinstance(key, (str, Axis)):
        return self._asdict()[key]
    return tuple.__getitem__(self, key)
