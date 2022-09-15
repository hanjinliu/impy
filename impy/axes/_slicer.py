from __future__ import annotations
from typing import Any, Iterable
from numbers import Number

class Slicer:
    def __init__(self, sl: dict[str, Any] = None):
        self._dict = sl or {}

    def __getattr__(self, axis: str) -> _AxisSlice:
        return _AxisSlice(self, axis)

    def __call__(self, axis: str) -> _AxisSlice:
        return _AxisSlice(self, axis)
    
    def __len__(self) -> int:
        return len(self._dict)
    
    def __repr__(self) -> str:
        s = []
        for k, v in self._dict.items():
            s.append(f"\n\t{k} ==> {_fmt(v)}")
        s = "".join(s)
        return f"{self.__class__.__name__} of {s}"

    def get_formatter(self, axes: Iterable[str]) -> SliceFormatter:
        """Return a formatter for the given axes."""
        return SliceFormatter(axes, self._dict)

class _AxisSlice:
    def __init__(self, slicer: Slicer, axis: str):
        self.dict = slicer._dict
        self.axis = axis
    
    def __getitem__(self, key) -> Slicer:
        if isinstance(key, tuple):
            # img[slicer.x[1, 3, 5]] == img[slicer.x[[1, 3, 5]]]
            # to avoid to many "[]".
            key = list(key)
        d = self.dict.copy()
        if self.axis in d.keys():
            raise ValueError(f"Axis {self.axis} was sliced twice.")
        d[self.axis] = key
        return Slicer(d)

class SliceFormatter:
    def __init__(self, axes: Iterable[str], defined: dict[str, Any]):
        self._defined = defined
        self.axes = list(axes)
    
    def __getitem__(self, key) -> Slicer | SliceFormatter:
        if not isinstance(key, tuple):
            key = (key,)
        n = len(key)
        d = self._defined.copy()
        for i, k in enumerate(key):
            d.update({self.axes[i]: k})
        if n == len(self.axes):
            return Slicer(d)
        else:
            return SliceFormatter(self.axes[n:], d)
    
    def __repr__(self) -> str:
        s = []
        for k, v in self._defined.items():
            s.append(f"\n\t{k} ==> {_fmt(v)}")
        for a in self.axes:
            s.append(f"\n\t{a} ==> Undefined")
        s = "".join(s)
        return f"{self.__class__.__name__} of {s}"
    
    def zeros(self) -> Slicer:
        return self[(0,) * len(self.axes)]


def _fmt(s):
    if isinstance(s, slice):
        r = ":".join("" if a is None else str(a) for a in [s.start, s.stop, s.step])
        if r.count(":") == 2 and r.endswith(":"):
            r = r[:-1]
    elif isinstance(s, (Number, list)):
        r = repr(s)
    else:
        r = str(s.__class__)
    return r
        