from __future__ import annotations
from typing import Any


class Slicer:
    def __init__(self, sl: dict[str, Any] = None):
        self._dict = sl or {}

    def __getattr__(self, axis: str) -> _AxisSlice:
        return _AxisSlice(self, axis)

    def __call__(self, axis: str) -> _AxisSlice:
        return _AxisSlice(self, axis)
    
    def __len__(self) -> int:
        return len(self._dict)


class _AxisSlice:
    def __init__(self, slicer: Slicer, axis: str):
        self.slicer = slicer
        self.axis = axis
    
    def __getitem__(self, key) -> Slicer:
        if isinstance(key, tuple):
            # img[slicer.x[1, 3, 5]] == img[slicer.x[[1, 3, 5]]]
            # to avoid to many "[]".
            key = list(key)
        d = self.slicer._dict.copy()
        if self.axis in d.keys():
            raise ValueError(f"Axis {self.axis} was sliced twice.")
        d[self.axis] = key
        return Slicer(d)
