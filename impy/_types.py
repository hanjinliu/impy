import numpy as np
from typing import Union, Sequence, Callable, Iterable, Tuple
from .frame import MarkerFrame

__all__ = ["Sequence", "Callable", "Iterable", "nDFloat", "nDInt", "Coords", "Slices"]

nDFloat = Union[Sequence[float], float]
nDInt = Union[Sequence[int], int]
Coords = Union[np.ndarray, MarkerFrame]
Slices = Tuple[Union[slice,int],...]
