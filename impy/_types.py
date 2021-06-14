import numpy as np
from typing import Union, Sequence, Callable
from .specials import MarkerFrame

nDFloat = Union[Sequence[float], float]
nDInt = Union[Sequence[int], int]
Coords = Union[np.ndarray, MarkerFrame]