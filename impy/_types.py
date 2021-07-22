import numpy as np
import pandas as pd
from typing import Union, Sequence, Callable, Iterable, Tuple

__all__ = ["Sequence", "Callable", "Iterable", "nDFloat", "nDInt", "Coords", "Slices"]

nDFloat = Union[Sequence[float], float]
nDInt = Union[Sequence[int], int]
Coords = Union[np.ndarray, pd.DataFrame]
Slices = Tuple[Union[slice,int],...]
