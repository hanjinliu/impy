import numpy as np
import pandas as pd
from typing import Union, Sequence, Callable, Iterable, Tuple, Any

__all__ = ["Sequence", "Callable", "Iterable", "nDFloat", "nDInt", "Coords", "Slices", "Dims", "Any"]

nDFloat = Union[Sequence[float], float]
nDInt = Union[Sequence[int], int]
Coords = Union[np.ndarray, pd.DataFrame]
Slices = Tuple[Union[slice,int], ...]
Dims = Union[str, int, None]