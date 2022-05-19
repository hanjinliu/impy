import numpy as np
from typing import Union, Sequence, Callable, Iterable, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Sequence", "Callable", "Iterable", "nDFloat", "nDInt", "Coords", "Slices", "Dims", "Any"]

nDFloat = Union[Sequence[float], float]
nDInt = Union[Sequence[int], int]
Coords = Union[np.ndarray, "pd.DataFrame"]
Slices = Tuple[Union[slice, int], ...]
Dims = Union[str, int, None]
