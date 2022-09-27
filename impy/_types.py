import numpy as np
from typing import Union, Sequence, Callable, Iterable, Tuple, Any, TYPE_CHECKING, Literal
from .axes import Slicer, Axes, AxisLike

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Sequence", "Callable", "Iterable", "nDFloat", "nDInt", "Coords",
           "Slices", "Dims", "Any", "AxesTargetedSlicer"]

nDFloat = Union[Sequence[float], float]
nDInt = Union[Sequence[int], int]
Coords = Union[np.ndarray, "pd.DataFrame"]
Slices = Tuple[Union[slice, int], ...]
if TYPE_CHECKING:
    Dims = list[AxisLike]
else:
    Dims = Union[Sequence[AxisLike], int, None]
AxesTargetedSlicer = Union[str, dict[str, Any], Slicer]
PaddingMode = Union[
    Literal["reflect"], 
    Literal["constant"], 
    Literal["nearest"], 
    Literal["mirror"], 
    Literal["wrap"]
]