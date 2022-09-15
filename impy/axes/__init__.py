from ._axes import Axes, AxesLike, ImageAxesError, ScaleView, broadcast
from ._axis import Axis, as_axis, UndefAxis, AxisLike
from ._slicer import Slicer
from ._axes_tuple import AxesTuple

slicer = Slicer()  # default slicer object

__all__ = [
    "Axes",
    "AxesLike",
    "ImageAxesError",
    "ScaleView",
    "broadcast",
    "Axis",
    "as_axis",
    "UndefAxis",
    "AxisLike",
    "AxesTuple",
    "slicer",
]