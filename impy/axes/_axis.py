from __future__ import annotations
from collections import defaultdict, OrderedDict
from copy import copy
from typing import Any, MutableSequence, TypeVar, Union, Iterable, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class Axis:
    """
    An axis object.
    
    This object behaves like a length-1 string as much as possible.
    
    Parameters 
    ----------
    name : str
        Name of axis.
    """
    
    def __init__(self, name: str):
        self._name = str(name)
    
    def __str__(self) -> str:
        """String representation of the axis."""
        return self._name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._name!r}]"
    
    def __hash__(self) -> int:
        """Hash as a string."""
        return hash(str(self))
    
    def __eq__(self, other) -> bool:
        """Check equality as a string."""
        return str(self) == other
    
    def __copy__(self) -> Self:
        """Copy object."""
        return self.__class__(self._name)
    
    def __add__(self, other: str) -> str:
        """Add as a string ans returns a string."""
        return self._name + other
    
    def __radd__(self, other: str) -> str:
        """Add as a string ans returns a string."""
        return other + self._name
    
    def __lt__(self, other) -> bool:
        """To support alphabetic ordering."""
        return str(self) < str(other)
    
    def __len__(self) -> int:
        """Length of the string representation."""
        return len(str(self))
    
    def slice_axis(self, sl: int | slice) -> Self:
        """This method is called every time an array is sliced."""
        return self

AxisLike = Union[str, Axis]
    
class AnnotatedAxis(Axis):
    def __init__(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(name)
        self._metadata = metadata or {}
    
    def __copy__(self) -> Self:
        return self.__class__(self._name, self._metadata)
    
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

class ScaledAxis(AnnotatedAxis):
    def __init__(self, name: str, scale: float = 1.0, unit: str = "px"):
        super().__init__(name, metadata={"scale": scale, "unit": unit})
        
    @property
    def scale(self) -> float:
        return self.metadata["scale"]
    
    @scale.setter
    def scale(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"Cannot set negative scale: {value!r}.")
        self.metadata["scale"] = value
    
    @property
    def unit(self):
        return self.metadata["unit"]
    
    @unit.setter
    def unit(self, value: str):
        if value.startswith(r"\u"):
            value = "μ" + value[6:]
        
        self.metadata["unit"] = value
    
    def slice_axis(self, sl: int | slice) -> Self:
        if not isinstance(sl, slice):
            return self
        step = sl.step or 1
        if step == 1:
            return self
        new_scale = self.scale * step
        return self.__class__(self._name, scale=new_scale, unit=self.unit)

_T = TypeVar("_T")

class LabeledAxis(AnnotatedAxis):
    def __init__(self, name: str, label: _T = None):
        super().__init__(name, metadata={"label": label})
        
    @property
    def labels(self) -> _T:
        return self.metadata["labels"]
    
    @labels.setter
    def labels(self, value: Iterable[_T]) -> None:
        self.metadata["labels"] = list(value)
    
    def slice_axis(self, key: int | slice) -> Self:
        metadata = self.metadata.copy()
        metadata.update(labels=self.labels[key])
        return self.__class__(self._name, metadata, copy_metadata=False)

class UndefAxis(Axis):
    """Undefined axis object."""
    def __init__(self, name: None = None):
        super().__init__("#")
    
    def __repr__(self) -> str:
        return "#undef"
    
    def __hash__(self) -> str:
        return id(self)
    
    def __eq__(self, other) -> bool:
        return False


_DEFAULT_AXIS = {
    "#": UndefAxis,
    "c": LabeledAxis,
    "t": ScaledAxis,
    "z": ScaledAxis,
    "y": ScaledAxis,
    "x": ScaledAxis,
}

def as_axis(obj: Any) -> Axis:
    if isinstance(obj, str):
        axis = _DEFAULT_AXIS.get(obj, Axis)(obj)
    elif isinstance(obj, Axis):
        axis = copy(obj)
    else:
        raise TypeError(f"Cannot use {type(obj)} as an axis.")
    return axis
