from __future__ import annotations
from copy import copy
from typing import Any, TypeVar, Union, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

_T = TypeVar("_T")

class Axis:
    """
    An axis object.
    
    This object behaves like a length-1 string as much as possible.
    
    Parameters 
    ----------
    name : str
        Name of axis.
    """
    
    def __init__(self, name: str, metadata: dict[str, Any] | None = None):
        self._name = str(name)
        self._metadata = metadata or {}
    
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
    
    def __copy__(self) -> Self:
        return self.__class__(self._name, self._metadata.copy())
    
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
    
    @property
    def scale(self) -> float:
        return self.metadata.get("scale", 1.0)
    
    @scale.setter
    def scale(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"Cannot set negative scale: {value!r}.")
        self.metadata["scale"] = value
    
    @property
    def unit(self) -> str:
        return self.metadata.get("unit", "px")
    
    @unit.setter
    def unit(self, value: str):
        if value.startswith(r"\u"):
            value = "μ" + value[6:]
        
        self.metadata["unit"] = value
    
    @property
    def labels(self) -> list[_T] | None:
        return self.metadata.get("labels", None)
    
    @labels.setter
    def labels(self, value: Iterable[_T]) -> None:
        self.metadata["labels"] = list(value)
        
    def slice_axis(self, sl: Any) -> Self:
        if not isinstance(sl, (slice, list)):
            return self
        metadata = self.metadata.copy()
        if "scale" in metadata:
            if isinstance(sl, slice):
                step = sl.step or 1
                if step == 1:
                    return self
                new_scale = metadata["scale"] * step
                metadata.update(scale=new_scale)
            else:
                metadata.pop("scale")
        if "labels" in metadata:
            labels = metadata["labels"]
            if isinstance(sl, slice):
                metadata.update(labels=labels[sl])
            else:
                metadata.update(labels=[labels[i] for i in sl])
        return self.__class__(self._name, metadata=metadata)

    

AxisLike = Union[str, Axis]


class UndefAxis(Axis):
    """Undefined axis object."""
    
    def __init__(self, *args, **kwargs):
        super().__init__("#")
    
    def __repr__(self) -> str:
        return "#undef"
    
    def __hash__(self) -> str:
        return id(self)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, str) and other == "#"

def as_axis(obj: Any) -> Axis:
    if isinstance(obj, str):
        if obj == "#":
            axis = UndefAxis()
        else:
            axis = Axis(obj)
    elif isinstance(obj, Axis):
        axis = copy(obj)
    else:
        raise TypeError(f"Cannot use {type(obj)} as an axis.")
    return axis