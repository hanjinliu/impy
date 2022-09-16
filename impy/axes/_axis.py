from __future__ import annotations
from abc import ABC, abstractmethod
import warnings
from copy import copy
from typing import Any, Hashable, Sequence, SupportsIndex, TypeVar, Union, Iterable, TYPE_CHECKING
import numpy as np

from ._misc import ImageAxesWarning

if TYPE_CHECKING:
    from typing_extensions import Self

_T = TypeVar("_T", bound=Hashable)
_Slicable = Union[SupportsIndex, slice, list[int], np.ndarray]

_COORDINATES = "labels"
_SCALE = "scale"
_UNIT = "unit"
_COMPONENTS = "components"

class CoordinatesBase(Sequence[_T], ABC):
    @abstractmethod
    def to_indexer(self, coords: _T | slice) -> _Slicable:
        """Convert a coordinate into a slicable object."""


# TODO: implement this class
class RangeCoordinates(CoordinatesBase[_T]):
    def __init__(self, start: int, stop: int, step: int):
        self._start = start
        self._stop = stop
        self._step = step
    
    def __getitem__(self, key):
        val = self._start + key * self._step
        if val > self._stop:
            raise IndexError("Index out of range.")
        return val
    
    def __len__(self) -> int:
        return (self._stop - self._start) // self._step
    
    def to_indexer(self, coords: _T | slice) -> _Slicable:
        if isinstance(coords, slice):
            if coords.step not in (None, 1, -1):
                raise ValueError("Step size must be 1 or -1.")
            start = coords.start
            stop = coords.stop
            if start is not None:
                start = int(np.ceil((start - self._start) / self._step))
            if stop is not None:
                stop = int((stop - self._start) / self._step)
            return slice(start, stop, coords.step)
        else:
            return int(np.round((coords - self._start) / self._step))

class Coordinates(CoordinatesBase[_T]):
    """A tuple-like object storing label specifiers."""
    
    def __init__(self, seq: Iterable[_T]):
        self._labels = tuple(seq)
        self._hash_map: dict[_T, int] = {}
        for i, label in enumerate(self._labels):
            self._hash_map.setdefault(label, i)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}{self._labels!r}"
    
    def __len__(self) -> int:
        """Length of labels"""
        return len(self._labels)
    
    def __getitem__(self, key):
        out = self._labels[key]
        if isinstance(key, slice):
            return Coordinates(out)
        return out
    
    def __eq__(self, other: Sequence[_T]) -> bool:
        return self._labels == other
    
    @property
    def has_duplicate(self) -> bool:
        """True if self has duplicated labels."""
        return len(self._labels) != len(self._hash_map)
        
    def get_item(self, key: _T | slice):
        idx = self.to_indexer(key)
        return self._labels[idx]
    
    def to_indexer(self, coords: _T | slice) -> int | slice:
        if self.has_duplicate:
            raise ValueError("Labels have duplicate.")

        if isinstance(coords, slice):
            if coords.start is None:
                start = None
            else:
                start = self._hash_map[coords.start]
            if coords.stop is None:
                stop = None
            else:
                stop = self._hash_map[coords.stop]
            idx = slice(start, stop, coords.step)
        else:
            idx = self._hash_map[coords]
        return idx

    def index(self, val: _T) -> int:
        try:
            out = self._hash_map[val]
        except KeyError:
            raise ValueError(f"{val!r} not in the labels.")
        return out

class Axis:
    """
    An axis object.
    
    This object behaves like a length-1 string as much as possible.
    
    Parameters 
    ----------
    name : str
        Name of axis.
    """
    
    def __init__(
        self,
        name: str,
        *,
        scale: float | None = None,
        unit: str | None = None,
        coords: Sequence[Hashable] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._name = str(name)
        self._metadata = metadata or {}
        if scale is not None:
            self.scale = scale
        if unit is not None:
            self.unit = unit
        if coords is not None:
            self.coords = coords
    
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
    
    def __neg__(self) -> Self:
        """Invert axis."""
        return TransformedAxis.from_linear_combination([(-1., self)])
    
    def __add__(self, other: Axis) -> TransformedAxis:
        """Add as a string ans returns a string."""
        if isinstance(other , str):
            warnings.warn(
                "Adding a string to an axis is deprecated. "
                "This will raise an error in the future.", 
                DeprecationWarning,
            )
            return self._name + other
        elif isinstance(other, Axis):
            return TransformedAxis.from_linear_combination([(1., self), (1., other)])
        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}.")
    
    def __radd__(self, other: Axis) -> TransformedAxis:
        """Add as a string ans returns a string."""
        if isinstance(other , str):
            warnings.warn(
                "Adding a string to an axis is deprecated. "
                "This will raise an error in the future.", 
                DeprecationWarning,
            )
            return other + self._name
        elif isinstance(other, Axis):
            return TransformedAxis.from_linear_combination([(1., other), (1., self)])
        else:
            raise TypeError(f"Cannot add {type(other)} and {type(self)}.")

    def __sub__(self, other: Axis) -> TransformedAxis:
        """Subtract another axis."""
        return TransformedAxis.from_linear_combination([(1., self), (-1., other)])

    def __mul__(self, coef: int | float) -> TransformedAxis:
        """Multiply by a scalar."""
        return TransformedAxis.from_linear_combination([(coef, self)])
    
    def __rmul__(self, coef: int | float) -> TransformedAxis:
        """Multiply by a scalar."""
        return TransformedAxis.from_linear_combination([(coef, self)])        
            
    def __lt__(self, other) -> bool:
        """To support alphabetic ordering."""
        return str(self) < str(other)
    
    def __len__(self) -> int:
        """Length of the string representation."""
        return len(str(self))
    
    def __iter__(self) -> Iterable[str]:
        """Iterate as a string."""
        return iter(str(self))
    
    def __copy__(self) -> Self:
        return self.__class__(self._name, metadata=self._metadata.copy())
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dictionary."""
        return self._metadata
    
    @property
    def scale(self) -> float:
        """Physical scale of axis."""
        return self.metadata.get(_SCALE, 1.0)
    
    @scale.setter
    def scale(self, value: float) -> None:
        """Set physical scale to the axis."""
        value = float(value)
        if value <= 0:
            raise ValueError(f"Cannot set negative scale: {value!r}.")
        self.metadata[_SCALE] = value
    
    @property
    def unit(self) -> str:
        """Physical scale unit of axis."""
        return self.metadata.get(_UNIT, "px")
    
    @unit.setter
    def unit(self, value: str | None):
        """Set physical unit to the axis."""
        if value is None:
            value = "px"
        elif value.startswith("\\u00B5") or value.startswith("\\u03BC"):
            value = "μ" + value[6:]
        
        self.metadata[_UNIT] = value
    
    @property
    def labels(self) -> Coordinates | None:
        """Axis labels."""
        return self.metadata[_COORDINATES]
    
    @labels.setter
    def labels(self, value: Iterable[Hashable]) -> None:
        """Set axis labels."""
        self.metadata[_COORDINATES] = Coordinates(value)
    
    @labels.deleter
    def labels(self) -> None:
        """Set axis labels."""
        del self.metadata[_COORDINATES]
    
    @property
    def coords(self) -> Coordinates | None:
        """Axis coordinates."""
        return self.metadata[_COORDINATES]
    
    @coords.setter
    def coords(self, value: Iterable[Hashable]) -> None:
        """Set axis coordinates."""
        self.metadata[_COORDINATES] = Coordinates(value)
    
    @coords.deleter
    def coords(self) -> None:
        """Set axis coordinates."""
        del self.metadata[_COORDINATES]

    def isin(self, values: Iterable[Hashable]) -> np.ndarray:
        """Check if labels are in values."""
        if self.coords is None:
            raise ValueError("Axis has no coordinates.")
        return np.array([label in values for label in self.coords])

    def slice_axis(self, sl: Any) -> Self:
        """Return sliced axis."""
        if not isinstance(sl, (slice, list)):
            return self
        metadata = self.metadata.copy()
        if _SCALE in metadata:
            if isinstance(sl, slice):
                step = sl.step or 1
                if step == 1:
                    return self
                new_scale = metadata[_SCALE] * abs(step)
                metadata.update(scale=new_scale)
            else:
                metadata.pop(_SCALE)
        if _COORDINATES in metadata:
            labels: Coordinates = metadata[_COORDINATES]
            if isinstance(sl, slice):
                metadata.update(labels=labels[sl])
            else:
                metadata.update(labels=Coordinates([labels[i] for i in sl]))
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
    """Convert an object into an ``Axis`` object."""
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


class TransformedAxis(Axis):
    """
    Axis that is a linear combination of other axes.
    
    Examples
    --------
    >>> x = Axis("x")
    >>> y = Axis("y")
    >>> u = 0.3 * x + 0.4 * y
    >>> u
    TransformedAxis['0.3x+0.4y']
    >>> u - x
    TransformedAxis['-0.7x+0.4y']
    """

    def __init__(self, name: str, metadata=None):
        super().__init__(name, metadata=metadata)
    
    def __hash__(self) -> int:
        """Hash as a string."""
        return hash(tuple(self.components.items()))
    
    @classmethod
    def from_linear_combination(
        cls: type[TransformedAxis],
        components: Iterable[tuple[float, Axis]],
        name: str | None = None,
    ) -> Self:
        """
        Construct a TransformedAxis from a linear combination of other axes.

        Parameters
        ----------
        components : Iterable of (float, Axis)
            Coefficient and axis.
        name : str, optional
            Name of the axis. By default, a simplified expression of the linear combination
            will be used.

        Returns
        -------
        TransformedAxis
            Axis with the given linear combination of other axes.        
        """
        axis_to_coef: dict[Axis, float] = {}
        base_units: set[str] = set()
        for k, axis in components:
            if isinstance(axis, UndefAxis):
                raise TypeError("Cannot use undefined axis in a linear combination.")
            else:
                _increment_axis_component(axis_to_coef, k, axis)
            base_units.add(axis.unit)
            
        if name is None:
            name = "".join(f"{k:+.2g}{axis}" for axis, k in axis_to_coef.items()).lstrip("+")
        
        metadata = {_COMPONENTS: axis_to_coef}
        
        if len(base_units) == 1:
            scale = np.linalg.norm(
                [axis.scale * abs(coef) for axis, coef in axis_to_coef.items()]
            )
            unit = base_units.pop()
            metadata.update({_SCALE: scale, _UNIT: unit})
        else:
            warnings.warn("Cannot combine axes with different units.", ImageAxesWarning)
            
        return cls(name, metadata=metadata)

    @property
    def components(self) -> dict[Axis, float]:
        return self.metadata[_COMPONENTS]
    
    @property
    def vector(self) -> np.ndarray:
        """Vector representation of the axis."""
        return np.array(list(self.components.values()), dtype=np.float32)
    
    @property
    def bases(self) -> list[Axis]:
        """Base axes."""
        return list(self.components.keys())
    
    def transform(self, matrix: np.ndarray) -> Self:
        """Transform axis by a matrix."""
        new = self.vector.dot(matrix)
        return self.from_linear_combination(zip(new, self.components.keys()))
    
    def __matmul__(self, matrix: np.ndarray) -> Self:
        """Transform axis by a matrix."""
        return self.transform(matrix)


def _increment_axis_component(dict_: dict, coef: float, axis: Axis) -> None:
    """Increment a component of the axis."""
    if isinstance(axis, TransformedAxis):
        for _axis, _coef in axis.components.items():
            _increment_axis_component(dict_, _coef * coef, _axis)
    else:
        if axis in dict_:
            dict_[axis] += coef
        else:
            dict_[axis] = coef
    return None
