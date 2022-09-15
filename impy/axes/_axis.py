from __future__ import annotations
import warnings
from copy import copy
from numbers import Real
from typing import Any, Hashable, Sequence, TypeVar, Union, Iterable, TYPE_CHECKING
import numpy as np

from ._misc import ImageAxesWarning

if TYPE_CHECKING:
    from typing_extensions import Self

_T = TypeVar("_T", bound=Hashable)

_LABELS = "labels"
_SCALE = "scale"
_UNIT = "unit"
_COMPONENTS = "components"

class LabelBase(Sequence):
    pass

# TODO: implement this class
class RangeLabels(LabelBase[_T]):
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


class Labels(LabelBase):
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
            return Labels(out)
        return out
    
    def __eq__(self, other: Sequence[_T]) -> bool:
        return self._labels == other
    
    @property
    def has_duplicate(self) -> bool:
        """True if self has duplicated labels."""
        return len(self._labels) != len(self._hash_map)
        
    def get_item(self, key: _T | slice):
        idx = self.get_slice(key)
        return self._labels[idx]
    
    def get_slice(self, key: _T | slice) -> int | slice:
        if self.has_duplicate:
            raise ValueError("Labels have duplicate.")

        if isinstance(key, slice):
            if key.start is None:
                start = None
            else:
                start = self._hash_map[key.start]
            if key.stop is None:
                stop = None
            else:
                stop = self._hash_map[key.stop]
            idx = slice(start, stop, key.step)
        else:
            idx = self._hash_map[key]
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
    
    def __neg__(self) -> Self:
        """Invert axis."""
        return TransformedAxis.from_linear_combination([(-1., self)])
    
    def __add__(self, other: str) -> str:
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
    
    def __radd__(self, other: str) -> str:
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
        return self + (-other)

    def __mul__(self, coef: Real) -> TransformedAxis:
        """Multiply by a scalar."""
        return TransformedAxis.from_linear_combination([(coef, self)])
    
    def __rmul__(self, coef: Real) -> TransformedAxis:
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
        return self.__class__(self._name, self._metadata.copy())
    
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
    def labels(self) -> Labels | None:
        """Axis labels."""
        return self.metadata[_LABELS]
    
    @labels.setter
    def labels(self, value: Iterable[Hashable]) -> None:
        """Set axis labels."""
        self.metadata[_LABELS] = Labels(value)
    
    @labels.deleter
    def labels(self) -> None:
        """Set axis labels."""
        del self.metadata[_LABELS]
    
    def isin(self, values: Iterable[Hashable]) -> np.ndarray:
        """Check if labels are in values."""
        if self.labels is None:
            raise ValueError("Axis has no labels.")
        return np.array([label in values for label in self.labels])

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
        if _LABELS in metadata:
            labels: Labels = metadata[_LABELS]
            if isinstance(sl, slice):
                metadata.update(labels=labels[sl])
            else:
                metadata.update(labels=Labels([labels[i] for i in sl]))
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
        base_scales: list[float] = []
        base_units: set[str] = set()
        for k, axis in components:
            if isinstance(axis, UndefAxis):
                raise TypeError("Cannot use undefined axis in a linear combination.")
            else:
                _increment_axis_component(axis_to_coef, k, axis)
            base_scales.append(axis.scale * abs(k))
            base_units.add(axis.unit)
            
        if name is None:
            name = "".join(f"{k:+.2g}{axis}" for axis, k in axis_to_coef.items()).lstrip("+")
        
        metadata = {_COMPONENTS: axis_to_coef}
        
        if len(base_units) == 1:
            scale = np.linalg.norm(np.array(base_scales))
            unit = base_units.pop()
            metadata.update({_SCALE: scale, _UNIT: unit})
        else:
            warnings.warn("Cannot combine axes with different units.", ImageAxesWarning)
            
        return cls(name, metadata=metadata)

    @property
    def components(self) -> dict[Axis, float]:
        return self.metadata[_COMPONENTS]

def _increment_axis_component(dict_: dict, coef: float, axis: Axis) -> None:
    """Increment a component of the axis."""
    if isinstance(axis, TransformedAxis):
        for _axis, _coef in axis.components.items():
            _increment_axis_component(dict_, _coef, _axis)
    else:
        if axis in dict_:
            dict_[axis] += coef
        else:
            dict_[axis] = coef
    return None
