from __future__ import annotations
import numpy as np
from typing import Iterable, Iterator, MutableSequence, TYPE_CHECKING
from roifile import ImagejRoi, ROI_TYPE, ROI_SUBTYPE, roiread, roiwrite

from .axes import Axes, AxesLike, ImageAxesError

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import ArrayLike
    from matplotlib import axes
    from .arrays.bases import MetaArray

class Roi:
    def __init__(
        self,
        data: ArrayLike,
        axes: AxesLike,
        multi_dims: ArrayLike | None = None,
    ):
        self._data = np.asarray(data)
        if multi_dims is None:
            self._multi_dims = np.empty(0)
        else:
            self._multi_dims = np.atleast_1d(multi_dims)
        self.axes = axes
    
    @property
    def ndim(self) -> int:
        return len(self._multi_dims) + self._data.shape[1]

    @property
    def axes(self) -> Axes:
        return self._axes
    
    @axes.setter
    def axes(self, value):
        _axes = Axes(value)
        if len(_axes) != self.ndim:
            raise ImageAxesError("Dimension mismatch.")
        self._axes = _axes
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}(axes={self.axes!r})"
    
    def __array__(self, dtype=None) -> np.ndarray:
        arr = np.stack([self._multi_dims] * self._data.shape[0], axis=0)
        return np.concatenate([arr, self._data], axis=1)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Roi):
            return False
        return (
            np.all(self._data == other._data) 
            and self.axes == other.axes
            and np.all(self._multi_dims == other._multi_dims)
        )
    
    @staticmethod
    def get_coordinates(coords: np.ndarray):
        """Convert coordinates."""
        return coords[:, ::-1]
        
    @classmethod
    def from_imagejroi(cls, roi: ImagejRoi) -> Self:
        yx: np.ndarray = (cls.get_coordinates(roi.coordinates()) - 1)
        p = roi.position
        c = roi.c_position
        t = roi.t_position
        z = roi.z_position
        d = [x - 1 for x in [p, t, z, c] if x > 0]
        axes = [a for a, x in zip("ptzc", [p, t, z, c]) if x > 0] + ["y", "x"]
        return cls(data=yx, axes=axes, multi_dims=np.array(d))
    
    def _slice_by(self, key) -> Self | None:
        if isinstance(key, np.ndarray):
            return None
        
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        
        nmdim = len(self._multi_dims)
        multi_dims: list[int] = []
        data_list: list[np.ndarray] = []
        axes = []
        for i, k in enumerate(key):
            if isinstance(k, slice):
                s0, s1, step = k.start, k.stop, k.step
                if i < nmdim:
                    crds = self._multi_dims[i]
                    if step > 0:
                        multi_dims.append((crds - s0) / step + s0)
                    else:
                        multi_dims.append((crds - s1) / step)
                else:
                    crds = self._data[:, i - nmdim]
                    if step > 0:
                        data_list.append((crds - s0) / step + s0)
                    else:
                        data_list.append((crds - s1) / step)
                    
                axes.append(self.axes[i])
        
        if len(data_list) < 2:
            return None
        data = np.stack(data_list, axis=1)
        return self.__class__(data, axes=axes, multi_dims=multi_dims)
    
    def copy(self) -> Self:
        return self.__class__(
            data=self._data, axes=self.axes, multi_dims=self._multi_dims
        )
    
    def _dimension_matches(self, arr: MetaArray) -> bool:
        return all(a in self.axes for a in arr.axes)
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
            ax.set_aspect("equal")
        return self._plot(ax, **kwargs)

    def _plot(self, ax, **kwargs):
        raise NotImplementedError


class PolygonRoi(Roi):
    @staticmethod
    def get_coordinates(coords: np.ndarray):
        """Convert coordinates."""
        return coords[:-1, ::-1]
    
    def _plot(self, ax: axes.Axes, **kwargs):
        coords = np.concatenate([self._data, self._data[:1]], axis=0)
        ax.plot(coords[:, 1], coords[:, 0], **kwargs)
        return ax

class RectangleRoi(PolygonRoi):
    pass

class PolyLineRoi(Roi):
    def _plot(self, ax: axes.Axes, **kwargs):
        coords = self._data
        ax.plot(coords[:, 1], coords[:, 0], **kwargs)
        return ax

class LineRoi(PolyLineRoi):
    pass

class PointRoi(Roi):
    def _plot(self, ax: axes.Axes, **kwargs):
        coords = self._data
        ax.scatter(coords[:, 1], coords[:, 0], **kwargs)
        return ax

# class EllipseRoi(Roi):
#     ...

class RoiList(MutableSequence[Roi]):
    def __init__(self, rois: Iterable[Roi] = ()) -> None:
        self._rois = list(rois)
    
    def __getitem__(self, key):
        return self._rois[key]
    
    def __setitem__(self, key: int, roi: Roi):
        if not isinstance(roi, Roi):
            raise TypeError(
                f"Cannot set {type(roi)} to a {type(RoiList).__name__} object."
            )
        self._rois[key] = roi
    
    def __delitem__(self, key) -> None:
        del self._rois[key]
    
    def __len__(self) -> int:
        return len(self._rois)
    
    def __iter__(self) -> Iterator[Roi]:
        return iter(self._rois)
    
    def __repr__(self) -> str:
        s = ",\n\t".join(repr(roi) for roi in self)
        return f"{type(self).__name__}(\n\t{s}\n)"

    def insert(self, index: int, roi: Roi):
        if not isinstance(roi, Roi):
            raise TypeError(f"Cannot set {type(roi)} to a RoiList object.")
        self._rois.insert(index, roi)
    
    def _slice_by(self, key) -> Self:
        data: list[Roi] = []
        for roi in self:
            r = roi._slice_by(key)
            if r is not None:
                data.append(r)
        return self.__class__(data)
    
    @classmethod
    def fromfile(cls, path: str) -> Self:
        rois = roiread(path)
        if not isinstance(rois, list):
            rois = [rois]
        
        data: list[Roi] = []
        for roi in rois:
            roicls = _ROI_TYPE_MAP[roi.roitype, roi.subtype]
            data.append(roicls.from_imagejroi(roi))
        return cls(data)
    
    def tofile(self, path: str) -> None:
        ijrois: list[ImagejRoi] = []
        for roi in self:
            roitype, subtype = _ROI_TYPE_INV_MAP[type(roi)]
            if roi._data.dtype.kind in "ui":
                integer_coordinates = roi._data
                subpixel_coordinates = None
            else:
                integer_coordinates = None
                subpixel_coordinates = roi._data
            ijroi = ImagejRoi(
                roitype=roitype,
                subtype=subtype,
                integer_coordinates=integer_coordinates,
                subpixel_coordinates=subpixel_coordinates,
                multi_coordinates=roi._multi_dims
            )
            ijrois.append(ijroi)
        roiwrite(path, ijrois)
        return None


_ROI_TYPE_MAP: dict[tuple[ROI_TYPE, ROI_SUBTYPE], type[Roi]] = {
    (ROI_TYPE.POINT, ROI_SUBTYPE.UNDEFINED): PointRoi,
    (ROI_TYPE.LINE, ROI_SUBTYPE.UNDEFINED): LineRoi,
    (ROI_TYPE.POLYLINE, ROI_SUBTYPE.UNDEFINED): PolyLineRoi,
    (ROI_TYPE.RECT, ROI_SUBTYPE.UNDEFINED): RectangleRoi,
    (ROI_TYPE.POLYGON, ROI_SUBTYPE.UNDEFINED): PolygonRoi,
}

_ROI_TYPE_INV_MAP: dict[type[Roi], tuple[ROI_TYPE, ROI_SUBTYPE]] = {
    PointRoi: (ROI_TYPE.POINT, ROI_SUBTYPE.UNDEFINED),
    LineRoi: (ROI_TYPE.LINE, ROI_SUBTYPE.UNDEFINED),
    PolyLineRoi: (ROI_TYPE.POLYLINE, ROI_SUBTYPE.UNDEFINED),
    RectangleRoi: (ROI_TYPE.RECT, ROI_SUBTYPE.UNDEFINED),
    PolygonRoi: (ROI_TYPE.POLYGON, ROI_SUBTYPE.UNDEFINED),
}
