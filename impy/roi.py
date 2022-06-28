from __future__ import annotations
import numpy as np
from typing import Iterable, Iterator, MutableSequence, TYPE_CHECKING
from roifile import ImagejRoi, ROI_TYPE, ROI_SUBTYPE, roiread, roiwrite

from .axes import Axis, Axes, AxisLike, AxesLike, ImageAxesError

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import ArrayLike
    from matplotlib import axes as plt_axes
    from .arrays.bases import MetaArray

POS = Axis("position")

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
    def n_spatial_dims(self) -> int:
        return self._data.shape[1]

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
        nsdim = self.n_spatial_dims
        s_multi = ", ".join(
            f"{a}={d}" for d, a in
            zip(self._multi_dims, self.axes[:-nsdim])
        )
        sp = "yx" if nsdim == 2 else "zyx"
        s_sp = repr(self._data.tolist())
        if len(s_sp) > 28:
            s_sp = s_sp[:26] + " ..."
        if s_multi:
            return f"{type(self).__name__}({s_multi}, {sp}={s_sp})"
        else:
            return f"{type(self).__name__}({sp}={s_sp})"
    
    def __array__(self, dtype=None) -> np.ndarray:
        arr = np.stack([self._multi_dims] * self._data.shape[0], axis=0)
        out: np.ndarray = np.concatenate([arr, self._data], axis=1)
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
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
        axes = [a for a, x in zip([POS, "t", "z", "c"], [p, t, z, c]) if x > 0] + ["y", "x"]
        return cls(data=yx, axes=axes, multi_dims=np.array(d))
    
    def _slice_by(self, key) -> Self | None:
        if isinstance(key, np.ndarray):
            return None
        
        if not isinstance(key, tuple):
            key = (key,)
        
        nmdim = len(self._multi_dims)
        multi_dims: list[int] = []
        data_list: list[np.ndarray] = []
        axes = []
        i = -1
        for i, k in enumerate(key):
            if isinstance(k, slice):
                s0, s1, step = k.start, k.stop, k.step
                if i < nmdim:
                    crds = self._multi_dims[i]
                    if step > 0:
                        multi_dims.append((crds - s0) / step)
                    else:
                        multi_dims.append((crds - s1) / step)
                else:
                    crds = self._data[:, i - nmdim]
                    if step > 0:
                        data_list.append((crds - s0) / step)
                    else:
                        data_list.append((crds - s1) / step)
                    
                axes.append(self.axes[i])
            
            elif isinstance(k, list):
                if i >= nmdim or self._multi_dims[i] not in k:
                    return None
                else:
                    multi_dims.append(k.index(self._multi_dims[i]))
                axes.append(self.axes[i])

            elif isinstance(k, int):
                if i >= nmdim or self._multi_dims[i] != k:
                    return None
        
        for j in range(i + 1, self.ndim):
            if j < nmdim:
                multi_dims.append(self._multi_dims[j])
            else:
                data_list.append(self._data[:, j - nmdim])
            axes.append(self.axes[j])
        
        if len(data_list) < 2:
            return None
        data = np.stack(data_list, axis=1)
        return self.__class__(data=data, axes=axes, multi_dims=multi_dims)
    
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
    
    def drop(self, axis: int | AxisLike):
        if not isinstance(axis, int):
            axis = self.axes.find(axis)
        multi_dims = [a for i, a in enumerate(self._multi_dims) if i != axis]
        return self.__class__(
            data=self._data, axes=self.axes.drop(axis), multi_dims=multi_dims,
        )


class PolygonRoi(Roi):
    @staticmethod
    def get_coordinates(coords: np.ndarray):
        """Convert coordinates."""
        return coords[:-1, ::-1]
    
    def _plot(self, ax: plt_axes.Axes, **kwargs):
        coords = np.concatenate([self._data, self._data[:1]], axis=0)
        ax.plot(coords[:, 1], coords[:, 0], **kwargs)
        return ax

class RectangleRoi(PolygonRoi):
    pass

class PolyLineRoi(Roi):
    def _plot(self, ax: plt_axes.Axes, **kwargs):
        coords = self._data
        ax.plot(coords[:, 1], coords[:, 0], **kwargs)
        return ax

class LineRoi(PolyLineRoi):
    pass

class PointRoi(Roi):
    def _plot(self, ax: plt_axes.Axes, **kwargs):
        coords = self._data
        ax.scatter(coords[:, 1], coords[:, 0], **kwargs)
        return ax

# class EllipseRoi(Roi):
#     ...

class RoiList(MutableSequence[Roi]):
    """A list of ROIs."""
    
    def __init__(self, axes: AxesLike, rois: Iterable[Roi] = ()) -> None:
        self._axes = Axes(axes)
        self._rois: list[Roi] = []
        for roi in rois:
            if not self._axes.contains(roi.axes):
                raise ImageAxesError(
                    f"Cannot add ROI with axes {roi.axes} to {type(self).__name__} with "
                    f"axes {self.axes}."
                )
            self._rois.append(roi)
    
    @property
    def axes(self) -> Axes:
        return self._axes
    
    @axes.setter
    def axes(self, val: AxesLike):
        _axes = Axes(val)
        if len(_axes) != len(self._axes):
            raise ImageAxesError(f"Cannot change the length of axes")
        _old_to_new_map = {k: v for k, v in zip(self.axes, val)}
        for roi in self:
            roi.axes = [_old_to_new_map[a] for a in roi.axes]
        self._axes = Axes(val)
    
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
        """
        Insert a ROI at the given index.
        
        The axes of the ROI must match the axes of the ROI list.

        Parameters
        ----------
        index : int
            Index at which to insert the ROI.
        roi : Roi
            ROI to insert.
        """
        if not isinstance(roi, Roi):
            raise TypeError(f"Cannot set {type(roi)} to a RoiList object.")
        elif not self.axes.contains(roi.axes):
            raise ImageAxesError(
                f"Cannot add ROI with axes {roi.axes} to {type(self).__name__} with "
                f"axes {self.axes}."
            )
        self._rois.insert(index, roi)
    
    def _slice_by(self, key) -> Self:
        data: list[Roi] = []
        if not isinstance(key, tuple):
            key = (key,)
        
        for roi in self:
            _key = tuple(
                key[i] for i, a in enumerate(self.axes) 
                if (a in roi.axes and i < len(key))
            )
            r = roi._slice_by(_key)
            if r is not None:
                data.append(r)
        if len(data) == 0:
            return None
        axes = [self.axes[i] for i, sl in enumerate(key) if not isinstance(sl, int)] + self.axes[len(key):]
        return self.__class__(axes, data)

    def _dimension_matches(self, arr: MetaArray) -> bool:
        """Check if dimension matches."""
        return all(a in self.axes for a in arr.axes)
    
    @classmethod
    def fromfile(cls, path: str) -> Self:
        """
        Construct a RoiList from a file.

        Parameters
        ----------
        path : str
            Path to the ROI file.

        Returns
        -------
        RoiList
            A RoiList object with the ROIs read from the file.
        """
        from .axes import broadcast
        rois = roiread(path)
        if not isinstance(rois, list):
            rois = [rois]
        
        data: list[Roi] = []
        all_axes: list[Axes] = []
        for ijroi in rois:
            roicls = _ROI_TYPE_MAP[ijroi.roitype, ijroi.subtype]
            roi = roicls.from_imagejroi(ijroi)
            data.append(roi)
            all_axes.append(roi.axes)
        
        return cls(broadcast(*all_axes), data)
    
    def tofile(self, path: str) -> None:
        """
        Save the RoiList to a file.

        Parameters
        ----------
        path : str
            Path to the file.
        """
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

    def plot(self, ax=None, **kwargs):
        """Plot all the ROIs."""
        for roi in self:
            roi.plot(ax, **kwargs)
        return None

    def drop(self, axis: int | AxisLike):
        """Drop an axis from all the ROIs."""
        return self.__class__(self.axes.drop(axis), [roi.drop(axis) for roi in self])

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
