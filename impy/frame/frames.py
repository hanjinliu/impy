from __future__ import annotations
from typing import MutableMapping, TYPE_CHECKING
import numpy as np
import pandas as pd
from functools import wraps

from ..utils.axesop import complement_axes
from ..utils.deco import dims_to_spatial_axes
from ..utils.slicer import str_to_slice
from ..utils.utilcls import ImportOnRequest
from .._const import Const
from ..axes import Axes, ImageAxesError

if TYPE_CHECKING:
    import trackpy as tp
else:
    tp = ImportOnRequest("trackpy")

class AxesFrame(pd.DataFrame):
    _metadata = ["_axes"]
    
    @property
    def _constructor(self):
        return self.__class__
    
    def __init__(self, data=None, columns=None, **kwargs):
        if isinstance(columns, (str, Axes)):
            kwargs["columns"] = [a for a in columns]
        elif isinstance(data, AxesFrame):
            kwargs["columns"] = data.columns.tolist()
            if columns is None:
                columns = str(data._axes)
        else:
            kwargs["columns"] = columns
        
        super().__init__(data, **kwargs)
        self._axes = Axes(self.columns)
    
    def _get_coords_cols(self):
        return [a for a in self.columns if len(a) == 1]
    
    def get_coords(self):
        return self[self.columns[self.columns.isin([a for a in self.columns if len(a) == 1])]]
    
    
    def __getitem__(self, k):
        # sl_tuple = solve_slicer(key, self.axes)
        # drops: list[int] = []
        # for i, sl in enumerate(sl_tuple):
        #     arr = self._data[:, i]
        #     out: bool | np.ndarray = True
        #     if isinstance(sl, slice):
        #         s0, s1, step = sl.indices(len(self))
        #         binary = np.logical_and(s0 <= arr, arr < s1)
        #         if step > 1:
        #             binary = np.logical_and(binary, (arr - s0) % step == 0)
        #         elif step < 0:
        #             raise NotImplementedError
                
        #     else:
        #         binary = arr == sl
        #         drops.append(i)
        #     out *= binary
        # inherits = [i for i in range(len(self)) if i not in drops]
        # df_out = super().__getitem__(out)
        # if isinstance(df_out, AxesFrame):
        #     df_out._axes = Axes(df_out._get_coords_cols())
        #     df_out.set_scale(self)
        # return df_out
        if isinstance(k, str):
            if ";" in k:
                for each in k.split(";"):
                    self = self.__getitem__(each.strip())
                return self
            
            elif "=" in k:
                axis, sl = [a.strip() for a in k.split("=")]
                sl = str_to_slice(sl)
                if isinstance(sl, int):
                    out = self[self[axis]==sl]
                elif isinstance(sl, slice):
                    out = self[(sl.start<=self[axis]) & (self[axis]<sl.stop)]
                elif isinstance(sl, list):
                    out = self[self[axis].isin(sl)]
                else:
                    raise ValueError(f"Wrong key: {k} returned {sl}")
            elif "" == k:
                return self
            else:
                out = super().__getitem__(k)                
        else:
            out = super().__getitem__(k)
            
        if isinstance(out, AxesFrame):
            out._axes = Axes(out._get_coords_cols())
            out.set_scale(self)
        return out
    
    @property
    def col_axes(self):
        return self._axes
    
    @col_axes.setter
    def col_axes(self, value):
        naxes = self.shape[1]
        if value is None:
            self._axes = Axes.undef(naxes)
        else:
            axes = Axes(value)
            if naxes != len(axes):
                raise ImageAxesError(
                    "Inconpatible dimensions: "
                    f"array (ndim={naxes}) and axes ({value})"
                )
            self._axes = axes
    
    @property
    def scale(self):
        return self._axes.scale
    
    def set_scale(self, other=None, **kwargs) -> None:
        """
        Set scales of each axis.

        Parameters
        ----------
        other : dict, AxesFrame or MetaArray, optional
            New scales. If dict, it should be like {"x": 0.1, "y": 0.1}. If MetaArray, only
            scales of common axes are copied.
        kwargs : 
            This enables function call like set_scale(x=0.1, y=0.1).

        """
        if isinstance(other, MutableMapping):
            # yx-scale can be set with one keyword.
            if "yx" in other:
                yxscale = other.pop("yx")
                other["x"] = other["y"] = yxscale
            if "xy" in other:
                yxscale = other.pop("xy")
                other["x"] = other["y"] = yxscale
            # check if all the keys are contained in axes.
            for a, val in other.items():
                if a not in self._axes:
                    raise ImageAxesError(f"Image does not have axis {a}.")    
                elif not np.isscalar(val):
                    raise TypeError(f"Cannot set non-numeric value as scales.")
            
            for k, v in other.items():
                self.col_axes[k].scale = v
            
        elif hasattr(other, "scale"):
            self.set_scale({a: s for a, s in other.scale.items() if a in self._axes})
            
        elif kwargs:
            self.set_scale(dict(kwargs))
            
        else:
            raise TypeError(f"'other' must be str or MetaArray, but got {type(other)}")
        
        return None
    
    def as_standard_type(self) -> AxesFrame:
        """
        t or c -> uint16
        p -> uint32
        z, y, x -> float32
        """
        dtype = lambda a: np.uint16 if a in ["t", "c"] else (np.uint32 if a == Const["ID_AXIS"] else np.float32)
        out = self.__class__(self.astype({a: dtype(a) for a in self.col_axes}))
        out._axes = self._axes
        return out
        
    
    def split(self, axis="c") -> list[AxesFrame]:
        """
        Split DataFrame according to its indices. For example, if self is an DataFrame with columns
        "t", "c", "y", "x" and it is split along "c" axis, then output is a list of DataFrame with 
        columns "t", "y", "x".

        Parameters
        ----------
        axis : str, default is "c"
            Along which axis to split

        Returns
        -------
        list of AxesFrame
            Separate DataFrames.
        """
        out_list = []
        for _, af in self.groupby(axis):
            out = af[af.columns[af.columns != axis]]
            out.set_scale(self)
            out_list.append(out)
        return out_list

    def iter(self, axes:str):
        """
        Iteration along any axes. This method is almost doing the same thing as `groupby`, but sub-
        DataFrames without axes in columns are yielded.

        Parameters
        ----------
        axes : str
            Along which axes to iterate.
            
        Yields
        -------
        tuple and AxesFrame
            slice to generate the AxesFrame.
        """
        indices = [i for i, a in enumerate(self.col_axes) if a in axes]
        outsl = [slice(None)] * len(self.col_axes)
        cols = [a for a in self.col_axes if a not in axes]
        groupkeys = [a for a in axes]
        
        if len(groupkeys) == 0:
            yield (slice(None),), self
        
        else:
            if len(groupkeys) == 1:
                groupkeys = groupkeys[0]
            for sl, af in self.groupby(groupkeys):
                af = af[cols]
                if isinstance(sl, int):
                    sl = (sl,)
                [outsl.__setitem__(i, s) for i, s in zip(indices, sl)]
                yield tuple(outsl), af
    
    def sort(self):
        ids = self._axes.argsort()
        return self[[self._axes[i] for i in ids]]
    
    def proj(self, axis=None):
        if axis is None:
            axis = complement_axes("yx", self._axes)
        cols = [a for a in self.col_axes if a not in axis]
        return self[cols]
        
def tp_no_verbose(func):
    """
    Temporary suppress logging in trackpy.
    """    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        tp.quiet(suppress=True)
        out = func(self, *args, **kwargs)
        tp.quiet(suppress=False)
        return out
    return wrapper

class MarkerFrame(AxesFrame):
    @tp_no_verbose
    @dims_to_spatial_axes
    def link(self, search_range:float|tuple[float, ...], *, memory:int=0, min_dwell:int=0, dims=None,
             **kwargs) -> TrackFrame:
        """
        Link separate points to generate tracks.

        Parameters
        ----------
        search_range : float or tuple of float
            How far a molecule can move in the next frame. Large value causes long calculation time.
        memory : int, default is 0
            How long a molecule can vanish.
        min_dwell : int, default is 0
            Minimum number of frames that single track should dwell.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        TrackFrame
            Result of particle tracking.
        """        
            
        linked = tp.link(pd.DataFrame(self), search_range=search_range, t_column="t", memory=memory, **kwargs)
        
        linked.rename(columns = {"particle":Const["ID_AXIS"]}, inplace=True)
        linked = linked.reindex(columns=[a for a in Const["ID_AXIS"]+str(self.col_axes)])
        
        track = TrackFrame(linked, columns="".join(linked.columns.tolist()))
        track.set_scale(self)
        if min_dwell > 0:
            out = track.filter_stubs(min_dwell)
        else:
            out = track.as_standard_type()
        out.index = np.arange(len(out))
        return out

class TrackFrame(AxesFrame):
    def _renamed_df(self):
        df = pd.DataFrame(self, copy=True, dtype=np.float32)
        df.rename(columns = {"t":"frame", Const["ID_AXIS"]:"particle"}, inplace=True)
        return df
        
    @tp_no_verbose
    def track_drift(self, smoothing=0):
        df = self._renamed_df()
        shift = -tp.compute_drift(df, smoothing=smoothing)
        # trackpy.compute_drift does not return the initial drift so that here we need to start with [0, 0]
        ori = pd.DataFrame({"y":[0.], "x":[0.]}, dtype=np.float32)
        shift = pd.concat([ori, shift], axis=0)
        return MarkerFrame(shift)
    
    @tp_no_verbose
    def msd(self, max_lagt:int=100, detail:bool=False):
        df = self._renamed_df()
        return tp.motion.msd(df, self.scale["x"], self.scale["t"], 
                             max_lagtime=max_lagt, detail=detail)
    
    @tp_no_verbose
    def imsd(self, max_lagt=100):
        df = self._renamed_df()
        return tp.motion.imsd(df, self.scale["x"], self.scale["t"], 
                             max_lagtime=max_lagt)
    
    @tp_no_verbose
    def emsd(self, max_lagt=100, detail=False):
        df = self._renamed_df()
        return tp.motion.emsd(df, self.scale["x"], self.scale["t"], 
                             max_lagtime=max_lagt, detail=detail)
    
    @tp_no_verbose
    def filter_stubs(self, min_dwell=3):
        df = self._renamed_df()
        df = tp.filtering.filter_stubs(df, threshold=min_dwell)
        df.rename(columns = {"frame":"t", "particle":Const["ID_AXIS"]}, inplace=True)
        df = df.astype({"t":np.uint16, Const["ID_AXIS"]:np.uint32})
        out = TrackFrame(df, columns=self.col_axes)
        out.set_scale(self)
        return out.as_standard_type()
    
    def as_path(self):
        df = self[[a for a in self._axes if a != "t"]]
        return PathFrame(df, columns=df._axes)

class PathFrame(AxesFrame):
    def length(self, dims=None):
        if dims is None:
            dims = [a for a in "zyx" if a in self._axes]
        path_len = lambda df: np.sum(np.sqrt(sum(np.diff(df[a])**2 for a in dims)))
        c_axes = complement_axes(dims, self._axes)
        return self.groupby(list(c_axes)).apply(path_len)