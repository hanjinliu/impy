from __future__ import annotations
from .axes import Axes, ImageAxesError
from .metaarray import MetaArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inspect import signature
from scipy import optimize as opt
from .func import *
from .deco import *
from .axes import ORDER
from .utilcls import *

SCALAR_PROP = (
    "area", "bbox_area", "convex_area", "eccentricity", "equivalent_diameter", "euler_number",
    "extent", "feret_diameter_max", "filled_area", "label", "major_axis_length", "max_intensity",
    "mean_intensity", "min_intensity", "minor_axis_length", "orientation", "perimeter",
    "perimeter_crofton", "solidity", "phase_mean")

tp = ImportOnRequest("trackpy")

class PropArray(MetaArray):
    additional_props = ["dirpath", "metadata", "name", "propname"]
    def __new__(cls, obj, *, name=None, axes=None, dirpath=None, 
                metadata=None, propname=None, dtype=None):
        if propname is None:
            propname = "User_Defined"
        
        if dtype is None and propname in SCALAR_PROP:
            dtype = np.float32
        elif dtype is None:
            dtype = object
        else:
            dtype = dtype
            
        self = super().__new__(cls, obj, name, axes, dirpath, metadata, dtype=dtype)
        self.propname = propname
        
        return self
    
    def __repr__(self):
        if self.axes.is_none():
            shape_info = self.shape
        else:
            shape_info = ", ".join([f"{s}({o})" for s, o in zip(self.shape, self.axes)])

        return f"\n"\
               f"    shape     : {shape_info}\n"\
               f"    dtype     : {self.dtype}\n"\
               f"  directory   : {self.dirpath}\n"\
               f"original image: {self.name}\n"\
               f"property name : {self.propname}\n"
    
    def plot(self, along=None, cmap="jet", cmap_range=(0, 1)):
        if self.dtype == object:
            raise TypeError(f"Cannot call plot_profile for {self.propname} "
                            "because dtype == object.")
        if along is None:
            along = find_first_appeared("tzpyxc", include=self.axes)
        
        iteraxes = del_axis(self.axes, self.axisof(along))
        cmap = plt.get_cmap(cmap)
        positions = np.linspace(*cmap_range, self.size//self.sizeof(along), endpoint=False)
        x = np.arange(self.sizeof(along))*self.scale[along]
        for i, (sl, y) in enumerate(self.iter(iteraxes)):
            plt.plot(x, y, color=cmap(positions[i]))
        
        plt.title(f"{self.propname}")
        plt.xlabel(along)
        plt.show()
        
        return self
    
    def hist(self, along="p", bins=None, cmap="jet", cmap_range=(0, 1)):
        if self.dtype == object:
            raise TypeError(f"Cannot call plot_profile for {self.propname} "
                            "because dtype == object.")
        
        iteraxes = del_axis(self.axes, self.axisof(along))
        cmap = plt.get_cmap(cmap)
        positions = np.linspace(*cmap_range, self.size//self.sizeof(along), endpoint=False)
        for i, (sl, y) in enumerate(self.iter(iteraxes)):
            plt.hist(y, color=cmap(positions[i]), bins=bins, alpha=0.5)
        
        plt.title(f"{self.propname}")
        plt.show()
        
        return self
    
    def curve_fit(self, f, p0=None, dims="t", return_fit=True) -> ArrayDict:
        """
        Run scipy.optimize.curve_fit for each dimesion.

        Parameters
        ----------
        f : callable
            Model function.
        p0 : array or callable, optional
            Initial parameter. Callable object that estimate initial paramter can also 
            be passed here.
        dims : str, by default "t"
            Along which axes fitting algorithms are conducted.
        return_fit : bool, by default True
            If fitting trajectories are returned. If the input array is large, this will 
            save much memory.

        Returns
        -------
        ArrayDict
            params : Fitting parameters
            covs : Covariances
            fit : fitting trajectories (if return_fit==True)
        """        
        c_axes = complement_axes(dims, all_axes=self.axes)
        
        if len(dims)!=1:
            raise NotImplementedError("Only 1-dimensional fitting is implemented.")
        
        n_params = len(signature(f).parameters)-1
        
        params = np.empty(self.sizesof(c_axes) + (n_params,), dtype=np.float32)
        errs = np.empty(self.sizesof(c_axes) + (n_params,), dtype=np.float32)
        if return_fit:
            fit = np.empty(self.sizesof(c_axes) + (self.sizeof(dims),), dtype=self.dtype)
            
        xdata = np.arange(self.sizeof(dims))*self.scale[dims]
        
        for sl, data in self.iter(c_axes, exclude=dims):
            p0_ = p0 if not callable(p0) else p0(data)
            result = opt.curve_fit(f, xdata, data, p0_)
            params[sl], cov = result
            errs[sl] = np.sqrt(np.diag(cov))
            if return_fit:
                fit[sl] = f(xdata, *(result[0]))
        
        # set infos
        params = params.view(self.__class__)
        errs = errs.view(self.__class__)
        params._set_info(self, new_axes=del_axis(self.axes, dims)+"m")
        errs._set_info(self, new_axes=del_axis(self.axes, dims)+"m")
        if return_fit:
            fit = fit.view(self.__class__)
            fit._set_info(self, new_axes=del_axis(self.axes, dims)+dims)
        
        if return_fit:
            return ArrayDict(params=params, errs=errs, fit=fit)
        else:
            return ArrayDict(params=params, errs=errs)
       


class AxesFrame(pd.DataFrame):
    _metadata=["_axes"]
    @property
    def _constructor(self):
        return self.__class__
    
    def __init__(self, data=None, columns=None, **kwargs):
        if isinstance(columns, (str, Axes)):
            kwargs["columns"] = [a for a in columns]
        elif isinstance(data, AxesFrame):
            kwargs["columns"] = data.columns.tolist()
        else:
            kwargs["columns"] = columns
            
        super().__init__(data, **kwargs)
        self._axes = Axes(columns)
    
    def _get_coords_cols(self):
        return "".join(a for a in self.columns if len(a) == 1)
    
    def get_coords(self):
        return self[self.columns[self.columns.isin([a for a in self.columns if len(a) == 1])]]
    
    
    def __getitem__(self, k):
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
        if isinstance(value, str):
            self._axes.axes = value
            self.columns = [a for a in value]
        else:
            raise TypeError("Only str can be set to `col_axes`.")
    
    
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
        if self._axes.is_none():
            raise ImageAxesError("Frame does not have axes.")
        
        elif isinstance(other, dict):
            # check if all the keys are contained in axes.
            for a, val in other.items():
                if a not in self._axes:
                    raise ImageAxesError(f"Image does not have axis {a}.")    
                elif not np.isscalar(val):
                    raise TypeError(f"Cannot set non-numeric value as scales.")
            self._axes.scale.update(other)
            
        elif isinstance(other, (AxesFrame, MetaArray)):
            self.set_scale({a: s for a, s in other.scale.items() if a in self._axes})
            
        elif kwargs:
            self.set_scale(dict(kwargs))
            
        else:
            raise TypeError(f"'other' must be str or MetaArray, but got {type(other)}")
        
        return None
    
    def as_standard_type(self):
        dtype = lambda a: np.uint16 if a in "tc" else (np.uint32 if a == "p" else np.float32)
        out = self.__class__(self.astype({a: dtype(a) for a in self.col_axes}))
        out._axes = self._axes
        return out
        
    
    def split(self, axis="c"):
        out_list = []
        for _, af in self.groupby(axis):
            out = af[af.columns[af.columns != axis]]
            out.set_scale(self)
            out_list.append(out)
        return out_list

    def iter(self, axes):
        reg = "|".join(a for a in self.col_axes if a not in axes)
        groupkeys = [a for a in axes]
        if len(groupkeys) == 0:
            yield slice(None), self
        
        else:
            for sl, af in self.groupby(groupkeys):
                af = af.filter(regex=reg)
                yield sl, af
    
    def sort(self):
        ids = np.argsort([ORDER[k] for k in self._axes])
        return self[[self._axes[i] for i in ids]]
  
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
    def link(self, search_range, memory=0, min_dwell=0, predictor=None, adaptive_stop=None, adaptive_step=0.95,
             neighbor_strategy=None, link_strategy=None, dist_func=None, to_eucl=None):
        
        linked = tp.link(pd.DataFrame(self), search_range=search_range, t_column="t", memory=memory, predictor=predictor, 
                         adaptive_stop=adaptive_stop, adaptive_step=adaptive_step, neighbor_strategy=neighbor_strategy, 
                         link_strategy=link_strategy, dist_func=dist_func, to_eucl=to_eucl)
        
        linked.rename(columns = {"particle":"p"}, inplace=True)
        linked = linked.reindex(columns=[a for a in "p"+str(self.col_axes)])
        
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
        df = pd.DataFrame(self, copy=True, dtype="float32")
        df.rename(columns = {"t":"frame", "p":"particle"}, inplace=True)
        return df
        
    @tp_no_verbose
    def track_drift(self, smoothing=0, show_drift=True):
        df = self._renamed_df()
        shift = -tp.compute_drift(df, smoothing=smoothing)
        # trackpy.compute_drift does not return the initial drift so that here we need to start with [0, 0]
        ori = pd.DataFrame({"y":[0.], "x":[0.]}, dtype="float32")
        shift = pd.concat([ori, shift], axis=0)
        show_drift and plot_drift(shift)
        return MarkerFrame(shift)
    
    @tp_no_verbose
    def msd(self, max_lagt=100, detail=False):
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
        df.rename(columns = {"frame":"t", "particle":"p"}, inplace=True)
        df = df.astype({"t":np.uint16, "p":np.uint32})
        out = TrackFrame(df, columns=self.col_axes)
        out.set_scale(self)
        return out.as_standard_type()