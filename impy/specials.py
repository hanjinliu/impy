from __future__ import annotations
from .axes import ImageAxesError
from .metaarray import MetaArray
import numpy as np
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
from inspect import signature
from scipy import optimize as opt
from .func import *
from .deco import *
from .utilcls import *

SCALAR_PROP = (
    "area", "bbox_area", "convex_area", "eccentricity", "equivalent_diameter", "euler_number",
    "extent", "feret_diameter_max", "filled_area", "label", "major_axis_length", "max_intensity",
    "mean_intensity", "min_intensity", "minor_axis_length", "orientation", "perimeter",
    "perimeter_crofton", "solidity")

class PropArray(MetaArray):
    def __new__(cls, obj, *, name=None, axes=None, dirpath=None, 
                metadata=None, propname=None, dtype=None):
        if propname is None:
            propname = "User_Defined"
        
        if dtype is None and propname in SCALAR_PROP:
            dtype = "float32"
        else:
            dtype = object
            
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
            along = self.axes[-1]
        
        iteraxes = del_axis(self.axes, self.axisof(along))
        plt.figure(figsize=(4, 1.7))
        cmap = plt.get_cmap(cmap)
        positions = np.linspace(*cmap_range, self.size//self.sizeof(along), endpoint=False)
        x = np.arange(self.sizeof(along))*self.scale[along]
        for i, (sl, y) in enumerate(self.iter(iteraxes)):
            plt.plot(x, y, color=cmap(positions[i]))
        
        plt.title(f"{self.propname}")
        plt.xlabel(along)
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
        
        n_params = len(signature(f).parameters)-1 if callable(p0) else len(p0)
        
        params = np.empty(self.sizesof(c_axes) + (n_params,), dtype="float32")
        covs = np.empty(self.sizesof(c_axes) + (n_params, n_params), dtype="float32")
        if return_fit:
            fit = np.empty(self.sizesof(c_axes) + (self.sizeof(dims),), dtype=self.dtype)
            
        xdata = np.arange(self.sizeof(dims))*self.scale[dims]
        
        for sl, data in self.iter(c_axes, exclude=dims):
            p0_ = p0 if not callable(p0) else p0(data)
            result = opt.curve_fit(f, xdata, data, p0_)
            params[sl], covs[sl] = result
            if return_fit:
                fit[sl] = f(xdata, *(result[0]))
        
        # set infos
        params = params.view(self.__class__)
        covs = covs.view(self.__class__)
        params._set_info(self, new_axes=del_axis(self.axes, dims)+"m")
        covs._set_info(self, new_axes=del_axis(self.axes, dims)+"mn")
        if return_fit:
            fit = fit.view(self.__class__)
            fit._set_info(self, new_axes=del_axis(self.axes, dims)+dims)
        
        if return_fit:
            return ArrayDict(params=params, covs=covs, fit=fit)
        else:
            return ArrayDict(params=params, covs=covs)
    
    
    def melt(self):
        """
        Make a melted array.
        
        Example
        -------
        Suppose:
        A["t=0"] = [[1,3], [5,4]]
        A["t=1"] = [[6,7]]
        
        after running A.melt():
        [[0, 1, 3],
         [0, 5, 4],
         [1, 6, 7]]

        Returns
        -------
        MarkerArray
            Melter array with r-axis being melted axis. If self.axes="tc" and its contents
            have yx axes, then the length of r-axis of returned MarkerArray will be 4.]
        """        
        out = []
        dtype = "uint16"
        for sl, data in self.iter(self.axes, False):
            if not isinstance(data, MarkerArray):
                raise TypeError("In melt(), all the object must be MarkerArray, "
                                f"but got {type(data)}")
            if not data.axes in ("rp","pr"):
                raise ImageAxesError("In melt(), all the object must have "
                                     f"'p' and 'r' axes, but got {data.axes}")
            
            for _, a in data.iter("p", False):
                out.append(sl + tuple(a))
                if a.dtype.kind == "f":
                    dtype = "float32"
        out = np.array(out, dtype=dtype)
        n_spatial_dim = out.shape[1] - len(self.axes)
        cols = str(self.axes) + {1:"x", 2:"yx", 3:"zyx"}[n_spatial_dim]
        return MarkerFrame(out, columns=cols, dtype=dtype)
    
        
    def _set_info(self, other, new_axes:str="inherit"):
        super()._set_info(other, new_axes)
        self.propname = other.propname
        return None

    
class MarkerArray(MetaArray):    
    def plot(self, **kwargs):
        if self.ndim != 2:
            raise ValueError("Cannot plot non-2D markers.")
        kw = dict(color="red", marker="x")
        kw.update(kwargs)
        plt.scatter(self["r=1"], self["r=0"], **kw)
        return None

class IndexArray(MarkerArray):
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                metadata=None, dtype=None):
        if dtype is None:
            dtype = "uint16"
        return super().__new__(cls, obj, name=name, axes=axes, dirpath=dirpath,
                               metadata=metadata, dtype=dtype)


class AxesFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return self.__class__
    
    def __init__(self, data=None, columns=None, **kwargs):
        if isinstance(columns, str):
            columns = [a for a in columns]
        super().__init__(data, columns=columns, **kwargs)
    
    @property
    def col_axes(self):
        return "".join(self.columns.values)
    
    @col_axes.setter
    def col_axes(self, value):
        if isinstance(value, str):
            self.columns = [a for a in value]
        else:
            raise TypeError("Only str can be set to `col_axes`.")
    
    def split(self, axis="c"):
        a_unique = self[axis].unique()
        return [self[self[axis]==a] for a in a_unique]


class MarkerFrame(AxesFrame):
    def link(self, search_range, memory=0, predictor=None, adaptive_stop=None, adaptive_step=0.95,
             neighbor_strategy=None, link_strategy=None, dist_func=None, to_eucl=None):
        
        linked = tp.link(self, search_range=search_range, t_column="t", memory=memory, predictor=predictor, 
                         adaptive_stop=adaptive_stop, adaptive_step=adaptive_step, neighbor_strategy=neighbor_strategy, 
                         link_strategy=link_strategy, dist_func=dist_func, to_eucl=to_eucl)
        
        linked.rename(columns = {"particle":"p"}, inplace=True)
        linked = linked.reindex(columns=[a for a in "p"+self.col_axes])
        
        return TrackFrame(linked)
        

class TrackFrame(AxesFrame):
    def id(self, p_id):
        return self[self.p==p_id]

    