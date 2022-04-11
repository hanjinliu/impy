from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import numpy as np
from inspect import signature
from scipy import optimize as opt
from .bases import MetaArray
from ..axes import ImageAxesError
from ..utils.axesop import complement_axes, find_first_appeared, del_axis
from ..collections import DataDict

if TYPE_CHECKING:
    from ..frame import AxesFrame

SCALAR_PROP = (
    "area", "bbox_area", "convex_area", "eccentricity", "equivalent_diameter", "euler_number",
    "extent", "feret_diameter_max", "filled_area", "label", "major_axis_length", "max_intensity",
    "mean_intensity", "min_intensity", "minor_axis_length", "orientation", "perimeter",
    "perimeter_crofton", "solidity", "phase_mean", "phase_std")


class PropArray(MetaArray):
    additional_props = ["_source", "_metadata", "_name", "propname"]
    def __new__(cls, obj, *, name=None, axes=None, source=None, 
                metadata=None, propname=None, dtype=None):
        if propname is None:
            propname = "User_Defined"
        
        if dtype is None and propname in SCALAR_PROP:
            dtype = np.float32
        elif dtype is None:
            dtype = object
        else:
            dtype = dtype
            
        self = super().__new__(cls, obj, name, axes, source, metadata, dtype=dtype)
        self.propname = propname
        
        return self
    
    def _repr_dict_(self):
        return {
            "name": self.name,
            "shape": self.shape_info,
            "dtype": self.dtype,
            "source": self.source,
            "property name": self.propname}
    
    def plot(self, along=None, cmap="jet", cmap_range=(0, 1)):
        """
        Plot all the results with x-axis defined by `along`.

        Parameters
        ----------
        along : str, optional
            Which axis will be the x-axis of plot, by default None
        cmap : str, default is "jet"
            Colormap of each graph.
        cmap_range : tuple, default is (0, 1)
            Range of float for colormap iteration.
        """        
        import matplotlib.pyplot as plt
        if self.dtype == object:
            raise TypeError(f"Cannot call plot_profile for {self.propname} "
                            "because dtype == object.")
        if along is None:
            along = find_first_appeared("tzp<yxc", include=self.axes)
        
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
    
    def hist(self, along: str = "p", bins: int = None, cmap: str = "jet", 
             cmap_range: tuple[float, float] = (0., 1.)) -> PropArray:
        """
        Plot histogram.

        Parameters
        ----------
        along : str, optional
            Which axis will be the x-axis of plot, by default None
        bins : int, optional
            Bin number of histogram.
        cmap : str, default is "jet"
            Colormap of each graph.
        cmap_range : tuple, default is (0, 1)
            Range of float for colormap iteration.
        
        Returns
        -------
        self
        """        
        import matplotlib.pyplot as plt

        if self.dtype == object:
            raise TypeError(f"Cannot call plot_profile for {self.propname} "
                            "because dtype == object.")
        
        if along is None:
            along = find_first_appeared("pxyzt<c", include=self.axes)
            
        iteraxes = del_axis(self.axes, self.axisof(along))
        cmap = plt.get_cmap(cmap)
        positions = np.linspace(*cmap_range, self.size//self.sizeof(along), endpoint=False)
        for i, (sl, y) in enumerate(self.iter(iteraxes)):
            plt.hist(y, color=cmap(positions[i]), bins=bins, alpha=0.5)
        
        plt.title(f"{self.propname}")
        plt.show()
        
        return self
    
    def curve_fit(self, f: Callable, p0 = None, dims = "t", return_fit: bool = True) -> DataDict[str, PropArray]:
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
        DataDict
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
            return DataDict(params=params, errs=errs, fit=fit)
        else:
            return DataDict(params=params, errs=errs)
    
    def as_frame(self, colname: str = "f") -> AxesFrame:
        """
        N-dimensional data to DataFrame. The intensity data is stored in the `colname` column.

        Parameters
        ----------
        colname : str, default is "f"
            The name of new column.

        Returns
        -------
        AxesFrame
            DataFrame with PropArray data.
        """        
        from ..frame import AxesFrame
        if colname in self.axes:
            raise ImageAxesError(f"Axis {colname} already exists.")
        if self.dtype == object:
            raise TypeError("Cannot make AxesFrame with dtype object.")
        indices = np.indices(self.shape)
        new_axes = str(self.axes) + colname
        data_list = [ind.ravel() for ind in indices] + [self.value.ravel()]
        data_dict = {a: data for a, data in zip(new_axes, data_list)}
        df = AxesFrame(data_dict, columns=new_axes)
        df.set_scale(self)
        return df
        
