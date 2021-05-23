from __future__ import annotations
from impy.utilcls import Progress
import numpy as np
from .labeledarray import LabeledArray
from ._process import *
from .deco import *
from .func import *
from .specials import PropArray

class PhaseArray(LabeledArray):
    additional_props = ["dirpath", "metadata", "name", "unit", "periodicity"]
    
    def __new__(cls, obj, name=None, axes=None, dirpath=None, history=None, 
                metadata=None, dtype=None, unit="rad", periodicity=None):
        if dtype is None:
            dtype = np.float32
        if periodicity is None:
            periodicity = {"rad": 2*np.pi, "deg": 360.0}[unit]
            
        self = super().__new__(cls, obj, name=name, axes=axes, dirpath=dirpath, 
                               history=history, metadata=metadata, dtype=dtype)
        self.unit = unit
        self.periodicity = periodicity
        return self
        
    @record()
    def deg2rad(self, *, update=False):
        if self.unit == "rad":
            raise ValueError("Array is already in radian.")
        out = np.deg2rad(self)
        out.history.pop(-1)
        out.unit = "rad"
        out.periodicity = np.deg2rad(out.periodicity)
        return out
    
    @record()
    def rad2deg(self, *, update=False):
        if self.unit == "deg":
            raise ValueError("Array is already in degree.")
        out = np.rad2deg(self)
        out.history.pop(-1)
        out.unit = "deg"
        out.periodicity = np.rad2deg(out.periodicity)
        return out
    
    @record()
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    def mean_filter(self, radius:float=1, *, dims=None, update:bool=False) -> PhaseArray:
        """
        Mean filter using phase averaging method, which is:
        arg(sum(e^j(X0 + X1 + ...)))

        Parameters
        ----------
        radius : float, default is 1
            Radius of kernel.
        dims : str or int, optional
            Spatial dimensions.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        PhaseArray
            Filtered image.
        """        
        disk = ball_like(radius, len(dims))
        a = 2*np.pi/self.periodicity
            
        out = self.parallel(phase_mean_, complement_axes(dims, self.axes), disk, a, outdtype=self.dtype)
        return out
    
    def imshow(self, dims="yx", **kwargs):
        if "cmap" not in kwargs:
            kwargs["cmap"] = "hsv"
        return super().imshow(dims=dims, **kwargs)
    
    def quiver(self, steps=None, dims="yx", **kwargs):
        ny, nx = self.sizesof(dims)
        if steps is None:
            stepy = ny//16
            stepx = nx//16
        else:
            stepy, stepx = steps
        
        yy, xx = np.mgrid[:ny:stepy, :nx:stepx]
        phase = self.value[::stepy, ::stepx]
        plt.imshow(np.zeros_like(phase), cmap="gray")
        plt.quiver(xx, yy, np.cos(phase), np.sin(phase), color="red", units="dots", 
                   angles="xy", scale_units="xy", lw=1)
        self.hist()
        return self
    
    @dims_to_spatial_axes
    def reslice(self, src, dst, *, order:int=1, dims=None) -> PropArray:
        """
        Measure line profile iteratively for every slice of image. Because input is phase, we can
        not apply standard interpolation to calculate intensities on float-coordinates.

        Parameters
        ----------
        src : array, shape (2,)
            Source coordinate.
        dst : array, shape (2,)
            Destination coordinate.
        order : int, default is 1
            Spline interpolation order.
        dims : int or str, optional
            Spatial dimensions.

        Returns
        -------
        PropArray
            Line scans.
        """        
        a = 2*np.pi/self.periodicity
        vec_re = np.cos(a*self).view(LabeledArray)
        vec_im = np.sin(a*self).view(LabeledArray)
        with Progress("reslice"):
            out_re = vec_re.reslice(src, dst, order=order, dims=dims)
            out_im = vec_im.reslice(src, dst, order=order, dims=dims)
        out = (np.arctan2(out_im, out_re)/a)
        return out