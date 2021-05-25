from __future__ import annotations
from impy.utilcls import Progress
import numpy as np
from .labeledarray import LabeledArray
from ._process import *
from .deco import *
from .func import *
from .specials import PropArray
from .utilcls import *

def _calc_phase_mean(sl, img, periodicity):
    a = 2 * np.pi / periodicity
    out = np.sum(np.exp(1j*a*img[sl]))
    return np.angle(out)/a

def _calc_phase_std(sl, img, periodicity):
    a = 2 * np.pi / periodicity
    return np.sqrt(-2*np.log(np.abs(np.mean(np.exp(1j*a*img[sl])))))/a

class PhaseArray(LabeledArray):
    additional_props = ["dirpath", "metadata", "name", "unit", "border"]
    
    def __new__(cls, obj, name=None, axes=None, dirpath=None, history=None, 
                metadata=None, dtype=None, unit="rad", border=None):
        if dtype is None:
            dtype = np.float32
        if border is None:
            border = {"rad": (0, 2*np.pi), "deg": (0, 360.0)}[unit]
            
        self = super().__new__(cls, obj, name=name, axes=axes, dirpath=dirpath, 
                               history=history, metadata=metadata, dtype=dtype)
        self.unit = unit
        self.border = border
        return self
    
    def __add__(self, value):
        out = super().__add__(value)
        out.fix_border()
        return out
    
    def __iadd__(self, value):
        out = super().__iadd__(value)
        out.fix_border()
        return out
    
    def __sub__(self, value):
        out = super().__sub__(value)
        out.fix_border()
        return out
    
    def __isub__(self, value):
        out = super().__isub__(value)
        out.fix_border()
        return out
    
    def __mul__(self, value):
        out = super().__mul__(value)
        out.fix_border()
        return out
        
    def __imul__(self, value):
        out = super().__imul__(value)
        out.fix_border()
        return out
        
    def __truediv__(self, value):
        out = super().__truediv__(value)
        out.fix_border()
        return out
    
    def __itruediv__(self, value):
        out = super().__itruediv__(value)
        out.fix_border()
        return out
    
    @property
    def periodicity(self):
        a, b = self.border
        return b - a
    
    def fix_border(self) -> None:
        """
        Considering periodic boundary condition, fix the values by `__divmod__` method.
        """        
        self[:] = (self.value - self.border[0]) % self.periodicity + self.border[0]
        return None
    
    def set_border(self, a, b) -> None:
        """
        Set new border safely.

        Parameters
        ----------
        border : tuple[float]
            New border
        """        
        if not (np.isscalar(a) and np.isscalar(b)):
            raise TypeError("Both border values must be scalars.")
            
        if b - a != self.periodicity:
            raise ValueError("New border does not match current periodicity.")
        self.border = (b, a)
        self.fix_border()
        return None
    
    def deg2rad(self) -> PhaseArray:
        if self.unit == "rad":
            raise ValueError("Array is already in radian.")
        np.deg2rad(self, out=self.value[:])
        self.unit = "rad"
        self.border = tuple(np.deg2rad(self.border))
        return self
    
    def rad2deg(self) -> PhaseArray:
        if self.unit == "deg":
            raise ValueError("Array is already in degree.")
        np.rad2deg(self, out=self.value[:])
        self.unit = "deg"
        self.border = tuple(np.rad2deg(self.border))
        return self
    
    @record()
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    def mean_filter(self, radius:float=1, *, dims=None, update:bool=False) -> PhaseArray:
        """
        Mean filter using phase averaging method:
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
        src : array-like
            Source coordinate.
        dst : array-like
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
        out = np.arctan2(out_im, out_re)/a
        return out
    
    @need_labels
    @record(append_history=False)
    def regionprops(self, properties:tuple[str,...]|str=("phase_mean",), *, 
                    extra_properties=None) -> ArrayDict:
        """
        Run skimage's regionprops() function and return the results as PropArray, so
        that you can access using flexible slicing. For example, if a tcyx-image is
        analyzed with properties=("X", "Y"), then you can get X's time-course profile
        of channel 1 at label 3 by prop["X"]["p=5;c=1"] or prop.X["p=5;c=1"].
        In PhaseArray, instead of mean_intensity you should use "phase_mean". The
        phase_mean function is included so that it can be passed in `properties` argument.

        Parameters
        ----------
        properties : iterable, optional
            properties to analyze, see skimage.measure.regionprops.
        extra_properties : iterable of callable, optional
            extra properties to analyze, see skimage.measure.regionprops.

        Returns
        -------
            ArrayDict of PropArray
            
        Example
        -------
        Measure region properties around single molecules.
        >>> coords = reference_img.centroid_sm()
        >>> img.specify(coords, 3, labeltype="circle")
        >>> props = img.regionprops()
        """        
        def phase_mean(sl, img):
            return _calc_phase_mean(sl, img, self.periodicity)
        def phase_std(sl, img):
            return _calc_phase_std(sl, img, self.periodicity)
        additional_props = {"phase_mean": phase_mean, "phase_std": phase_std}
                
        # check arguments
        if isinstance(properties, str):
            properties = (properties,)
        
        if extra_properties is None:
            extra_properties = tuple()
        
        # add extra_properties and move additional properties to extra_properties
        properties = properties + tuple(ex.__name__ for ex in extra_properties)
        for prop in properties:
            if prop in additional_props.keys():
                extra_properties = (additional_props[prop],) + extra_properties
                
        if "p" in self.axes:
            # this dimension will be label
            raise ValueError("axis 'p' is forbidden in regionprops().")
        
        prop_axes = complement_axes(self.labels.axes, self.axes)
        shape = self.sizesof(prop_axes)
        
        out = ArrayDict({p: PropArray(np.empty((self.labels.max(),) + shape, dtype=np.float32),
                                      name=self.name, 
                                      axes="p"+prop_axes,
                                      dirpath=self.dirpath,
                                      propname=p)
                         for p in properties})
        
        # calculate property value for each slice
        for sl, img in self.iter(prop_axes, exclude=self.labels.axes):
            props = skmes.regionprops(self.labels, img, cache=False,
                                      extra_properties=extra_properties)
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                # Both sides have length of p-axis (number of labels) so that values
                # can be correctly substituted.
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]
        
        for parr in out.values():
            parr.set_scale(self)
        return out



