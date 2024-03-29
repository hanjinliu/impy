from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from ._utils import _filters, _structures
from .specials import PropArray
from .labeledarray import LabeledArray
from .bases import MetaArray

from ..utils.axesop import complement_axes
from ..utils.deco import check_input_and_output, dims_to_spatial_axes, same_dtype
from ._utils import _misc, _docs
from ..collections import DataDict
from .._types import Dims, PaddingMode
from ..array_api import xp


if TYPE_CHECKING:
    from .imgarray import ImgArray

def _calc_phase_mean(sl, img, periodicity):
    a = 2 * np.pi / periodicity
    out = np.sum(np.exp(1j*a*img[sl]))
    return np.angle(out)/a

def _calc_phase_std(sl, img, periodicity):
    a = 2 * np.pi / periodicity
    return np.sqrt(-2*np.log(np.abs(np.mean(np.exp(1j*a*img[sl])))))/a

class PhaseArray(LabeledArray):
    additional_props = ["_source", "_metadata", "_name", "unit", "border"]
    unit: str
    border: tuple[float, float]
    
    def __new__(cls, obj, name=None, axes=None, source=None, 
                metadata=None, dtype=None, unit="rad", border=None) -> PhaseArray:
        if dtype is None:
            dtype = np.float32
        if border is None:
            border = {"rad": (0, 2*np.pi), "deg": (0, 360.0)}[unit]
            
        self = super().__new__(cls, obj, name=name, axes=axes, source=source, 
                               metadata=metadata, dtype=dtype)
        self.unit = unit
        self.border = border
        return self
    
    def __add__(self, value) -> PhaseArray:
        out = super().__add__(value)
        out._fix_border()
        return out
    
    def __iadd__(self, value) -> PhaseArray:
        out = super().__iadd__(value)
        out._fix_border()
        return out
    
    def __sub__(self, value) -> PhaseArray:
        out = super().__sub__(value)
        out._fix_border()
        return out
    
    def __isub__(self, value) -> PhaseArray:
        out = super().__isub__(value)
        out._fix_border()
        return out
    
    def __mul__(self, value) -> PhaseArray:
        out = super().__mul__(value)
        out._fix_border()
        return out
        
    def __imul__(self, value) -> PhaseArray:
        out = super().__imul__(value)
        out._fix_border()
        return out
        
    def __truediv__(self, value) -> PhaseArray:
        out = super().__truediv__(value)
        out._fix_border()
        return out
    
    def __itruediv__(self, value) -> PhaseArray:
        out = super().__itruediv__(value)
        out._fix_border()
        return out
    
    @property
    def periodicity(self) -> float:
        """Return the periodicity of current border."""
        a, b = self.border
        return b - a
    
    def _fix_border(self) -> None:
        """
        Considering periodic boundary condition, fix the values by `__mod__` method.
        """        
        self[:] = (self.value - self.border[0]) % self.periodicity + self.border[0]
        return None
    
    def set_border(self, a: float, b: float) -> None:
        """
        Set new border safely.

        Parameters
        ----------
        a : float
            New lower border.
        b : float
            New higher border.
        """        
        if not (np.isscalar(a) and np.isscalar(b)):
            raise TypeError("Both border values must be scalars.")
            
        if b - a != self.periodicity:
            raise ValueError("New border does not match current periodicity.")
        self.border = (b, a)
        self._fix_border()
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
    
    @check_input_and_output
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    def binning(self, binsize: int = 2, *, check_edges: bool = True, dims: Dims = None):
        if binsize == 1:
            return self
        img_to_reshape, shape, scale_ = _misc.adjust_bin(self.value, binsize, check_edges, dims, self.axes)
        
        reshaped_img = img_to_reshape.reshape(shape)
        axes_to_reduce = tuple(i*2+1 for i in range(self.ndim))
        a = 2 * np.pi / self.periodicity
        out = np.sum(np.exp(1j*a*reshaped_img), axis=axes_to_reduce)
        out = np.angle(out)/a
        out: PhaseArray = out.view(self.__class__)
        out._set_info(self)
        out.axes = self.axes.copy()  # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out
    
    @check_input_and_output
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    def mean_filter(self, radius: float = 1, *, dims: Dims = None, update: bool = False) -> PhaseArray:
        r"""
        Mean filter using phase averaging method:
        
        :math:`\arg{\sum_{k}{e^{i X_k}}}`

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
        disk = _structures.ball_like(radius, len(dims))
        a = 2*np.pi/self.periodicity
            
        return self._apply_dask(_filters.phase_mean_filter,
                               c_axes=complement_axes(dims, self.axes),
                               args=(disk, a),
                               dtype=self.dtype)
    
    @dims_to_spatial_axes
    def imshow(self, label: bool = False, dims = 2, alpha=0.3, **kwargs):
        if "cmap" not in kwargs and self.ndim > 1:
            kwargs["cmap"] = "hsv"
        return super().imshow(label=label, dims=dims, alpha=alpha, **kwargs)
    
    def reslice(self, src, dst, *, order: int = 1) -> PropArray:
        """
        Measure line profile iteratively for every slice of image. 
        
        Because input is phase, we can not apply standard interpolation to calculate
        intensities on float-coordinates.

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
        out_re = vec_re.reslice(src, dst, order=order)
        out_im = vec_im.reslice(src, dst, order=order)
        out = np.arctan2(out_im, out_re)/a
        return out
    
    def regionprops(self, properties: tuple[str,...] | str = ("phase_mean",), *, 
                    extra_properties = None) -> DataDict[str, PropArray]:
        """
        Measure region properties.
        
        Run ``skimage``'s ``regionprops()`` function and return the results as PropArray, so
        that you can access using flexible slicing. For example, if a tcyx-image is
        analyzed with ``properties=("X", "Y")``, then you can get X's time-course profile
        of channel 1 at label 3 by ``prop["X"]["N=5;c=1"]`` or ``prop.X["N=5;c=1"]``.
        In PhaseArray, instead of mean_intensity you should use "phase_mean". The
        phase_mean function is included so that it can be passed in ``properties`` argument.

        Parameters
        ----------
        properties : iterable, optional
            properties to analyze, see skimage.measure.regionprops.
        extra_properties : iterable of callable, optional
            extra properties to analyze, see skimage.measure.regionprops.

        Returns
        -------
            DataDict of PropArray
            
        Example
        -------
        Measure region properties around single molecules.
            >>> coords = reference_img.centroid_sm()
            >>> img.specify(coords, 3, labeltype="circle")
            >>> props = img.regionprops()
        """
        from skimage.measure import regionprops
        def phase_mean(sl, img):
            return _calc_phase_mean(sl, img, self.periodicity)
        def phase_std(sl, img):
            return _calc_phase_std(sl, img, self.periodicity)
        id_axis = "N"
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
                
        if id_axis in self.axes:
            # this dimension will be label
            raise ValueError(f"axis {id_axis} is forbidden in regionprops().")
        
        prop_axes = complement_axes(self.labels.axes, self.axes)
        shape = self.sizesof(prop_axes)
        
        out = DataDict(
            {p: PropArray(
                np.empty((self.labels.max(),) + shape, dtype=np.float32),
                name=self.name+"-prop", 
                axes=[id_axis]+prop_axes,
                source=self.source,
                propname=p
            )
            for p in properties}
        )
        
        # calculate property value for each slice
        for sl, img in self.iter(prop_axes, exclude=self.labels.axes):
            props = regionprops(
                self.labels,
                img,
                cache=False,
                extra_properties=extra_properties,
            )
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                # Both sides have length of p-axis (number of labels) so that values
                # can be correctly substituted.
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]
        
        for parr in out.values():
            parr.set_scale(self)
        return out

    def as_exp(self) -> ImgArray:
        a = 2 * np.pi / self.periodicity
        exps = np.exp(1j / a * self.value)
        from .imgarray import ImgArray
        return ImgArray(
            exps, name=self.name, axes=self.axes, source=self.source,
            metadata=self.metadata
        )

    @_docs.write_docs
    @same_dtype(asfloat=True)
    @dims_to_spatial_axes
    def map_coordinates(
        self,
        coordinates,
        *, 
        mode: PaddingMode = "constant",
        cval: float = 0,
        order: int = 3,
        prefilter: bool | None = None,
        dims: Dims = None,
    ):
        r"""
        Coordinate mapping in the image. See ``scipy.ndimage.map_coordinates``.
        
        For a ``PhaseArray``, standard interpolation does not work because of its specific
        addition rule. Phase image is first converted into a complex array by :math:`e^{-i \psi}`
        and the phase is restored by logarithm after interpolation.

        Parameters
        ----------
        coordinates : ArrayLike
            Interpolation coordinates. Must be (D, N) or (D, X_1, ..., X_D) shape.
        {mode}
        {cval}
        {order}
        prefilter : bool, optional
            Spline prefilter applied to the array. By default set to True if ``order`` is larger
            than 1.
        {dims}

        Returns
        -------
        PhaseArray
            Transformed image.
        """
        coords = xp.asarray(coordinates)
        c_axes = complement_axes(dims, self.axes)
        
        if coords.ndim != 2:
            drop_axis = []
        else:
            drop_axis = [self.axisof(a) for a in dims[:-1]]
        
        if prefilter is None:
            prefilter = order > 1
        
        a = 2 * np.pi / self.periodicity
        complex_array: PhaseArray = np.exp(1j * a * self)

        out = complex_array._apply_dask(
            xp.ndi.map_coordinates,
            c_axes,
            dtype=complex_array.dtype,
            drop_axis=drop_axis,
            args=(coords,),
            kwargs=dict(mode=mode, cval=cval, order=order, prefilter=prefilter),
        )
        
        out: np.ndarray = np.angle(out.value) / a
        
        if coords.ndim == len(dims) + 1:
            if isinstance(coordinates, MetaArray):
                new_axes = c_axes + coordinates.axes[1:]
            else:
                new_axes = self.axes
        else:
            if isinstance(coordinates, MetaArray):
                new_axes = c_axes + coordinates.axes[1:2]
            else:
                new_axes = c_axes + ["#"]
            
        out: PhaseArray = out.view(self.__class__)
        out._set_info(self, new_axes=new_axes)
        return out
