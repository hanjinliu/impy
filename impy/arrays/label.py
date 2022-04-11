from __future__ import annotations
import numpy as np

from ._utils._skimage import skimage, skseg
from ._utils import _filters, _structures, _docs
from .bases import MetaArray

from ..utils.axesop import complement_axes
from ..utils.deco import record, dims_to_spatial_axes

def best_dtype(n:int):
    if n < 2**8:
        return np.uint8
    elif n < 2**16:
        return np.uint16
    elif n < 2**32:
        return np.uint32
    else:
        return np.uint64

class Label(MetaArray):
    def __new__(cls, obj, name=None, axes=None, source=None, 
                metadata=None, dtype=None) -> Label:
        if dtype is None:
            dtype = best_dtype(np.max(obj))
        self = super().__new__(cls, obj, name, axes, source, metadata, dtype)
        return self
    
    def increment(self, n: int) -> Label:
        # return view if possible
        if self.max() + n > np.iinfo(self.dtype).max:
            out = self.astype(best_dtype(self.max() + n))
            out[out>0] += n
            return out
        else:
            self[self>0] += n
            return self
    
    def increment_iter(self, axes) -> Label:
        min_nlabel = 0
        imax = np.iinfo(self.dtype).max
        for sl, _ in self.iter(axes):
            self[sl][self[sl]>0] += min_nlabel
            min_nlabel = self[sl].max()
            if min_nlabel > imax:
                raise OverflowError("Number of labels exceeded maximum.")
        return self
    
    def as_larger_type(self):
        if self.dtype == np.uint8:
            return self.astype(np.uint16)
        elif self.dtype == np.uint16:
            return self.astype(np.uint32)
        elif self.dtype == np.uint32:
            return self.astype(np.uint64)
        else:
            raise OverflowError

    def optimize(self) -> Label:
        self.relabel()
        m = self.max()
        if m < 2**8 and np.iinfo(self.dtype).max >= 2**8:
            return self.astype(np.uint8)
        elif m < 2**16 and np.iinfo(self.dtype).max >= 2**16:
            return self.astype(np.uint16)
        elif m < 2**32 and np.iinfo(self.dtype).max >= 2**32:
            return self.astype(np.uint32)
        else:
            return self
    
    def relabel(self) -> Label:
        self.value[:] = skseg.relabel_sequential(self.value)[0]
        return self
    
    
    @dims_to_spatial_axes
    @record
    def expand_labels(self, distance:int=1, *, dims=None) -> Label:
        """
        Expand areas of labels.

        Parameters
        ----------
        distance : int, optional
            The distance to expand, by default 1
        {dims}

        Returns
        -------
        Label
            Same array but labels are updated.
        """        
        labels = self.apply_dask(skseg.expand_labels,
                                c_axes=complement_axes(dims, self.axes),
                                dtype=self.dtype,
                                kwargs=dict(distance=distance)
                                )
        self.value[:] = labels
        
        return self

    def proj(self, axis=None, forbid_overlap=False) -> Label:
        """
        Label projection. This function is useful when zyx-labels are drawn but you want to reduce the 
        dimension.
        
        Parameters
        ----------
        axis : str, optional
            Along which axis projection will be calculated. If None, most plausible one will be chosen.
        forbid_overlap : bool, default is False
            If True and there were any label overlap, this function will raise ValueError.

        Returns
        -------
        Label
            Projected labels.
        """        
        c_axes = complement_axes(axis, self.axes)
        new_labels:Label = np.max(self, axis=axis)
        if forbid_overlap:
            test_array = np.sum(self>0, axis=axis)
            if (test_array>1).any():
                raise ValueError("Label overlapped.")
        new_labels._set_info(self, new_axes=c_axes)
        return new_labels
    
    def add_label(self, label_image):
        label_image = label_image.view(self.__class__).relabel()
        label_image = label_image.increment(self.max())
        self = self.astype(label_image.dtype)
        self[label_image>0] = label_image[label_image>0]
        return self
    
    def delete_label(self, label_ids):
        to_del = np.isin(self.value, label_ids)
        self[to_del] = 0
        return None
        
    def imshow(self, **kwargs):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(skimage.color.label2rgb(self.value, bg_label=0), **kwargs)
        return self
    
    def __truediv__(self, value):
        raise NotImplementedError("Cannot divide label. If you need to divide, convert it to np.ndarray.")
    
    @_docs.write_docs
    @dims_to_spatial_axes
    @record
    def opening(self, radius:float=1, *, dims=None, update:bool=False) -> Label:
        """
        Morphological opening. 

        Parameters
        ----------
        {radius}
        {dims}
        {update}

        Returns
        -------
        Label
            Opened labels
        """        
        disk = _structures.ball_like(radius, len(dims))
        if self.dtype == bool:
            f = _filters.binary_opening
            kwargs = dict(structure=disk)
        else:
            f = _filters.opening
            kwargs = dict(footprint=disk)
        out = (self>0).apply_dask(f, 
                                  c_axes=complement_axes(dims, self.axes), 
                                  dtype=self.dtype,
                                  kwargs=kwargs
                                  )
        self.value[~out] = 0
        return self.optimize()
    