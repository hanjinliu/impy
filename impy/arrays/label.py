from __future__ import annotations
from skimage.color import label2rgb
from .utils._skimage import skseg
import matplotlib.pyplot as plt
from ..func import *
from ..utilcls import *
from ..deco import *
from .bases import HistoryArray

def best_dtype(n:int):
    if n < 2**8:
        return np.uint8
    elif n < 2**16:
        return np.uint16
    elif n < 2**32:
        return np.uint32
    else:
        return np.uint64

class Label(HistoryArray):
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                history=None, metadata=None, dtype=None):
        if dtype is None:
            dtype = best_dtype(np.max(obj))
        self = super().__new__(cls, obj, name, axes, dirpath, history, metadata, dtype)
        return self
    
    def increment(self, n:int):
        # return view if possible
        if self.max() + n > np.iinfo(self.dtype).max:
            out = self.astype(best_dtype(self.max() + n))
            out[out>0] += n
            return out
        else:
            self[self>0] += n
            return self
    
    def increment_iter(self, axes):
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

    def optimize(self):
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
    
    def relabel(self):
        self.value[:] = skseg.relabel_sequential(self.value)[0]
        return self
    
    
    @dims_to_spatial_axes
    @record()
    def expand_labels(self, distance:int=1, *, dims=None) -> Label:
        """
        Expand areas of labels.

        Parameters
        ----------
        distance : int, optional
            The distance to expand, by default 1
        dims : int or str, optional
            Dimension of axes.

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
        plt.figure()
        plt.imshow(label2rgb(self.value, bg_label=0), **kwargs)
        return self
    
    def __truediv__(self, value):
        raise NotImplementedError("Cannot divide label. If you need to divide, convert it to np.ndarray.")
    
    