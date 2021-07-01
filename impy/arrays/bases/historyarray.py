from __future__ import annotations
from ...func import *
from ...utilcls import *
from .metaarray import MetaArray
import itertools

class HistoryArray(MetaArray):
    def __new__(cls, obj, name=None, axes=None, dirpath=None, 
                history=None, metadata=None, dtype=None):
        
        self = super().__new__(cls, obj, name, axes, dirpath, metadata, dtype)
        self.history = [] if history is None else history
        return self
    
    def __repr__(self):
        return f"\n"\
               f"    shape     : {self.shape_info}\n"\
               f"    dtype     : {self.dtype}\n"\
               f"  directory   : {self.dirpath}\n"\
               f"original image: {self.name}\n"\
               f"   history    : {'->'.join(self.history)}\n"
    
    
    def _getitem_additional_set_info(self, other, **kwargs):
        keystr = kwargs["keystr"]
        self._set_info(other, f"getitem[{keystr}]", kwargs["new_axes"])
        return None
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)  # set item as np.ndarray
        keystr = key_repr(key)           # write down key e.g. "0,*,*"
        new_history = f"setitem[{keystr}]"
        
        self._set_info(self, new_history)
        
    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.history = getattr(obj, "history", [])
    
    def _inherit_meta(self, obj, ufunc, **kwargs):
        """
        Copy axis, history etc. from obj.
        This is called in __array_ufunc__(). Unlike _set_info(), keyword `axis` must be
        considered because it changes `ndim`.
        """
        if "axis" in kwargs.keys() and not obj.axes.is_none():
            new_axes = del_axis(obj.axes, axis = kwargs["axis"])
        else:
            new_axes = "inherit"
        self._set_info(obj, ufunc.__name__, new_axes=new_axes)
        return self
    
    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        super()._set_info(other, new_axes)
        
        # set history
        if next_history is not None:
            self.history = other.history + [next_history]
        else:
            self.history = other.history.copy()
        
        return None
    
    def split(self, axis=None) -> list:
        """
        Split n-dimensional image into (n-1)-dimensional images.

        Parameters
        ----------
        axis : str or int, optional
            Along which axis the original image will be split, by default "c"

        Returns
        -------
        list of arrays
            Separate images
        """
        # determine axis in int.
        if axis is None:
            axis = find_first_appeared(self.axes, "cztp")
        axisint = self.axisof(axis)
            
        imgs = list(np.moveaxis(self, axisint, 0))
        for i, img in enumerate(imgs):
            img.history[-1] = f"split({axis}={i})"
            img.axes = del_axis(self.axes, axisint)
            img.set_scale(self)
            
        return imgs

    def tile(self, shape:tuple[int, int]|None=None, along:str|None=None, order:str|None=None):
        """
        Tile images in a certain order.

        Parameters
        ----------
        shape : tuple[int, int], optional
            Grid shape. This parameter must be specified unless the length of `along` is 2.
        along : str, optional
            Axis (Axes) over which will be iterated.
        order : str, {"r", "c"}, optional
            Order of iteration. "r" means row-wise and "c" means column-wise.
        
            row-wise
                ----->
                ----->
                ----->
            
            column-wise
                | | |
                | | |
                v v v

        Returns
        -------
        HistoryArray
            Tiled array
        """        
        if along is None:
            for a in self.axes:
                l = np.prod(shape)
                if self.sizeof(a) == l:
                    along = a
                    break
            else:
                raise ValueError(f"Could not find axis that can be reshaped to shape {shape}")
        elif len(along) == 2:
            uyaxis, uxaxis = self.axisof(along[0]), self.axisof(along[1])
            if uyaxis < uxaxis:
                shape = self.sizesof(along)
                order = "r"
            else:
                order = "c"
                shape = self.sizesof(along[::-1])
        elif len(along) == 1:
            if shape is None:
                raise ValueError("`shape` must be specified unless the length of `along` is 2.")
        else:
            raise ValueError("`along` must be a string with length 1 or 2.")
        
        if order is None:
            order = "r"
            
        uy_max, ux_max = shape
        imgy, imgx = self.sizesof("yx")
        if len(shape) == 2:
            c_axes = complement_axes("yx"+along, self.axes)
            new_axes = c_axes + "yx"
            outshape = self.sizesof(c_axes) + (uy_max*imgy, ux_max*imgx)
        else:
            raise ValueError("Shape mismatch")
        
        out = np.zeros(outshape, dtype=self.dtype)
        
        if order == "r":
            iter_tile = _iter_tile_yx
        elif order == "c":
            iter_tile = _iter_tile_xy
        else:
            raise ValueError(f"Could not interpret order={repr(order)}.")
        
        for (_, img), sl in zip(self.iter(along), iter_tile(uy_max, ux_max, imgy, imgx)):
            out[sl] = img
            
        out = out.view(self.__class__)
        out._set_info(self, next_history=f"tile({shape}, along={repr(along)}, order={order})", new_axes=new_axes)
        return out
        
def _iter_tile_yx(ymax, xmax, imgy, imgx):
    """
    +--+--+--+
    | 0| 1| 2|
    +--+--+--+
    | 3| 4| 5|
    +--+--+--+
    | 6| 7|..|
    +--+--+--+
    """    
    for uy, ux in itertools.product(range(ymax), range(xmax)):
        sly = slice(uy*imgy, (uy+1)*imgy, None)
        slx = slice(ux*imgx, (ux+1)*imgx, None)
        yield ..., sly, slx

def _iter_tile_xy(ymax, xmax, imgy, imgx):
    """
    +--+--+--+
    | 0| 3| 6|
    +--+--+--+
    | 1| 4| 7|
    +--+--+--+
    | 2| 5|..|
    +--+--+--+
    """    
    for uy, ux in itertools.product(range(xmax), range(ymax)):
        sly = slice(uy*imgy, (uy+1)*imgy, None)
        slx = slice(ux*imgx, (ux+1)*imgx, None)
        yield ..., slx, sly