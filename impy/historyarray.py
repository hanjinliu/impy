from .func import *
from .utilcls import *
from .metaarray import MetaArray


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
    
    # def __getitem__(self, key):
    #     if isinstance(key, str):
    #         # img["t=2;z=4"] ... ImageJ-like method
    #         sl = self.str_to_slice(key)
    #         return self.__getitem__(sl)

    #     if isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
    #         # img[arr] ... where arr is 2-D boolean array
    #         key = add_axes(self.axes, self.shape, key)

    #     out = np.ndarray.__getitem__(self, key) # get item as np.ndarray
    #     keystr = key_repr(key)                 # write down key e.g. "0,*,*"
    #     if isinstance(out, self.__class__):   # cannot set attribution to such as numpy.int32 
    #         if hasattr(key, "__array__"):
    #             # fancy indexing will lose axes information
    #             new_axes = None
                
    #         elif "new" in keystr:
    #             # np.newaxis or None will add dimension
    #             new_axes = None
                
    #         elif self.axes:
    #             del_list = [i for i, s in enumerate(keystr.split(",")) if s != "*"]
    #             new_axes = del_axis(self.axes, del_list)
    #         else:
    #             new_axes = None
            
    #         out._getitem_additional_set_info(self, next_history=f"getitem[{keystr}]",
    #                                          new_axes=new_axes)
            
    #     # TODO: Ellipsis does not work now.
    #     return out
    
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
            axis = kwargs["axis"]
            new_axes = del_axis(obj.axes, axis)
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
            img.history[-1] = f"axis({axis})={i}"
            img.axes = del_axis(self.axes, axisint)
            
        return imgs

    