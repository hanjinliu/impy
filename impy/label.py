from .func import *
from .utilcls import *
from .historyarray import HistoryArray
from skimage.color import label2rgb
from skimage import segmentation as skseg

def best_dtype(n:int):
    if n < 256:
        return "uint8"
    elif n < 65536:
        return "uint16"
    else:
        return "uint32"

class Label(HistoryArray):
    def increment(self, n:int):
        # return view if possible
        if self.max() + n > np.iinfo(self.dtype).max:
            out = self.astype(best_dtype(self.max() + n))
            out[out>0] += n
            return out
        else:
            self[self>0] += n
            return self

    def optimize(self):
        m = self.max()
        if m < 256 and np.iinfo(self.dtype).max >= 256:
            return self.astype("uint8")
        elif m < 65536 and np.iinfo(self.dtype).max >= 65535:
            return self.astype("uint16")
        else:
            return self
    
    def relabel(self):
        self[:] = skseg.relabel_sequential(self.value)
        return self
    
    def imshow(self, **kwargs):
        plt.figure()
        plt.imshow(label2rgb(self, bg_label=0), **kwargs)
        return self
    
    def __truediv__(self, value):
        raise NotImplementedError("Cannot divide label. If you need to divide, convert it to np.ndarray.")