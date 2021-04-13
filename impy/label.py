from .func import *
from .utilcls import *
from .historyarray import HistoryArray
from skimage.color import label2rgb

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
    
    def imshow(self, **kwargs):
        plt.figure()
        plt.imshow(label2rgb(self, bg_label=0), **kwargs)
        return self
        