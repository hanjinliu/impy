import time
import numpy as np

class BaseDict(dict):
    def __getattr__(self, name:str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __repr__(self):
        if self.keys():
            return self.__keys_repr__()
        else:
            return self.__class__.__name__ + "()"
    
    def __keys_repr__(self):
        pass
    
class ArrayDict(BaseDict):
    def __keys_repr__(self):
        out = f"{self.__class__.__name__} with\n"
        for k, v in self.items():
            out += f"{k} : {v.shape_info}\n"
        return out

class FrameDict(BaseDict):
    def __keys_repr__(self):
        out = f"{self.__class__.__name__} with\n"
        for k, v in self.items():
            out += f"{k} : {v.col_axes} x{len(v)} rows\n"
        return out

class Timer:
    def __init__(self):
        self.tic()
        
    def tic(self):
        self.t = time.time()
    
    def toc(self):
        self.t = time.time() - self.t
    
    def __str__(self):
        minute, sec = divmod(self.t, 60)
        sec = np.round(sec, 2)
        if minute == 0:
            out = f"{sec} sec"
        else:
            out = f"{int(minute)} min {sec} sec"
        return out