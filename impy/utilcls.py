import time
import numpy as np

class ArrayDict(dict):
    def __getattr__(self, name:str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __repr__(self):
        if self.keys():
            v0 = next(iter(self.values()))
            return f"keys: {', '.join(self.keys())}\n" \
                   f"values: {repr(v0)}      etc."
        else:
            return self.__class__.__name__ + "()"


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