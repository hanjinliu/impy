import time
from ._const import Const
from importlib import import_module

__all__ = ["ArrayDict", "FrameDict", "Timer", "ImportOnRequest", "Progress"]

class BaseDict(dict):
    def __init__(self, d=None, **kwargs):
        if isinstance(d, dict):
            kwargs = d
        
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"cannot set '{k}' because it conflicts with builtin methods.")
            else:
                super().__setattr__(k, v)
        super().__init__(**kwargs)
    
    def __setattr__(self, k, v):
        if hasattr(self, k):
            self.__setitem__(k,v)
        else:
            raise AttributeError(f"{self.__class__} has no attribute {k}")
    
    def __repr__(self):
        if self.keys():
            return self.__keys_repr__()
        else:
            return self.__class__.__name__ + "()"
    
    def __keys_repr__(self):
        pass
        
            
class ArrayDict(BaseDict):
    def __keys_repr__(self):
        maxl = max(len(aa) for aa in self.keys())
        out = f"{self.__class__.__name__} with\n"
        for k, v in self.items():
            out += f"{k:>{maxl}} : {v.shape_info}\n"
        return out

class FrameDict(BaseDict):
    def __keys_repr__(self):
        maxl = max(len(aa) for aa in self.keys())
        out = f"{self.__class__.__name__} with\n"
        for k, v in self.items():
            if hasattr(v, "col_axes"):
                caxes = v.col_axes
                out += f"[{k:>{maxl}} : {v.__class__.__name__} with {len(v)} rows x {len(caxes)} columns ({caxes})]\n"
            else:
                out += f"[{k:>{maxl}} : {v.__class__.__name__} with {len(v)} rows x {len(v.columns)} columns]\n"
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
        sec = round(sec, 2)
        if minute == 0:
            out = f"{sec} sec"
        else:
            out = f"{int(minute)} min {sec} sec"
        return out

class ImportOnRequest:
    def __init__(self, name:str):
        self.name = name
    
    def __getattr__(self, name:str):
        try:
            mod = super().__getattribute__("mod")
        except AttributeError:
            self.mod = import_module(self.name)
            mod = super().__getattribute__("mod")
        return getattr(mod, name)

class Progress:
    n_ongoing = 0
    def __init__(self, name):
        self.name = name
        self.timer = None
    
    def __enter__(self):
        self.__class__.n_ongoing += 1
        if Const["SHOW_PROGRESS"] and self.__class__.n_ongoing == 1:
            print(f"{self.name} ... ", end="")
            self.timer = Timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.__class__.n_ongoing -= 1
        if Const["SHOW_PROGRESS"] and self.__class__.n_ongoing == 0:
            self.timer.toc()
            print(f"\r{self.name} finished ({self.timer})")