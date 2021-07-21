import time
from ._const import Const
from importlib import import_module

__all__ = ["Timer", "ImportOnRequest", "Progress"]

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