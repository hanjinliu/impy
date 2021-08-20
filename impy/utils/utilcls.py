import time
import sys
from importlib import import_module
import threading
from .._const import Const

__all__ = ["Timer", "ImportOnRequest", "Progress"]

class Timer:
    def __init__(self):
        self.tic()
        
    def tic(self):
        self.t = time.time()
    
    @property
    def now(self):
        return time.time() - self.t
    
    def toc(self):
        self.t = self.now
    
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
    def __init__(self, name, out=sys.stdout):
        self.name = name
        self.timer = None
        if out is None:
            self.out = DummyOut()
        elif out == "stdout":
            self.out = sys.stdout
        else:
            self.out = out
        
    def update(self):
        event = self.stop_event
        chars = r"-\|/"
        i = 0
        while not event.is_set():
            event.wait(0.25)
            self.out.write(f"\b{chars[i]}")
            self.out.flush()
            i = i + 1
            if i >= 4:
                i = 0
    
    def __enter__(self):
        self.__class__.n_ongoing += 1
        if Const["SHOW_PROGRESS"] and self.__class__.n_ongoing == 1:
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self.update)
            self.thread.start()
            self.out.write(f"{self.name} -")
            self.out.flush()
            self.timer = Timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.__class__.n_ongoing -= 1
        if Const["SHOW_PROGRESS"] and self.__class__.n_ongoing == 0:
            self.timer.toc()
            self.stop_event.set()
            self.thread.join()
            self.out.write(f"\r{self.name} finished ({self.timer})\n")
            self.out.flush()

class DummyOut:
    def write(self, a):
        pass
    
    def flush(self):
        pass