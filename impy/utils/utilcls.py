import time
from importlib import import_module

__all__ = ["ImportOnRequest"]


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
