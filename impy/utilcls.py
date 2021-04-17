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

class SpatialList(list):
    def __getattr__(self, name:str):
        dims = self.dims
        
        if name in dims:
            i = dims.index(name)
            return self[i]
        else:
            raise AttributeError(f"SpetialList only have {dims} dimensions.")
    
    def __repr__(self):
        return f"spatial dimensions: {self.dims}\n" \
               f"values: {repr(self[0])}      etc."
    
    @property
    def dims(self):
        if len(self) == 3:
            _dims = ["z", "y", "x"]
        elif len(self) == 2:
            _dims = ["y", "x"]
        else:
            raise ValueError("Incorrect structure.")
        return _dims