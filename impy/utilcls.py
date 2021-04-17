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
