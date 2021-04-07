NONE = "_"
ORDER = {"q": -10, "s": 5, "p": -1, "t": 0, "z": 1, "c": 2, "y": 3, "x": 4}

class ImageAxesError(Exception):
    pass

def sort_axes(str_):
    dict_ = {}
    for s in str_:
        dict_[ORDER[s]] = s
    return "".join([dict_[k] for k in sorted(dict_.keys())])

def check_none(func):
    def checked(self, *args, **kwargs):
        if self.is_none():
            raise ImageAxesError("Axes not defined.")
        return func(self, *args, **kwargs)
    return checked


class Axes:
    def __init__(self, value=NONE, ndim=0) -> None:
        self.axes = NONE
        
        if value == NONE:
            pass
        elif isinstance(value, str):
            value = value.lower()
            counter = {"p":False, "t": False, "z": False, "c": False, "x": False, "y": False}
            for v in value:
                if v in "ptzcxys":
                    if (counter[v] == True):
                        raise ImageAxesError(f"'{v}' appeared twice: {value}")
                    counter[v] = True
                elif v in "q":
                    pass
                else:
                    raise ImageAxesError(f"axes cannot contain characters except for 'qtzcxys': got {value}")
            
            if ndim > 0 and len(value) != ndim:
                raise ImageAxesError(f"Inconpatible dimensions: image (ndim={ndim}) and axes ({value})")
        elif isinstance(value, self.__class__):
            value = value.axes
        else:
            raise ImageAxesError(f"Cannot set {type(value)} to axes.")
        
        self.axes = value
            
    @check_none
    def __str__(self):
        return self.axes
    
    @check_none
    def __len__(self):
        return len(self.axes)

    @check_none
    def __getitem__(self, key):
        return self.axes[key]
    
    @check_none
    def __iter__(self):
        return self.axes.__iter__()
    
    @check_none
    def __next__(self):
        return self.axes.__next__()
    
    @check_none
    def __eq__(self, other):
        if isinstance(other, str):
            return self.axes == other
        elif isinstance(other, self.__class__):
            return other == self.axes

    def __contains__(self, other):
        return other in self.axes
    
    def __bool__(self):
        return not self.is_none()
    
    def __repr__(self):
        if self.is_none():
            return "No axes defined"
        else:
            return self.axes

    def is_none(self):
        return self.axes == NONE

    def to_none(self):
        self.axes = NONE
        return None
    
    @check_none
    def is_sorted(self) -> bool:
        return self.axes == sort_axes(self.axes)
    
    def check_is_sorted(self):
        if self.is_sorted():
            pass
        else:
            raise ImageAxesError(f"Axes must in tzcxy order, but got {self.axes}")
    
    @check_none
    def find(self, axis) -> int:
        i = self.axes.find(axis)
        if i < 0:
            raise ImageAxesError(f"Image does not have {axis}-axis: {self.axes}")
        else:
            return i
    

    @check_none
    def sort(self) -> None:
        self.axes = sort_axes(self.axes)
        return None
    
    @check_none
    def argsort(self):
        sortedaxes = sort_axes(self.axes)
        arglist = []
        for a in self.axes:
            arglist.append(sortedaxes.find(a))
        return arglist

    
    def copy(self):
        copy_ = self.__class__()
        copy_.axes = self.axes
        return copy_