import dask
import psutil
from dask.cache import Cache

memory = psutil.virtual_memory()

MAX_GB_LIMIT = memory.total / 2 * 1e-9
DASK_CACHE_GB_LIMIT = memory.total / 4 * 1e-9

class GlobalConstant(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            super().__setattr__("_cache", Cache(1.0))
        except (ModuleNotFoundError, ImportError):
            super().__setattr__("_cache", None)

    def __getitem__(self, k):
        try:
            return super().__getitem__(k)
        except KeyError:
            raise KeyError(f"Global constants: {', '.join(self.keys())}")

    def __setitem__(self, k, v):
        if k == "MAX_GB":
            if not isinstance(v, (int, float)):
                raise TypeError("MAX_GB must be float.")
            elif v > MAX_GB_LIMIT:
                raise ValueError(f"Cannot exceed {MAX_GB_LIMIT} GB.")
        elif k == "SHOW_PROGRESS":
            if v in (0, 1):
                v = bool(v)
            elif not isinstance(v, bool):
                raise TypeError("SHOW_PROGRESS must be bool.")
        elif k == "ID_AXIS":
            if not isinstance(v, str):
                raise TypeError("ID_AXIS must be str.")
            elif len(v) != 1:
                raise ValueError("ID_AXIS must be single character.")
        elif k == "FONT_SIZE_FACTOR":
            if not isinstance(v, (int, float)):
                raise TypeError("MAX_GB must be float.")
        elif k == "RESOURCE":
            raise RuntimeError("Cannot set RESOURCE.")
        elif k == "DASK_CACHE_GB":
            if not isinstance(v, (int, float)):
                raise TypeError("DASK_CACHE_GB must be float.")
            elif v > DASK_CACHE_GB_LIMIT:
                raise ValueError(f"Cannot exceed {DASK_CACHE_GB_LIMIT} GB.")
            self._cache.cache.available_bytes = v
        elif k == "SCHEDULER":
            dask.config.set(scheduler=v)
        else:
            raise RuntimeError("Cannot set new keys.")
        
        super().__setitem__(k, v)

    def _setitem_(self, k, v):
        super().__setitem__(k, v)
    
    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def __delitem__(self, v):
        raise RuntimeError("Cannot delete any items.")
    
    def __repr__(self):
        return \
        f"""
              MAX_GB    : {self.MAX_GB:.2f} GB
          SHOW_PROGRESS : {self.SHOW_PROGRESS}
             ID_AXIS    : {self.ID_AXIS}
        FONT_SIZE_FACTOR: {self.FONT_SIZE_FACTOR}
             RESOURCE   : {self.RESOURCE}
          DASK_CACHE_GB : {self.DASK_CACHE_GB:.2f} GB
            SCHEDULER   : {self.SCHEDULER}
        """

Const = GlobalConstant(
    MAX_GB = MAX_GB_LIMIT/2,
    SHOW_PROGRESS = True,
    ID_AXIS = "p",
    FONT_SIZE_FACTOR = 1.0,
    RESOURCE = "numpy",
    DASK_CACHE_GB = DASK_CACHE_GB_LIMIT/2,
    SCHEDULER = "threads",
)

class SetConst:
    n_ongoing = 0
    def __init__(self, name=None, value=None, **kwargs):
        if name is None and value is None and len(kwargs) == 1:
            name, value = list(kwargs.items())[0]
        elif name in Const.keys() and value is not None:
            pass
        else:
            raise TypeError("Invalid input for SetConst")
        self.name = name
        self.value = value
    
    def __enter__(self):
        self.old_value = Const[self.name]
        Const[self.name] = self.value

    def __exit__(self, exc_type, exc_value, traceback):
        Const[self.name] = self.old_value