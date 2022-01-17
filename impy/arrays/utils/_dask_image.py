try:
    from dask_image import ndfilters as dafil
    from dask_image import ndmorph as damorph
    from dask_image import ndinterp as daintr
    from dask_image import ndmeasure as dames
except ImportError:
    class NotInstalled:
        def __init__(self, msg):
            self.msg = msg
        
        def __getattr__(self, key: str):
            return self.__class__(self.msg)
        
        def __call__(self, *args, **kwargs):
            raise ModuleNotFoundError(self.msg)
        
    dafil = damorph = daintr = dames = NotInstalled("dask-image is not installed.")

