try:
    from dask_image import ndfilters as dafil
    from dask_image import ndmorph as damorph
    from dask_image import ndinterp as daintr
    from dask_image import ndmeasure as dames
except ImportError:
    dafil = damorph = daintr = dames = None

