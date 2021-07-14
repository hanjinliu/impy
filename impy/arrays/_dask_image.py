try:
    from dask_image import ndfilters as dafil
    from dask_image import ndmorph as damorph
    from dask_image import ndmeasure as dames
except ImportError:
    pass
