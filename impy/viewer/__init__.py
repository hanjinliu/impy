try:
    from .viewer import napariViewers
except ImportError:
    class napariViewer:
        def __getattr__(self, key):
            raise ModuleNotFoundError(
                "napari viewer could not be constructed due to ImportError."
            )

gui = napariViewers()
