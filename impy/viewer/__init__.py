try:
    from .viewer import napariViewers
    gui = napariViewers()
except ImportError:
    gui = None