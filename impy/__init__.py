__version__ = "1.8.9"

# TODO
# - 3D Gabor filter
# - Colocalization ... https://note.com/sakulab/n/n0e2cf293cc1e#BGd2U

import warnings
from .core import (array, zeros, zeros_like, empty, empty_like, 
                   imread, imread_collection, read_meta, 
                   stack, set_cpu, set_verbose)
from .imgarray import ImgArray
from .specials import PropArray, MarkerFrame, TrackFrame
from .label import Label
from .phasearray import PhaseArray
from .viewer import window

r"""
Inheritance
-----------
              __ MetaArray _
             /              \ 
       HistoryArray     PropArray
       /         \    
  LabeledArray   Label  
    /     \
ImgArray PhaseArray

"""

# To silence Warnings in skimage
warnings.resetwarnings()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
