__version__ = "1.5.1"
# TODO: 
# - btrack ... https://github.com/quantumjot/BayesianTracker
# - directional median filter to denoinse images
# - Melted -> pd.DataFrame
# - window or padding in lucy()

import warnings
from .imgarray import (array, zeros, zeros_like, empty, empty_like, 
                       imread, imread_collection, read_meta, 
                       stack, set_cpu, ImgArray)
from .specials import PropArray, MarkerArray

try:
    from .viewer import window
except ImportError as e:
    print(f"Could not import viewer: {e}")
except Exception as e:
    print(f"Could not import viewer: {e}")

r"""
Inheritance
-----------
              ____    MetaArray   _______
             /              \            \
       HistoryArray        MarkerArray  PropArray
       /         \          /       \
 LabeledArray   Label  IndexArray MeltedMarkerArray
    /
ImgArray

"""

# To silence Warnings in skimage
warnings.resetwarnings()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

