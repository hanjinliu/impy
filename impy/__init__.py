__version__ = "1.5.3"
# TODO: 
# - directional median filter to denoinse images
# - window or padding in lucy()
# maybe useful functions in trackpy
# - tp.refine.refine_com, tp.refine.refine_com_arr
# - tp.find.where_close
# - tp.uncertainty.measure_noise 

import warnings
from .imgarray import (array, zeros, zeros_like, empty, empty_like, 
                       imread, imread_collection, read_meta, 
                       stack, set_cpu, ImgArray)
from .specials import PropArray

try:
    from .viewer import window
except ImportError as e:
    print(f"Could not import viewer: {e}")
except Exception as e:
    print(f"Could not import viewer: {e}")

r"""
Inheritance
-----------
              __ MetaArray _
             /              \ 
       HistoryArray    PropArray
       /         \    
 LabeledArray   Label  
    /
ImgArray

"""

# To silence Warnings in skimage
warnings.resetwarnings()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

