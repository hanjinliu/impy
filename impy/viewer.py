import napari
from .imgarray import ImgArray
from .specials import MarkerArray, IndexArray

viewer = None

"""
viewer = napari.Viewer()
viewer.add_image(img, name="raw image")
viewer.add_image(img2, name="DOG filter")

# marker
viewer.layers[-1].name="peaks"
a=viewer.layers[-1]
a.symbol="arrow"
a.size = 0.2

 
# labeling
viewer.add_labels(img.labels)

# tracking
https://napari.org/tutorials/applications/cell_tracking.html
"""


class napariWindow:
    def __init__(self):
        self.viewer = None
    
    def add_image(self, img:ImgArray):
        self.viewer.add_image(img,
                              channel_axis=img.axisof("c"),
                              scale=[img.scale[a] for a in "zyx"],
                              name=["Tub","Kmut","ratio"])
        if hasattr(img, "labels"):
            self.viewer.add_labels(img.labels)
        return None
    