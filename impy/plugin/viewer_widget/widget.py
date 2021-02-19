import ipywidgets as wds
import numpy as np

class Widget:
    def __init__(self) -> None:
        self.axis_list = []
        self.range_list = []
        self.dict = {}
        self.sl = ()
        self.n_range = 0

    def addone(self, img, axis):
        len_a = img.sizeof(axis)
        self.axis_list.append(axis)
        if (axis == "z"):
            value0 = (len_a + 1)//2
        else:
            value0 = 1
        self.dict[axis] = wds.IntSlider(min=1, max=len_a, step=1, value=value0)
        self.sl += (value0 - 1,)
        return None
    
    def add(self, img, axes):
        for axis in axes:
            if (axis in img.axes):
                self.addone(img, axis)
        return None
    
    def add_range(self, img):
        kw = dict(step=1, orientation="horizontal", layout=wds.Layout(width="500px"))
        name = f"range_{self.n_range + 1}"
        self.n_range += 1
        value0 = [np.percentile(img[img>0], 0.01), np.percentile(img[img>0], 99.99)]
        if (value0[0] == 1 and value0[1] == 1):
            value0[0] = 0
        slider = wds.IntRangeSlider(min=np.min(img), max=np.max(img), value=value0,
                                    description=name + ": ", **kw)
        self.dict[name] = slider
        self.range_list.append(name)
                
        return value0

    def add_threshold(self, img, name="Threshold"):
        value0 = np.median(img)
        slider = wds.IntSlider(min=0, max=img.max(), step=1, value=value0, layout=wds.Layout(width="500px"))
        self.dict[name] = slider
        
        return value0
    
    def add_roi_check(self, name="Show ROI"):
        cbox = wds.Checkbox(False, description=name)
        self.dict[name] = cbox
        return None
    
    def interact(self, func):
        return wds.interact(func, **self.dict)

