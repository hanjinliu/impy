from .metaarray import MetaArray
import numpy as np
import matplotlib.pyplot as plt
from .func import del_axis

SCALAR_PROP = (
    "area", "bbox_area", "convex_area", "eccentricity", "equivalent_diameter", "euler_number",
    "extent", "feret_diameter_max", "filled_area", "label", "major_axis_length", "max_intensity",
    "mean_intensity", "min_intensity", "minor_axis_length", "orientation", "perimeter",
    "perimeter_crofton", "solidity")

class PropArray(MetaArray):
    def plot_profile(self, along=None, cmap="jet", cmap_range=(0,1)):
        if along is None:
            along = self.axes[-1]
        
        iteraxes = del_axis(self.axes, self.axisof(along))
        plt.figure(figsize=(4, 1.7))
        cmap = plt.get_cmap(cmap)
        positions = np.linspace(*cmap_range, self.size//self.sizeof(along), endpoint=False)
        x = np.arange(self.sizeof(along))
        for i, (sl, y) in enumerate(self.iter(iteraxes)):
            plt.plot(x, y, color=cmap(positions[i]))
        
        plt.title(f"for each {iteraxes}")
        plt.xlabel(along)
        plt.show()
        
        return self
        
        