import numpy as np
from matplotlib.widgets import RectangleSelector, PolygonSelector, LassoSelector
import pandas as pd
from ...roi import Line, Curve, Rectangle, Polygon

class LineSelector(RectangleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xlim = sorted(self.ax.get_xlim())
        ylim = sorted(self.ax.get_ylim())

        x0 = min(max(xlim[0], x0), xlim[1])
        x1 = min(max(xlim[0], x1), xlim[1])
        y0 = min(max(ylim[0], y0), ylim[1])
        y1 = min(max(ylim[0], y1), ylim[1])
        
        self.to_draw.set_data([x0, x1], [y0, y1])
    
    @property
    def extents(self):
        """Return (xmin, xmax, ymin, ymax)."""
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        # Update displayed shape
        self.draw_shape(extents)
        # Update displayed handles
        self._corner_handles.set_data(*self.corners)
        self._edge_handles.set_data(*self.edge_centers)
        self._center_handle.set_data(*self.center)
        self.set_visible(self.visible)
        self.update()
        self.real_extents = extents # <- overloaded line.

class RectangleSelector2(RectangleSelector):
    def __init__(self, *args, size = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        
    def get_rect(self, x, y, xmax, ymax):
        """
        x, y -> rectangle positions
        """
        x = int(x + 0.5)
        y = int(y + 0.5)

        if (self.size//2 <= x and x < xmax - self.size//2 - 1):
            x0 = x - self.size//2
            x1 = x + (self.size + 1)//2
        elif (x < self.size//2):
            x0 = 0
            x1 = self.size
        else:
            x0 = xmax - self.size
            x1 = xmax
        
        if (self.size//2 <= y and y < ymax - self.size//2 - 1):
            y0 = y - self.size//2
            y1 = y + (self.size + 1)//2
        elif (y < self.size//2):
            y0 = 0
            y1 = self.size
        else:
            y0 = ymax - self.size
            y1 = ymax

        return x0, x1, y0 ,y1



class Drawer:
    def __init__(self, img2d, ax, im, tab=None):
        self.img = img2d
        self.ax = ax
        self.im = im
        self.tab = tab # if show result table in real time, set tab to plt.axes object.
        self.rois = []
        self._measures = []
    
    @property
    def measures(self):
        df = pd.DataFrame(data=self._measures, columns=self.columns)
        return df
    
    
    def onselect(self, eclick, erelease):
        """
        What would happen when the figure is clicked.
        """

    def draw(self):
        """
        Make a masked array to show ROIs.
        """
        mask = np.zeros(self.img.shape, dtype=bool)
        for r in self.rois:
            r.mask_array(mask)
        return np.ma.masked_equal(mask, False)
    
    def plot_table(self, ax):
        """
        Show measurement results in table.
        Last 12 results are shown.
        """
        df = self.measures
        celltext = np.round(np.array(df), 2)
        if (celltext.shape[0] > 10):
            celltext = celltext[-10:]
        ax.table(cellText=celltext, colLabels=df.columns, loc="upper center",
                 colColours=["skyblue"]*len(self.columns))
        return None
    
    def set_data(self):
        self.im.set_data(self.draw())
        return None
    
    def disconnect(self):
        self.selector.disconnect()


class RectangleDrawer(Drawer):
    """
    - Attributes
    rois = list of rectangle ROIs (Rectangle objects).
    measures = pd.DataFrame of measurement.
    """
    def __init__(self, img2d, ax, im, tab=None, size=None):
        super().__init__(img2d, ax, im, tab)
        rectprops = dict(facecolor="red", edgecolor = "red", alpha=0.6, fill=False)
        self.fixed_shape = (not size is None)
        
        if (size is None):
            self.selector = RectangleSelector(self.ax, self.onselect, drawtype="box", 
                                minspanx=0.01, minspany=0.01, rectprops=rectprops, button=1)
        else:
            self.selector = RectangleSelector2(self.ax, self.onselect, drawtype="none",
                                rectprops=rectprops, button=1, size=size)
        
        self.columns = ["area", "mean", "std", "max", "min"]

    def onselect(self, eclick, erelease):
        if (self.fixed_shape):
            rect = Rectangle(*(self.selector.get_rect(erelease.xdata, erelease.ydata, self.img.shape[1], self.img.shape[0])))
        else:
            x0, x1, y0 ,y1 = self.selector.extents
            rect = Rectangle(x0+1, x1, y0+1, y1)
        self.rois.append(rect)
        self._measures.append([rect.area(), rect.mean(self.img), rect.std(self.img), 
                               rect.max(self.img), rect.min(self.img)])
        
        self.set_data()
        if (self.tab):
            self.plot_table(self.tab)
    
    
    

class PolygonDrawer(Drawer):
    def __init__(self, img2d, ax, im, tab=None):
        super().__init__(img2d, ax, im, tab)
        lineprops = dict(color = "red", alpha=0.6, linewidth=0.3)
        markerprops = dict(marker=None)
        
        self.selector = PolygonSelector(self.ax, self.onselect, vertex_select_radius=5,
                            lineprops=lineprops, markerprops=markerprops)
        
        self.columns = ["area", "mean", "std", "max", "min"]

    def onselect(self, verts):
        self.s = self.selector.verts
        poly = Polygon(self.selector.verts)
        self.rois.append(poly)
        self._measures.append([poly.area(), poly.mean(self.img), poly.std(self.img), 
                               poly.max(self.img), poly.min(self.img)])
        
        self.set_data()
        if (self.tab):
            self.plot_table(self.tab)
        
        # reset polygon state without pushing Esc key.
        if(self.selector._polygon_completed):
            self.selector._xs, self.selector._ys = [0], [0]
            self.selector._polygon_completed = False
            self.selector.set_visible(True)
    


class LineDrawer(Drawer):
    """
    - Attributes
    rois = list of line ROIs (Line objects).
    measures = pd.DataFrame of measurement.
    scans = list of line-scanned profiles.
    """
    def __init__(self, img2d, ax, im, tab=None, scan=None):
        super().__init__(img2d, ax, im, tab)
        lineprops = dict(color="red", linestyle='-', linewidth=0.3, alpha=0.6)
        self.selector = LineSelector(self.ax, self.onselect, drawtype="line", minspanx=0.01, minspany=0.01,
                                     lineprops=lineprops, button=1)
        self.scan = scan
        self.columns = ["length", "mean", "std"]
        self.scans = []
    

    def onselect(self, eclick, erelease):
        line = Line(*(self.selector.real_extents))
        self.rois.append(line)
        self._measures.append([line.length(), line.mean(self.img), line.std(self.img)])
        scan = line.scan(self.img)
        self.scans.append(scan)
        
        self.set_data()
        if (self.tab):
            self.plot_table(self.tab)
        if (self.scan):
            self.scan.cla()
            self.scan.set_title("Line Scan")
            self.scan.grid(lw=0.3)
            self.scan.plot(np.arange(len(scan)), scan, color="red", lw=0.3)
    
    
class CurveDrawer(Drawer):
    def __init__(self, img2d, ax, im, tab=None, scan=None):
        super().__init__(img2d, ax, im, tab)
        lineprops = dict(color="red", linestyle='-', linewidth=0.3, alpha=0.6)
        self.selector = LassoSelector(self.ax, self.onselect, button=1, lineprops=lineprops)
        self.scan = scan
        self.columns = ["length", "mean", "std"]
        self.scans = []
    
    def onselect(self, verts):
        curve = Curve(self.selector.verts)
        self.rois.append(curve)
        self._measures.append([curve.length(), curve.mean(self.img), curve.std(self.img)])
        scan = curve.scan(self.img)
        self.scans.append(scan)
        
        self.set_data()
        if (self.tab):
            self.plot_table(self.tab)
        if (self.scan):
            self.scan.cla()
            self.scan.set_title("Line Scan")
            self.scan.grid(lw=0.3)
            self.scan.plot(np.arange(len(scan)), scan, color="red", lw=0.3)


class RectangleCropper(Drawer):
    def __init__(self, img2d, ax, im, img_total=None):
        super().__init__(img2d, ax, im)
        rectprops = dict(facecolor="pink", edgecolor = "pink", alpha=0.6, fill=False)        
        self.selector = RectangleSelector(self.ax, self.onselect, drawtype="box", 
                            minspanx=0.01, minspany=0.01, rectprops=rectprops, button=1)
        self.imgs = []
        self.img_total = img_total
    
    def onselect(self, eclick, erelease):
        rect = Rectangle(*(self.selector.extents))
        self.rois.append(rect)
        self.imgs.append(self.img_total[rect]) # append cropped image
        self.im.set_data(self.draw())
    