import numpy as np
from skimage import measure as skmes
from skimage import draw as skdraw

class ROI:
    def __as_roi__(self, img):
        """
        Natural definition of ROI.
        Defines img[self].
        """
        
    def mask_array(self, mask):
        """
        Mask the corresponding pixels to draw ROI.
        """

class Line(ROI):
    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
    
    def __repr__(self):
        return f"Line({self.x0:.1f}:{self.x1:.1f}/{self.y0:.1f}:{self.y1:.1f})"
    
    def __as_roi__(self, img):
        return self.scan(img)

    def scan(self, img):
        if (img.ndim != 2):
            raise ValueError("Cannot make line scan from an image stack")
        start = [self.y0, self.x0]
        end = [self.y1, self.x1]
        scan = skmes.profile_line(img, start, end, mode="reflect")
        return scan
    
    def mean(self, img):
        scan = self.scan(img)
        return np.mean(scan)
    
    def std(self, img):
        scan = self.scan(img)
        return np.std(scan)
    
    def length(self):
        return np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)
    
    def mask_array(self, mask):
        rows, cols = skdraw.line(int(self.y0+0.5), int(self.x0+0.5), int(self.y1+0.5), int(self.x1+0.5))
        mask[rows, cols] = True
        return None

class Curve(ROI):
    def __init__(self, verts, tolerance=1):
        approx_verts = skmes.approximate_polygon(np.array(verts), tolerance)
        self.verts = approx_verts
    
    def __repr__(self):
        x0, y0 = self.verts[0]
        xf, yf = self.verts[-1]
        return f"Curve([{x0:.1f},{y0:.1f}]-[{xf:.1f},{yf:.1f}])"

    def __len__(self):
        return self.verts.shape[0]
    
    def __as_roi__(self, img):
        return self.scan(img)

    def mean(self, img):
        scan = self.scan(img)
        return np.mean(scan)
    
    def std(self, img):
        scan = self.scan(img)
        return np.std(scan)

    def length(self):
        xs = self.verts[:,0]
        ys = self.verts[:,1]
        dxs = np.diff(xs)
        dys = np.diff(ys)
        return np.sum(np.sqrt(dxs**2 + dys**2))
            
    def scan(self, img):
        if (img.ndim != 2):
            raise ValueError("Cannot make line scan from an image stack")
        scan = []
        for i in range(len(self) - 1):
            start = [self.verts[i, 1], self.verts[i, 0]]
            end = [self.verts[i+1, 1], self.verts[i+1, 0]]
            scan += list(skmes.profile_line(img, start, end, mode="reflect"))
        return np.array(scan)

    def mask_array(self, mask):
        for i in range(len(self)-1):
            x0 = self.verts[i, 0]
            y0 = self.verts[i, 1]
            x1 = self.verts[i+1, 0]
            y1 = self.verts[i+1, 1]
            rows, cols = skdraw.line(int(y0+0.5), int(x0+0.5), int(y1+0.5), int(x1+0.5))
            mask[rows, cols] = True
        return None

class Rectangle(ROI):
    def __init__(self, x0, x1, y0, y1):
        self.x0 = int(x0 + 0.5)
        self.x1 = int(x1 + 0.5)
        self.y0 = int(y0 + 0.5)
        self.y1 = int(y1 + 0.5)

    def __repr__(self):
        return f"Rectangle({self.x0}:{self.x1}/{self.y0}:{self.y1})"
    
    def __as_roi__(self, img):
        sl = (slice(None), ) * (img.ndim - 2) + (slice(self.y0, self.y1), slice(self.x0, self.x1))
        return img[sl]

    def area(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)
    
    def mean(self, img):
        return np.mean(img[self])
    
    def std(self, img):
        return np.std(img[self])
    
    def max(self, img):
        return np.max(img[self])
    
    def min(self, img):
        return np.min(img[self])
    
    def mask_array(self, mask):
        rows, cols = skdraw.polygon_perimeter([self.y0-1, self.y0-1, self.y1, self.y1], 
                                              [self.x0-1, self.x1, self.x1, self.x0-1])
        mask[rows, cols] = True
        return None

class Polygon(ROI):
    def __init__(self, verts):
        """
        verts = [[x0, y0], [x1, y1], ...]
        """
        self.verts = np.array(verts)
    
    def __len__(self):
        return self.verts.shape[0]
    
    def __repr__(self):
        return f"Polygon({len(self)} points)"
    
    def __as_roi__(self, img):
        arr_b = skmes.grid_points_in_poly(img.xyshape(), self.verts)
        return img[arr_b.T]
    
    def area(self):
        n_px = int(np.max(self.verts))
        return np.sum(skmes.grid_points_in_poly((n_px, n_px), self.verts))
    
    def mean(self, img):
        return np.mean(img[self])
    
    def std(self, img):
        return np.std(img[self])
    
    def max(self, img):
        return np.max(img[self])
    
    def min(self, img):
        return np.min(img[self])
    
    def mask_array(self, mask):
        xs = np.array([int(x+0.5) for x in self.verts[:, 1]])
        ys = np.array([int(y+0.5) for y in self.verts[:, 0]])
        rows, cols = skdraw.polygon_perimeter(xs, ys, mask.shape)
        mask[rows, cols] = True
        return None