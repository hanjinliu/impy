import numpy as np
import matplotlib.pyplot as plt
from .viewer_widget.drawer import RectangleDrawer, PolygonDrawer, LineDrawer, CurveDrawer, RectangleCropper
from ..roi import ROI
from .viewer_widget.widget import Widget

__all__ = ["imshowc", "imshowz", "imshow_comparewith", 
           "threshold_manual",
           "measure_rectangles", "measure_polygons", "measure_lines", "measure",
           "crop_rectangles"]

def _imshow2d(self, ax, **kwargs):
    # show image
    im = ax.imshow(self, **kwargs)
    return im

def hist(self, ax, contrast=None, newfig=True):
    """
    Show intensity profile.
    """
    ax.cla()
    if (newfig):
        plt.figure(figsize=(4, 1.7))

    n_bin = min(int(np.sqrt(self.size / 3)), 100)
    hi = ax.hist(self.flat, color="grey", bins=n_bin, density=True)
    
    if (contrast is None):
        contrast = [self.min(), self.max()]
    x0, x1 = contrast
    
    ax.set_xlim(x0, x1)
    ax.set_yticks([])
    
    return hi

def imshow2d(self, ax, h, **kwargs):
    if (self.lut):
        cmap = self.lut[0]
    else:
        cmap = "gray"
    
    vmax = np.percentile(self[self>0], 99.99)
    vmin = np.percentile(self[self>0], 0.01)
    
    imshow_kwargs = {"cmap": cmap, "vmax": vmax, "vmin": vmin, "interpolation": "none"}
    imshow_kwargs.update(kwargs)

    # show image
    im = _imshow2d(self, ax, **imshow_kwargs)
    
    # show intensity profile
    hi = hist(self, h, contrast=[imshow_kwargs["vmin"], imshow_kwargs["vmax"]], newfig=False)
    
    return im, hi

def imshowc(self, **kwargs):
    """
    Show image in the most straight forward way.
    If self is multi-channel, then all the channel are displayed at the same time.
    t, z are set to common values with sliders.
    Contrasts can be adjusted separately.
    """
    
    self_ = self.as_uint16()

    wd = Widget()
    wd.add(self_, "tz")
    wd.add_roi_check()
    empty_mask = np.ma.masked_all(self_.xyshape())

    if ("c" in self_.axes):
        n_chn = min(self_.sizeof("c"), 4)
        fig, all_axs = subplots(n_chn)
        axs = all_axs[0]
        hs = all_axs[1]
        
        cmaps = self.get_cmaps()
        
        ims = [] # plt object, the image
        im_rois = [] # plt object, ROI mask
        his = [] # plt object, the histogram
        for c in range(n_chn):
            img0 = self_[wd.sl + (c,)]
            wd.add_range(img0)
            im, hi = imshow2d(img0, axs[c], hs[c], cmap=cmaps[c], **kwargs)
            im_roi = axs[c].imshow(empty_mask, cmap="hsv", interpolation="none")
            ims.append(im)
            im_rois.append(im_roi)
            his.append(hi)

        def func(**kw):
            sl = tuple(kw[a]-1 for a in wd.axis_list)
            
            for i, r in enumerate(wd.range_list):
                vmin, vmax = kw[r]
                # set new values to the existing imaging plane.
                ims[i].set_clim((vmin, vmax))
                ims[i].set_data(self_[sl + (i,)])

                # show ROIs
                if (hasattr(self, "rois")):
                    if (kw["Show ROI"]):
                        mask = empty_mask.copy()
                        for r in self.rois:
                            r.mask_array(mask)
                        im_rois[i].set_data(mask)
                    else:
                        im_rois[i].set_data(empty_mask)
                    
                
                # set new values to the existing histogram.
                update_hist(self_[sl + (i,)].flat, [vmin, vmax], hs[i], his[i])

            return None
        
    else:
        fig, (ax, h) = subplots(1)
        wd.add_range(self_)
        cmaps = self.get_cmaps()
        im, hi = imshow2d(self_[wd.sl], ax, h, cmap=cmaps[0], **kwargs)

        def func(**kw):
            sl = tuple(kw[a]-1 for a in wd.axis_list)
            vmin, vmax = kw[wd.range_list[0]]
            # set new values to the existing imaging plane.
            im.set_clim((vmin, vmax))
            im.set_data(self_[sl])
            
            # set new values to the existing histogram.
            update_hist(self_[sl].flat, [vmin, vmax], h, hi)

            return None
    
    wd.interact(func)
    return self_

def imshowz(self, zlist=[0, 1, 2, 3], **kwargs):
    """
    Show images in different z positions in parallel.
    """
    if (not "z" in self.axes):
        raise ValueError("Do not have z-axis")
    if (len(zlist) > 4):
        zlist = zlist[:4]

    
    sl = [slice(None)]
    
    if ("t" in self.axes):
        self_ = self[:, zlist].as_uint16()
        sl = [0] + sl
    else:
        self_ = self[zlist].as_uint16()
    if ("c" in self_.axes):
        sl = sl + [0]

    wd = Widget()
    wd.add(self_, "tc")
    wd.add_range(self_)
    
    fig, all_axs = subplots(len(zlist))
    axs = all_axs[0]
    hs = all_axs[1]
    
    wd.sl = tuple(sl)
    
    cmaps = self.get_cmaps()

    ims = [] # plt object, the image
    his = []
    for i in range(len(zlist)):
        im = _imshow2d(self_[wd.sl][i], axs[i], cmap=cmaps[0], **kwargs)
        ims.append(im)
        hi = hist(self_[wd.sl][i], hs[i], newfig=False)
        his.append(hi)
    

    def func(**kw):
        sl = [slice(None)]
        cmap = cmaps[0]
        for a in wd.axis_list:
            if (a == "t"):
                sl = [kw[a]-1] + sl
            elif (a == "c"):
                sl = sl + [kw[a]-1]
                cmap = cmaps[kw[a]-1]
            else:
                raise ValueError(f"Unknown axis: {a}")

        sl = tuple(sl)
        
        vmin, vmax = kw[wd.range_list[0]]
        
        for i, z in enumerate(zlist):
            # set new values to the existing imaging plane.
            ims[i].set_clim((vmin, vmax))
            ims[i].set_data(self_[sl][i])
            ims[i].set_cmap(cmap)
            axs[i].set_title(f"slice-{z}")
            # set new values to the existing histogram.
            update_hist(self_[sl][i].flat, [vmin, vmax], hs[i], his[i])

        return None

    wd.interact(func)
    return self_

def imshow_comparewith(self, other=None, titles=["image-1", "image-2"], **kwargs):
    """
    Compare two images (with same tzc dimensions).
    Show self on the left site, and show other on the right site.
    """
    self_ = self.as_uint16()
    
    if (other is None):
        other_ = self.as_uint16()
    else:
        other_ = other.as_uint16()

    if (self.axes != other.axes):
        raise ValueError("Cannot compare images with different dimension.")
    if (self.shape[:-2] != other.shape[:-2]):
        raise ValueError("Cannot compare images with different shape.")
    
    wd = Widget()
    wd.add(self_, "tzc")
    wd.add_range(self_)
    wd.add_range(other_)
    wd.sl = wd.sl

    cmaps = self.get_cmaps()
    
    fig, all_axs = plt.subplots(2, 2, figsize=(9, 4), gridspec_kw={"height_ratios": [3, 1]})
    axs = all_axs[0]
    hs = all_axs[1]

    ims = [] # plt object, the image
    his = []
    im = _imshow2d(self_[wd.sl], axs[0], cmap=cmaps[0], **kwargs)
    ims.append(im)
    hi = hist(self_[wd.sl], hs[0], newfig=False)
    his.append(hi)
    im = _imshow2d(other_[wd.sl], axs[1], cmap=cmaps[0], **kwargs)
    ims.append(im)
    hi = hist(other_[wd.sl], hs[1], newfig=False)
    his.append(hi)

    def func(**kw):
        sl = []
        cmap = cmaps[0]
        for a in wd.axis_list:
            sl.append(kw[a] - 1)
            if (a == "c"):
                cmap = cmaps[kw[a] - 1]
        sl = tuple(sl)
        
        # for image-1
        vmin, vmax = kw["range_1"]
        ims[0].set_clim((vmin, vmax))
        ims[0].set_data(self_[sl])
        ims[0].set_cmap(cmap)
        axs[0].set_title(titles[0])
        
        update_hist(self_[sl].flat, [vmin, vmax], hs[0], his[0])

        # for image-2
        vmin, vmax = kw["range_2"]
        ims[1].set_clim((vmin, vmax))
        ims[1].set_data(other_[sl])
        ims[1].set_cmap(cmap)
        axs[1].set_title(titles[1])
        
        update_hist(other_[sl].flat, [vmin, vmax], hs[1], his[1])
        
        return None
    
    wd.interact(func)
    return self_

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Thresholding and Labeling
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def threshold_manual(self, light_bg=False, **kwargs):
    self_ = self.as_uint16()
    wd = Widget()
    wd.add(self_, "tz")
    wd.add_range(self_)
    thr = wd.add_threshold(self)

    fig, all_axs = subplots(1)
    ax = all_axs[0]
    h = all_axs[1]

    cmaps = self.get_cmaps()
    
    im = _imshow2d(self_[wd.sl], ax, cmap=cmaps[0], interpolation="none", **kwargs)
    im_thr = ax.imshow(np.ma.masked_all(self_[wd.sl].shape), cmap="hsv", interpolation="none")
    hi = hist(self_[wd.sl], h, newfig=False)
    border, = h.plot([thr, thr], h.get_ylim(), color="red", lw=0.5)
    

    def func(**kw):
        sl = []
        cmap = cmaps[0]
        for a in wd.axis_list:
            if (a in "tz"):
                sl.append(kw[a]-1)
            elif (a == "c"):
                sl.append(kw[a]-1)
                cmap = cmaps[kw[a]-1]
            else:
                raise ValueError(f"Unknown axis: {a}")

        sl = tuple(sl)
        
        vmin, vmax = kw[wd.range_list[0]]
        thr = kw["Threshold"]
        
        # set new values to the existing imaging plane.
        im.set_clim((vmin, vmax))
        im.set_data(self_[sl])
        im.set_cmap(cmap)
        if (light_bg):
            mask = self_[sl] <= thr
        else:
            mask = self_[sl] >= thr
        im_thr.set_data(np.ma.masked_equal(mask, False))
        # set new values to the existing histogram.
        update_hist(self_[sl].flat, [vmin, vmax], h, hi)
        border.set_xdata([thr, thr])

        return None

    wd.interact(func)

    # connection problem
    self.temp = thr
    return self.threshold(thr, light_bg = light_bg)

def label(self, **kwargs):
    return 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Measurements
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def measure_rectangles(self, size=None, **kwargs):
    """
    Open a figure canvas and start recording rectangle ROIs.
    """
    self_ = self.as_uint16()
    wd = Widget()
    wd.add(self_, "tzc")
    wd.add_range(self_)

    
    fig, all_axs = subplots(2)
    ax = all_axs[0][0]
    h = all_axs[1][0]
    tab = all_axs[0][1]
    tab.axis("off")
    all_axs[1][1].axis("off")

    cmaps = self.get_cmaps()
    
    im = _imshow2d(self_[wd.sl], ax, cmap=cmaps[0], interpolation="none", **kwargs)
    im_roi = ax.imshow(np.ma.masked_all(self_[wd.sl].shape), cmap="hsv", interpolation="none")
    hi = hist(self_[wd.sl], h, newfig=False)
    
    rect = RectangleDrawer(self_[wd.sl], ax, im_roi, tab, size=size)
    if (hasattr(self, "rois") and type(self.rois) is list):
        rect.rois = self.rois
        rect.set_data()

    def func(**kw):
        sl = []
        cmap = cmaps[0]
        for a in wd.axis_list:
            if (a in "tz"):
                sl.append(kw[a]-1)
            elif (a == "c"):
                sl.append(kw[a]-1)
                cmap = cmaps[kw[a]-1]
            else:
                raise ValueError(f"Unknown axis: {a}")

        sl = tuple(sl)
        rect.img = self_[wd.sl]
        
        vmin, vmax = kw[wd.range_list[0]]
        
        # set new values to the existing imaging plane.
        im.set_clim((vmin, vmax))
        im.set_data(self_[sl])
        im.set_cmap(cmap)
        # set new values to the existing histogram.
        update_hist(self_[sl].flat, [vmin, vmax], h, hi)

        return None

    wd.interact(func)
    self.rois = rect.rois
    return rect

def measure_polygons(self, **kwargs):
    """
    Open a figure canvas and start recording rectangle ROIs.
    """
    self_ = self.as_uint16()
    wd = Widget()
    wd.add(self_, "tzc")
    wd.add_range(self_)

    
    fig, all_axs = subplots(2)
    ax = all_axs[0][0]
    h = all_axs[1][0]
    tab = all_axs[0][1]
    tab.axis("off")
    all_axs[1][1].axis("off")

    cmaps = self.get_cmaps()
    
    im = _imshow2d(self_[wd.sl], ax, cmap=cmaps[0], interpolation="none", **kwargs)
    im_roi = ax.imshow(np.ma.masked_all(self_[wd.sl].shape), cmap="hsv", interpolation="none")
    hi = hist(self_[wd.sl], h, newfig=False)
    
    poly = PolygonDrawer(self_[wd.sl], ax, im_roi, tab)
    if (hasattr(self, "rois") and type(self.rois) is list):
        poly.rois = self.rois
        poly.set_data()

    def func(**kw):
        sl = []
        cmap = cmaps[0]
        for a in wd.axis_list:
            if (a in "tz"):
                sl.append(kw[a]-1)
            elif (a == "c"):
                sl.append(kw[a]-1)
                cmap = cmaps[kw[a]-1]
            else:
                raise ValueError(f"Unknown axis: {a}")

        sl = tuple(sl)
        poly.img = self_[wd.sl]
        
        vmin, vmax = kw[wd.range_list[0]]
        
        # set new values to the existing imaging plane.
        im.set_clim((vmin, vmax))
        im.set_data(self_[sl])
        im.set_cmap(cmap)
        # set new values to the existing histogram.
        update_hist(self_[sl].flat, [vmin, vmax], h, hi)

        return None

    wd.interact(func)

    self.rois = poly.rois
    return poly

def measure_lines(self, curved=False, **kwargs):
    """
    Open a figure canvas and start recording line ROIs.
    """
    self_ = self.as_uint16()
    wd = Widget()
    wd.add(self_, "tzc")
    wd.add_range(self_)
    
    # prepare figure areas
    fig, all_axs = subplots(2)
    ax = all_axs[0][0]
    h = all_axs[1][0]
    tab = all_axs[0][1]
    scan = all_axs[1][1]
    tab.axis("off")
    scan.set_title("Line Scan")
    scan.grid()

    cmaps = self.get_cmaps()
    
    im = _imshow2d(self_[wd.sl], ax, cmap=cmaps[0], **kwargs)
    im_roi = ax.imshow(np.ma.masked_all(self_[wd.sl].shape), cmap="hsv", interpolation="none")
    hi = hist(self_[wd.sl], h, newfig=False)
    if (curved):
        line = CurveDrawer(self_[wd.sl], ax, im_roi, tab = tab, scan = scan)
    else:
        line = LineDrawer(self_[wd.sl], ax, im_roi, tab = tab, scan = scan)

    if (hasattr(self, "rois") and type(self.rois) is list):
        line.rois = self.rois
        line.set_data()

    def func(**kw):
        sl = []
        cmap = cmaps[0]
        for a in wd.axis_list:
            if (a in "tz"):
                sl.append(kw[a]-1)
            elif (a == "c"):
                sl.append(kw[a]-1)
                cmap = cmaps[kw[a]-1]
            else:
                raise ValueError(f"Unknown axis: {a}")

        sl = tuple(sl)
        line.img = self_[sl]
        
        vmin, vmax = kw[wd.range_list[0]]
        
        # set new values to the existing imaging plane.
        im.set_clim((vmin, vmax))
        im.set_data(self_[sl])
        im.set_cmap(cmap)
        # set new values to the existing histogram.
        update_hist(self_[sl].flat, [vmin, vmax], h, hi)

        return None

    wd.interact(func)

    self.rois = line.rois
    return line

def measure(self, rois=None, method="mean"):
    """
    Measure all the 2D-images for every ROI.
    
    Parameters:
    rois = single ROI instance or list of ROI instances.
    method = method of measurement.
    
    Returns:
    List of measurement results.
    e.g. If self is 5-frame, 3-slices, 2-channel image (i.e. shape=(5,3,2,?,?), axes=tzcyx),
            then returned array 
    """
    if (isinstance(rois, ROI)):
        rois = [rois]
    elif (rois is None and hasattr(self, "rois")):
        rois = self.rois
    else:
        raise TypeError("ROI not found.")

    func = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max, "median": np.median}[method]
    
    outputs = []
    for roi in rois:
        if (not isinstance(roi, ROI)):
            raise TypeError(f"non-ROI object included in 'rois': {type(roi)}")
        flatimg = self[roi].reshape(self.shape[:-2] + (-1,))
        result = func(flatimg, axis=-1)
        result.axes = self.axes[:-2]
        outputs.append(result)
    
    return outputs

def crop_rectangles(self, **kwargs):
    """
    Open a figure canvas and crop interactively.
    """
    self_ = self.as_uint16()
    wd = Widget()
    wd.add(self_, "tzc")
    wd.add_range(self_)
    
    fig, all_axs = subplots(1)
    ax = all_axs[0]
    h = all_axs[1]
    
    cmaps = self.get_cmaps()        
    
    im = _imshow2d(self_[wd.sl], ax, cmap=cmaps[0], interpolation="none", **kwargs)
    im_roi = ax.imshow(np.ma.masked_all(self_[wd.sl].shape), cmap="hsv", interpolation="none")
    hi = hist(self_[wd.sl], h, newfig=False)
    
    rect_cropper = RectangleCropper(self_[wd.sl], ax, im_roi, img_total=self)

    def func(**kw):
        sl = []
        cmap = cmaps[0]
        for a in wd.axis_list:
            if (a in "tz"):
                sl.append(kw[a]-1)
            elif (a == "c"):
                sl.append(kw[a]-1)
                cmap = cmaps[kw[a]-1]
            else:
                raise ValueError(f"Unknown axis: {a}")

        sl = tuple(sl)
        rect_cropper.img = self_[sl]
        rect_cropper.draw()
        
        vmin, vmax = kw[wd.range_list[0]]
        
        # set new values to the existing imaging plane.
        im.set_clim((vmin, vmax))
        im.set_data(self_[sl])
        im.set_cmap(cmap)
        # set new values to the existing histogram.
        update_hist(self_[sl].flat, [vmin, vmax], h, hi)

        return None

    wd.interact(func)

    return rect_cropper.imgs


def update_hist(data, xlim, ax, hist_output):
    heights, xvals = np.histogram(data, bins=hist_output[0].size, density=True)
    width = xvals[1] - xvals[0]
    ax.set_xlim(xlim)
    ax.set_ylim([0, heights.max()*1.05])
    for k, c in enumerate(hist_output[2].get_children()):
        c.set_x(xvals[k])
        c.set_height(heights[k])
        c.set_width(width)
    return None

def subplots(n_img):
    if (n_img == 1):
        size = (4, 5)
    elif (n_img == 2):
        size = (9, 4)
    elif (n_img == 3):
        size = (10, 4)
    else:
        size = (11, 4)
    return plt.subplots(2, n_img, figsize=size, gridspec_kw={"height_ratios": [3, 1]})