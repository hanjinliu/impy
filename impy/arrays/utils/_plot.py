import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ._skimage import skimage, skexp

def plot_drift(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, title="drift")
    ax.plot(result.x, result.y, marker="+", color="red")
    ax.grid()
    # delete the default axes and let x=0 and y=0 be new ones.
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # let the interval of x-axis and that of y-axis be equal.
    ax.set_aspect("equal")
    # set the x/y-tick intervals to 1.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    return None

def plot_gaussfit_result(raw, fit):
    x0 = raw.shape[1]//2
    y0 = raw.shape[0]//2
    plt.figure(figsize=(6,4))
    plt.subplot(2, 1, 1)
    plt.title("x-direction")
    plt.plot(raw[y0].value, color="gray", alpha=0.5, label="raw image")
    plt.plot(fit[y0], color="red", label="fit")
    plt.subplot(2, 1, 2)
    plt.title("y-direction")
    plt.plot(raw[:,x0].value, color="gray", alpha=0.5, label="raw image")
    plt.plot(fit[:,x0], color="red", label="fit")
    plt.tight_layout()
    plt.show()
    return None

subplots = plt.subplots
plot_1d = plt.plot
show = plt.show

def plot_2d(img, ax=None, **kwargs):
    vmax, vmin = _determine_range(img)
    interpol = "bilinear" if img.dtype == bool else "none"
    imshow_kwargs = {"cmap": "gray", "vmax": vmax, "vmin": vmin, "interpolation": interpol}
    imshow_kwargs.update(kwargs)
    if ax is None:
        plt.imshow(img, **imshow_kwargs)
    else:
        ax.imshow(img, **imshow_kwargs)

def plot_3d(imglist, **kwargs):
    vmax, vmin = _determine_range(np.stack(imglist))

    interpol = "bilinear" if imglist[0].dtype == bool else "none"
    imshow_kwargs = {"cmap": "gray", "vmax": vmax, "vmin": vmin, "interpolation": interpol}
    imshow_kwargs.update(kwargs)
    
    n_img = len(imglist)
    n_col = min(n_img, 4)
    n_row = int(n_img / n_col + 0.99)
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row))
    if len(imglist) > 1:
        ax = ax.flat
    else:
        ax = [ax]
    for i, img in enumerate(imglist):
        ax[i].imshow(img, **imshow_kwargs)
        ax[i].axis("off")
        ax[i].set_title(f"Image-{i+1}")

def plot_2d_label(img, label, alpha, ax=None,  **kwargs):
    vmax, vmin = _determine_range(img)
    imshow_kwargs = {"vmax": vmax, "vmin": vmin, "interpolation": "none"}
    imshow_kwargs.update(kwargs)
    vmin = imshow_kwargs["vmin"]
    vmax = imshow_kwargs["vmax"]
    if vmin and vmax:
        image = (np.clip(img, vmin, vmax) - vmin)/(vmax - vmin)
    else:
        image = img
    overlay = skimage.color.label2rgb(label, image=image, bg_label=0, alpha=alpha, image_alpha=1)

    if ax is None:
        plt.imshow(overlay, **imshow_kwargs)
    else:
        ax.imshow(overlay, **imshow_kwargs)

def plot_3d_label(imglist, labellist, alpha, **kwargs):
    vmax, vmin = _determine_range(np.stack(imglist))

    imshow_kwargs = {"vmax": vmax, "vmin": vmin, "interpolation": "none"}
    imshow_kwargs.update(kwargs)
    
    n_img = len(imglist)
    n_col = min(n_img, 4)
    n_row = int(n_img / n_col + 0.99)
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row))
    if len(imglist) > 1:
        ax = ax.flat
    else:
        ax = [ax]
    for i, img in enumerate(imglist):
        vmin = imshow_kwargs["vmin"]
        vmax = imshow_kwargs["vmax"]
        if vmin and vmax:
            image = (np.clip(img, vmin, vmax) - vmin)/(vmax - vmin)
        else:
            image = img
        overlay = skimage.color.label2rgb(labellist[i], image=image, bg_label=0, 
                                          alpha=alpha, image_alpha=1)
        ax[i].imshow(overlay, **imshow_kwargs)
        ax[i].axis("off")
        ax[i].set_title(f"Image-{i+1}")

def hist(img, contrast):
    plt.figure(figsize=(4, 1.7))

    nbin = min(int(np.sqrt(img.size / 3)), 256)
    d = img.astype(np.uint8).ravel() if img.dtype==bool else img.ravel()
    y, x = skexp.histogram(d, nbins=nbin)
    plt.plot(x, y, color="gray")
    plt.fill_between(x, y, np.zeros(len(y)), facecolor="gray", alpha=0.4)
    
    if contrast is None:
        contrast = [img.min(), img.max()]
    x0, x1 = contrast
    
    plt.xlim(x0, x1)
    plt.ylim(0, y[(x0<x)&(x<x1)].max())
    plt.yticks([])


def _determine_range(arr):
    """
    Called in imshow()
    """
    if arr.dtype == bool:
        vmax = 1
        vmin = 0
    elif arr.dtype.kind == "f":
        vmax = np.percentile(arr, 99.99)
        vmin = np.percentile(arr, 0.01)
    else:
        try:
            vmax = np.percentile(arr[arr>0], 99.99)
            vmin = np.percentile(arr[arr>0], 0.01)
        except IndexError:
            vmax = arr.max()
            vmin = arr.min()
    return vmax, vmin