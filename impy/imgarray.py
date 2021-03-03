import numpy as np
from skimage.morphology import white_tophat, disk
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import restoration as skres
from skimage import exposure as skexp
from scipy import optimize as opt
from .func import get_meta, record, gauss2d, square, circle, del_axis, add_axes
from .base import BaseArray
from .axes import Axes
from .roi import Rectangle

def _median(args):
    sl, data, selem = args
    return (sl, skfil.rank.median(data, selem))

def _mean(args):
    sl, data, selem = args
    return (sl, skfil.rank.mean(data, selem))

def _rolling_ball(args):
    sl, data, radius, smooth = args
    if (smooth):
        _, ref = _mean((sl, data, np.ones((3, 3))))
        back = skres.rolling_ball(ref, radius=radius)
        tozero = (back > data)
        back[tozero] = data[tozero]
    else:
        back = skres.rolling_ball(data, radius=radius)
    
    return (sl, data - back)

def _tophat(args):
    sl, data, selem = args
    return (sl, white_tophat(data, selem))
    


class ImgArray(BaseArray):
    def __init__(self, path:str, axes=None):
        super().__init__(path, axes)

    @record
    def affine(self, **kwargs):
        """
        Affine transformation
        kwargs: matrix, scale, rotation, shear, translation
        """
        mx = sktrans.AffineTransform(**kwargs)
        out = sktrans.warp(self.view(np.ndarray), mx).view(self.__class__)
        out._set_info(self, f"Affine-Translate({kwargs})")
        return out.as_uint16()
    
    @record
    def translate(self, translation=None):
        """
        Simple translation of image, i.e. (x, y) -> (x+dx, y+dy)
        """
        mx = sktrans.AffineTransform(translation=translation)
        out = sktrans.warp(self.view(np.ndarray), mx).view(self.__class__)
        out._set_info(self, f"Translate{translation}")
        return out.as_uint16()

    @record
    def rescale(self, scale=1/16):
        try:
            scale = float(scale)
        except:
            raise TypeError(f"scale must be float, but got {type(scale)}")
        scale_ = []
        for a in self.axes:
            if (a in "yx"):
                scale_.append(scale)
            else:
                scale_.append(1)
        out = sktrans.rescale(self.view(np.ndarray).astype("float64"), scale_, anti_aliasing=False).view(self.__class__)
        out._set_info(self, f"Rescale(x1/{np.round(1/scale, 1)})")
        return out.as_uint16()
    
    
    @record
    def gaussfit(self, p0=None):
        """
        Fit the image to 2-D Gaussian.

        Parameters
        ----------
        p0 : list or None, optional
            Initial parameters, by default None

        Returns
        -------
        ImgArray
            Fit image.

        Raises
        ------
        TypeError
            If self is not two dimensional.
        """        
        if (self.ndim != 2):
            raise TypeError(f"input must be two dimensional, but got {self.shape}")
        
        # initialize parameters
        if (p0 is None):
            mu1, mu2 = np.unravel_index(np.argmax(self), self.shape)  # 2-dim argmax
            sg1 = self.shape[0]
            sg2 = self.shape[1]
            B = np.percentile(self, 5)
            A = np.percentile(self, 95) - B
            p0 = mu1, mu2, sg1, sg2, A, B
        
        print("\n --------------------- GaussFit --------------------- ")
        print("Initial parameters:")
        print("mu1={:.3g}, mu2={:.3g}, sg1={:.3g}, sg2={:.3g}, A={:.3g}, B={:.3g}".format(*p0))
        
        param = opt.minimize(square, p0, args=(gauss2d, self.view(np.ndarray).astype("float64"))).x
        
        print("Fitting results:")
        print("mu1={:.3g}, mu2={:.3g}, sg1={:.3g}, sg2={:.3g}, A={:.3g}, B={:.3g}".format(*param))
        print(" ---------------------------------------------------- \n")

        # prepare fit image
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])

        out = gauss2d(x, y, *(param)).view(self.__class__)
        out._set_info(self, "Gaussian-Fit")

        # set temporal result
        out.temp = param

        return out
    
    @record
    def rough_gaussfit(self, scale=1/16, p0=None):
        """
        Execute gaussfit() after making a rough image using rescale(), for large image.
        Optimal parameter set is also correctly scaled and stored in attribute 'temp'.

        Parameters
        ----------
        scale: float
            Size of rough image.
            
        p0 : list or None, optional
            Initial parameters, by default None

        Returns
        -------
        ImgArray
            Fit image.

        Raises
        ------
        TypeError
            If self is not two dimensional.
        """ 
        rough = self.rescale(scale)
        rough_gauss = rough.gaussfit(p0)
        
        # rescale parameters
        param = rough_gauss.temp
        param[:4] /= scale
        
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        out = gauss2d(x, y, *param).view(self.__class__)
        out._set_info(self, f"Rough-Gaussian-Fit(x1/{np.round(1/scale, 1)})")
        out.temp = param
        return out
    
    @record
    def tophat(self, radius=50, n_cpu=4):
        """
        Subtract Background using top-hat algorithm.

        Parameters
        ----------
        radius : int, optional
            Radius of hat, by default 50
        n_cpu : int, optional
            Number of CPU to use

        Returns
        -------
        ImgArray
            Background subtracted image.
        """        
        disk_ = disk(radius)
        out = self.parallel(_tophat, "tzc", disk_, n_cpu=n_cpu)
        out._set_info(self, f"Top-Hat(R={radius})")
        return out
    
    @record
    def rolling_ball(self, radius=50, smoothing=True, n_cpu=4):
        """
        Subtract Background using rolling-ball algorithm.

        Parameters
        ----------
        radius : int, optional
            Radius of rolling ball, by default 50
        smoothing : bool, optional
            If apply 3x3 averaging before creating background.
        n_cpu : int, optional
            Number of CPU to use
            
        Returns
        -------
        ImgArray
            Background subtracted image.
        """        
        out = self.parallel(_rolling_ball, "tzc", radius, smoothing, n_cpu=n_cpu)
        out._set_info(self, f"Rolling-Ball(R={radius})")
        return out
    
    @record
    def mean_filter(self, radius=1, n_cpu=4):
        """
        Run mean filter.

        Parameters
        ----------
        radius : int, optional
            Radius of filter, by default 1
        n_cpu : int, optional
            Number of CPU to use

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        disk_ = disk(radius)
        out = self.parallel(_mean, "tzc", disk_, n_cpu=n_cpu)
        out._set_info(self, f"Mean-Filter(R={radius})")
        return out
    
        
    @record
    def median_filter(self, radius=1, n_cpu=4):
        """
        Run median filter. 

        Parameters
        ----------
        radius : int, optional
            Radius of filter, by default 1
        n_cpu : int, optional
            Number of CPU to use

        Returns
        -------
        ImgArray
            Filtered image.
        """        
        disk_ = disk(radius)
        out = self.parallel(_median, "tzc", disk_, n_cpu=n_cpu)
        out._set_info(self, f"Median-Filter(R={radius})")
        return out

    
    @record
    def fft(self):
        """
        Fast Fourier transformation.
        This function returns complex array. Inconpatible with many functions here.
        """
        if (self.ndim != 2):
            raise TypeError(f"input must be two dimensional, but got {self.shape}")
        freq = np.fft.fft2(self.view(np.ndarray))
        out = np.fft.fftshift(freq).view(self.__class__)
        out._set_info(self, "FFT")
        return out
    
    @record
    def ifft(self):
        if (self.ndim != 2):
            raise TypeError(f"input must be two dimensional, but got {self.shape}")
        freq = np.fft.fftshift(self.view(np.ndarray))
        out = np.fft.ifft2(freq).real.view(self.__class__)
        out._set_info(self, "IFFT")
        return out
    
    @record
    def threshold(self, thr=None, method:str="otsu", light_bg=False, iters="c", **kwargs):
        """
        Parameters
        ----------
        thr: int or array or None, optional
            Threshold value. If None, this will be determined by 'method'.
        method: str, optional
            Thresholding algorithm.
        light_bg: bool, default is False
            If background is brighter
        iters: str, default is 'c'
            Around which axes images will be iterated.
        **kwargs:
            Keyword arguments that will passed to function indicated in 'method'.

        """

        methods_ = {"isodata": skfil.threshold_isodata,
                    "li": skfil.threshold_li,
                    "local": skfil.threshold_local,
                    "mean": skfil.threshold_mean,
                    "min": skfil.threshold_minimum,
                    "minimum": skfil.threshold_minimum,
                    "multiotsu": skfil.threshold_multiotsu,
                    "niblack": skfil.threshold_niblack,
                    "otsu": skfil.threshold_otsu,
                    "sauvola": skfil.threshold_sauvola,
                    "triangle": skfil.threshold_triangle,
                    "yen": skfil.threshold_yen
                    }
        if (thr is None):
            method = method.lower()
            try:
                func = methods_[method]
            except KeyError:
                s = ", ".join(list(methods_.keys()))
                raise KeyError(f"{method}\nmethod must be: {s}")
            thr = func(self.view(np.ndarray), **kwargs)

            out = np.zeros(self.shape, dtype=bool)
            for t, img in self.as_uint16().iter(iters):
                if (light_bg):
                    out[t] = img <= thr
                else:
                    out[t] = img >= thr
            out = out.view(self.__class__)
            out._set_info(self, f"Thresholding({method})")

        else:
            if (light_bg):
                out = self <= thr
            else:
                out = self >= thr
            out._set_info(self, f"Thresholding({thr:.1f})")
        
        out.temp = thr
        return out
        
    @record
    def crop_circle(self, radius=None, outzero=True):
        """
        Make a circular window function.
        """
        if (radius is None):
            radius = np.min(self.xyshape()) // 2
        
        circ = circle(radius, self.xyshape())
        if (not outzero):
            circ = 1 - circ
        circ = add_axes(self.axes, self.shape, circ)
        out = self * circ
        out.history.pop()
        out._set_info(self, f"Crop-Circle(R={radius}, {'outzero' if outzero else 'inzero'})")
        return out
    
    def specify(self, x, y, dx, dy, position="corner"):
        """
        Make a rectancge ROI.
        """
        if (position == "corner"):
            pass
        elif (position == "center"):
            x -= dx//2
            y -= dy//2
        else:
            raise ValueError("'position' must be 'corner' or 'center'")
        rect = Rectangle(x, y, x+dx, y+dy)
        if (hasattr(self, "rois") and type(self.rois) is list):
            self.rois.append(rect)
        else:
            self.rois = [rect]
        
        return rect

    
    @record
    def crop_center(self, scale=0.5):
        """
        Crop out the center of an image.
        e.g. when scale=0.5, create 512x512 image from 1024x1024 image.
        """
        if (scale <= 0 or 1 < scale):
            raise ValueError(f"scale must be (0, 1], but got {scale}")
        
        sizex, sizey = self.xyshape()
        
        x0 = int(sizex / 2 * (1 - scale)) + 1
        x1 = int(sizex / 2 * (1 + scale))
        y0 = int(sizey / 2 * (1 - scale)) + 1
        y1 = int(sizey / 2 * (1 + scale))

        out = self.getitem(f"x={x0}-{x1},y={y0}-{y1}")
        out.history[-1] = f"Crop-Center(scale={scale})"
        
        return out
    
    @record
    def split(self, axis="c"):
        """
        Split n-dimensional image into (n-1)-dimensional images.

        Parameters
        ----------
        axis : str or int, optional
            Along which axis the original image will be split, by default "c"

        Returns
        -------
        list of ImgArray
            Separate images
        """        
        
        axisint = self.axisof(axis)
        imgs = list(np.moveaxis(self, axisint, 0))
        for i, img in enumerate(imgs):
            img.history[-1] = f"Split(axis={axis})"
            img.axes = del_axis(self.axes, axisint)
            if (axis == "c" and self.lut is not None):
                img.lut = [self.lut[i]]
            else:
                img.lut = None
        return imgs

    @record
    def proj(self, axis="z", method="mean"):
        """
        Z-projection.
        'method' must be in func_dict.keys() or some function like np.mean.
        """
        axisint = self.axisof(axis)
        func_dict = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max, "median": np.median}
        if (method in func_dict.keys()):
            func = func_dict[method]
        elif (callable(method)):
            func = method
        else:
            raise TypeError(f"'method' must be one of {', '.join(list(func_dict.keys()))} or callable object.")
        out = func(self.view(np.ndarray), axis=axisint).view(self.__class__)
        out._set_info(self, f"{method}-Projection(axis={axis})", del_axis(self.axes, axisint))
        return out.as_uint16()

    @record
    def rescale_intensity(self, lower=0, upper=100, dtype=np.uint16):
        """
        [min, max] -> [0, 1)
        2^-16 = 1.5 x 10^-5
        out = skimage.exposure.rescale_intensity(out, dtype=np.uint16, in_range=...)
        """
        out = self.view(np.ndarray).astype("float64")
        lowerlim = np.percentile(out, lower)
        upperlim = np.percentile(out, upper)
        out = skexp.rescale_intensity(out, in_range=(lowerlim, upperlim), out_range=dtype)
        
        out = out.view(self.__class__)
        out._set_info(self, f"Rescale-Intensity({lower:.1f}-{upper:.1f})")
        out.temp = [lowerlim, upperlim]
        return out

    def sort_axes(self):
        return self.transpose(self.axes.argsort())
    
    # numpy functions that will change/discard order
    def transpose(self, axes):
        """
        change the order of image dimensions.
        'axes' will also be arranged.
        """
        out = super().transpose(axes)
        if (self.axes.is_none()):
            new_axes = None
        else:
            new_axes = "".join([self.axes[i] for i in list(axes)])
        out._set_info(self, new_axes = new_axes)
        return out
    
    def flatten(self):
        out = super().flatten()
        out._set_info(self, new_axes = None)
        return out
    
    def ravel(self):
        out = super().ravel()
        out._set_info(self, new_axes = None)
        return out
    
    def reshape(self, newshape, axes=None):
        if (axes is not None and len(newshape) != len(axes)):
            raise ValueError("newshape and axes have incompatible lengths.")
        out = super().reshape(newshape)
        out._set_info(self, new_axes = axes)
        return out

# non-member functions.

def array(arr, name="array", dtype="uint16", axes=None, dirpath="", history=[], metadata={}, lut=None):
    """
    make an ImgArray object, just like np.array(x)
    """
    if (isinstance(arr, ImgArray)):
        return arr
    
    if (type(arr) is str):
        raise TypeError(f"String is invalid input. Do you mean imread(path)?")
        
    self = arr.view(ImgArray)
    self.axes = axes
    self.dirpath = dirpath
    self.name = name
    self.history = history
    self.metadata = metadata
    self.lut = lut
    
    if (dtype == "uint16"):
        return self.as_uint16()
    elif (dtype == "uint8"):
        return self.as_uint8()
    else:
        return self.astype("float32")

def imread(path:str):
    # make object
    meta = get_meta(path)
    self = ImgArray(path, axes=meta["axes"])
    self.metadata = meta["ijmeta"]
    if (meta["history"]):
        self.name = meta["history"].pop(0)
        self.history = meta["history"]
    
    # In case the image is in yxc-order. This sometimes happens.
    if ("c" in self.axes and self.sizeof("c") > self.sizeof("x")):
        self = np.moveaxis(self, -1, -3)
        _axes = self.axes.axes
        _axes = _axes[:-3] + "cyx"
        self.axes = _axes
    
    return self.transpose(self.axes.argsort()) # arrange in tzcyx-order


def read_meta(path:str):
    meta = get_meta(path)
    return meta


def stack(imgs, axis="c"):
    """
    imgs: list or tuple (or other iterable objects) of 2D-images.
    axis: to specify which axis will be the new axis.
    This function can be used to create stack, hyperstack or multi-channel image
    """
    
    if (isinstance(imgs, np.ndarray)):
        raise TypeError("cannot stack single array.")
    
    # find where to add new axis
    if (imgs[0].axes):
        new_axes = Axes(axis + str(imgs[0].axes))
        new_axes.sort()
        _axis = new_axes.find(axis)
    else:
        new_axes = None
        _axis = 0

    arrs = [np.array(img.as_uint16()) for img in imgs]

    out = np.stack(arrs, axis=0)
    out = np.moveaxis(out, 0, _axis)
    out = array(out)    
    out._set_info(imgs[0], f"Make-Stack(axis={axis})", new_axes)
    
    # connect LUT if needed.
    if (axis == "c"):
        luts = [img.lut[0] for img in imgs]
        out.lut = luts
    
    return out


