# impy

More Numpy in image analysis! History of analysis, the original image and directory are all recorded in the object. Many image analysis tools are coded using functions in [scikit-image](https://github.com/scikit-image/scikit-image).

## Example

```python
import impy as ip
img0 = ip.imread(r"...\images\XXX.tif")
img0
```
    [Out]
        shape     : 10(t), 3(c), 512(y), 512(x)
        dtype     : uint16
      directory   : ...\images
    original image: XXX
       history    : 

```python
img = img0.proj(axis="t") # projection
img = img.median_filter(radius=3) # median filter
img = img[0] # get first channel
img 
```
    [Out]
        shape     : 512(y), 512(x)
        dtype     : uint16
      directory   : ...\images
    original image: XXX
       history    : mean-Projection(axis=t)->2D-Median-Filter(R=3)->getitem[0]

## Basic Usage

Load image with `imread` function. `ImgArray` object is created.
```python
# load single tif
img = ip.imread(r"C:\Users\...\XXX.tif")
# load tifs recursively from a directory
img = ip.imread_collection(r"C:\Users\...\XX_100nM", ignore_exception=True)
```

Stacking images with `impy.stack`.

```python
# make stack along channel axis
img = ip.stack([img1, img2], axis="c", dtype="uint16") 
```

Making synthetic three-channel image with `impy.array` and manually set its axes and LUTs.

```python
img = ip.array(np.random.rand(3*40*30).reshape(3,40,30)*100, name="random noise")
img.axes = "cyx"
img.lut = ["teal", "violet", "gold"]
```

## Basic Attributes and Functions of ImgArray

### Attributes

- `name` = name of the original image.
- `dirpath` = absolute path to the original image.
- `history` = history of applied analysis.
- `axes` = dimensions of image, `ptzcyx`-order.
- `lut` = look up table.
- `value` (property) = show the array in numpy format.
- `range` (property) = return a tuple of min/max.

### Functions

- `imshow` = visualize 2-D or 3-D image.
- `hist` = show the histogram of image intensity profile.
- `imsave` = save image (by default save in the directory that the original image was loaded).

## Data Type Conversion

`uint8`, `uint16`, `bool` and `float32` are supported for type conversion.
- `as_uint8` = convert to `uint8`.
- `as_uint16` = convert to `uint16`.
- `as_float` = convert to `float32`.
- `as_img_type` = convert to any supported types.

## Automatic Saturation and Type Conversion

Overflow, underflow and type conversion is considered for operations `+`, `-`, `*` and `/`.
```python
# img = <uint16 image>

img + 10000     # pixel values larger than 65535 
                # is substituted to 63353

img - 10000     # pixel values smaller than 0 is
                # substituted to 0

img / 10        # output is converted to float32 
                # where `img` itself is not

img /= 10       # `img` is converted to float32
```

## Flexible Slicing

When you want to access the first channel and 4-th to 10-th time points, you can do it by:

```python
img["c=1;t=4-10"]       # get items
img["c=1;t=4-10"] = 0   # set items
```

List-like slicing is also supported:

```python
img["t=1,3-6,9"]  # this means [0,2,3,4,5,8] in t-axis
```


## Image Analysis

`ImgArray` has a lot of member functions for image analysis. Some of them supports multiprocessing.

- `drift_correction` (plugin) = automatic drift correction using `phase_cross_correlation` function in skimage.
- `lucy2d`, `lucy3d` (plugin) = deconvolution of images.
- `affine_correction` = Correction of such as chromatic aberration using Affine transformation.
- `hessian_eigval`, `hessian_eig` = feature detection using Hessian method.
- `structure_tensor_eigval`, `structure_tensor_eig` = feature detection using structure tensor.
- `dog_filter` = filtering using difference of Gaussian method.
- `mean_filter`, `meadian_filter`, `gaussian_filter` = for smoothing.
- `erosion`, `dilation`, `opening`, `closing` = for morphological processing.
- `rolling_ball`, `tophat` = for background subtraction.
- `gaussfit`, `gaussfit_particle` = fit the image to 2-D Gaussian (for correction of uneven irradiation or single molecular analysis).
- `fft`, `ifft` = Fourier transformation.
- `threshold` = thresholding.
- `peak_local_max` = find maxima.
- `label`, `regionprops` = labeling images and measure properties.
- `crop_center`, `crop_circle` = crop image.
- `clip_outliers`, `rescale_intensity` = rescale the intensity profile into certain range.
- `proj` = Z-projection along any axis.
- `split` = split the image along any axis.

# References
For 3-D PSF generation, [flowdec](https://github.com/hammerlab/flowdec) is imported in this package. For deconvolution, function `lucy` from Julia-coded package [Deconvolution.jl](https://github.com/JuliaDSP/Deconvolution.jl) is translated into Python.
