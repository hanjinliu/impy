# impy

More Numpy in image analysis! History of analysis, the original image and directory are all recorded in the object.

## Example

```python
import impy as ip
img0 = ip.imread(r"...\images\XXX.tif")
img0

#     shape     : 10(t), 3(c), 512(y), 512(x)
#     dtype     : uint16
#   directory   : ...\images
# original image: XXX
#    history    : 
```

```python
img = img0.proj(axis="t") # projection
img = img.median_filter(radius=3) # median filter
img = img[0] # get first channel
img 

#     shape     : 512(y), 512(x)
#     dtype     : uint16
#   directory   : ...\images
# original image: XXX
#    history    : mean-Projection(axis=t)->Median-Filter(R=3)->getitem[0]
```

## Basic Usage

Load image with `imread` function. `ImgArray` object is created.
```python
img = ip.imread(r"C:\Users\...\XXX.tif") # load single tif
img = ip.imread_collection(r"C:\Users\...\XX_100nM", ignore_exception=True) # load tifs recursively from a directory
```

Stacking images with `impy.stack`

```python
img = ip.stack([img1, img2], axis="c") # stack along channel
```

## Basic Attributes and Functions of ImgArray

### Attributes

- `name` = name of the original image.
- `dirpath` = absolute path to the original image.
- `history` = history of applied analysis.
- `axes` = dimensions of image, "tzcyxs"-order.
- `lut` = look up table.

### Functions

- `imshow` = visualize 2-D or 3-D image.
- `hist` = show the histogram of image intensity profile.
- `imsave` = save image (by default save in the directory that the original image was loaded).


## Flexible Slicing

When you want to access the first channel and 4-th to 10-th time points, you can do it by:

```python
img["c=1,t=4-10"]
```

## Interactive Plot and Measurement

With `ipywidgets` and in correct environment, you can visualize images interactively with several options:

- `imshowc` = all the channels are visualized in the same time.
- `imshowz` = several z-slices are visualized in the same time.
- `imshow_comparewith` = two images are visualized in the same time.

You can also measure images and obtain ROI

- `measure_rectangles`
- `measure_polygons`
- `measure_lines`

## Image Analysis

`ImgArray` has a lot of member functions for image analysis.

- `drift_correction` (plugin) = automatic drift correction using `phase_cross_correlation` function in skimage.
- `lucy3d` (plugin) = 3-D deconvolution of confocal images.
- `mean_filter`, `meadian_filter` = for filtering.
- `rolling_ball`, `tophat` = for background subtraction.
- `rough_gaussfit` = fit the image to 2-D Gaussian (for correction of uneven irradiation).
- `proj` = Z-projection along any axis.
- `split` = split the image along any axis.

# Reference
For 3-D PSF generation, [flowdec](https://github.com/hammerlab/flowdec) is imported in this package.
