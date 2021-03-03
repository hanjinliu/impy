# impy

## Basic Usage

Load image by following code.
```python
import impy as ip
img = ip.imread(r"C:\Users\...\XXX.tif")
```

- `imshow` = visualize 2-D or 3-D image.
- `hist` = show the histogram of image intensity profile.
- `imsave` = save image (by default in the directory that the original image were loaded).

## Attributes of ImgArray

- `name` = name of the original image.
- `dirpath` = absolute path to the original image.
- `history` = history of applied analysis.
- `axes` = dimensions of image, "tzcyx"-order.
- `lut` = look up table.

## Flexible Slicing

When you want to access the first channel and 4-th to 10-th time points, you can do it by following code:

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

- `drift_correction`(plugin) = automatic drift correction using `phase_cross_correlation` function in skimage.
- `deconvolution3d`(plugin) = 3-D deconvolution.
- `mean_filter`, `meadian_filter`
- `rolling_ball` = for background subtraction.
- `rough_gaussfit` = fit the image to 2-D Gaussian (for correction of uneven irradiation).
- `proj` = Z-projection.
- `split` = split the image along any axis.

# Reference
flowdec
