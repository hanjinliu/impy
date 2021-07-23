# impy

## More Numpy in Image Analysis! 

![](Figs/Img.png)

Image analysis programatically is sometimes troublesome like ...

1. for multi-dimensional images, you need to check which is time-axis and which is channel axis and so on.
2. you need to consider the output data types and shapes for every batch image processing.
3. you need to care about all the images' information such as the names and directories of original images.
4. hard to edit images interactively.

As a result, isn't it faster to analyze images using ImageJ? This module solves these major problems of Python based image analysis and makes it much more effective.

## Installation

```
pip install git+https://github.com/hanjinliu/impy
```

or

```
git clone https://github.com/hanjinliu/impy
```

`impy` is partly dependent on `numba`, `cupy`, `trackpy`, `mrcfile` and `dask-image`. Please install these packages if needed.

## Highlights

#### 1. Handling Axes Easily

**Image axes/scales are automatically read** from file metadata and as a result, arrays support **axis-targeted slicing** like:

```python
img["t=3;z=5:7"]
img["y=3,5,7"] = 0
```

Accordingly, broadcasting is more flexible. ([xarray](https://github.com/pydata/xarray) and [tensor_annotations](https://github.com/deepmind/tensor_annotations) seem similar in this sense)

#### 2. Automatic Batch Processing

Almost all the image processing functions can **automatically iterate** along all the axes needed. If you want to run batch Gaussian filter on a image hyperstack, just call `img.gaussian_filter()`, and the filtering function considers zyx-axes as a spatially connected dimensions and is repeated for every rest of axis like t, c. Prallel image processing is optimized for many function by temporarily converting into `dask` array. Check [Image Analysis Tools](#image-analysis-tools) for available functions.

You can even run batch processing **with your own functions** by decorating them with `@ip.bind`. See [Integrating Your Own Functions](#integrating-your-own-functions) part.

You may usually want to perform same filter function to images with different shapes and dimensions. `DataList` is a `list`-like object and `DataDict` is a `dict`-like object, which can iterate over all the images (or other objects) with `__getattr__` method.

```python
imglist = ip.DataList(imgs)
outputs = imglist.gaussian_filter(sigma=3) # filter is applied to all the images
```

With `for_params` method, you can easily repeat same function with different parameters.

```python
out = img.for_params("gaussian_filter", sigma=range(1,5))
```

#### 3. Metadata and History

All the information, history and metadata are inherited to outputs, like:

```python
img
```
```
    shape     : 10(t), 20(z), 256(y), 256(x)
  label shape : No label
    dtype     : uint16
  directory   : ...\images
original image: XXX
    history   : gaussian_filter(sigma=1)
```

Therefore, results can always be saved in the same directory, without copy-and-pasting paths.

#### 4. Image Viewer

`impy` provides seamless interface between [napari](https://github.com/napari/napari), a great image visualization tool. Image axes and other information are utilized before sending to `napari.Viewer`, so that you don't need to care about keyword arguments and what function should be called.

You can also **manually crop or label** `ImgArray` with `napari`'s `Shapes` objects, or **run impy functions inside the viewer**. I also implemented useful custom keybindings and widgets. See [Napari Interface](#napari-interface) for details.

#### 5. Extended Numpy Functions

In almost all the numpy functions, the keyword argument `axis` can be given as the symbol of axis like:

```python
np.mean(img, axis="z") # Z-projection
np.stack([img1, img2], axis="c") # color-merging
```

This is achieved by defining `__array_function__` method. See [here](https://numpy.org/devdocs/reference/arrays.classes.html) for details.

You can also make an `ImgArray` in a way similar to `numpy`:

```python
ip.array([2,4,6], dtype="uint16")
ip.random.normal(size=(100, 100))
```

#### 6. Reading/Processing Images Lazily

When you deal with large images, you may want to read just part of them to avoid waiting too long, or sometimes they are too large for the PC memory to read. In ImageJ there is an option called "virtual stack" but still it is not flexible enough.

In `impy`, there are several ways to efficiently deal with large datasets. See [Image I/O](#image-io) for details.

#### 7. GPU support

Affine transformation, deconvolution and many filter functions are automatically conducted with GPU if accessible. On importing `impy`, it checks if `cupy` and GPU are correctly installed, so that you don't have to change your code. See [Image Analysis Tools](#image-analysis-tools) for details.

## Contents

- **Arrays**
  - `ImgArray` is an array mainly used for image analysis here. Many `skimage`'s functions are wrapped in this class.
  - `PropArray` is an array that contains properties of another array, such as mean intensities of fixed regions of an array. 
  - `Label` is also an array type while it is only used for labeling of another image and is always attached to it. 
  - `PhaseArray` is an array that contains phase values. Unit (radian or degree) and periodicity are always tagged to itself so that you don't need to care about them. 
  - `LazyImgArray` keeps memory map to an image as an `dask` array and you can access image metadata and slice the images without reading them. Some filter functions are supported in `dask-image`.

- **DataFrames**
  - `MarkerFrame` is a subclass of `pandas.DataFrame` and it is specialized in storing coordinates and markers, such as xyz-coordinates of local maxima. This class also supports axis targeted slicing `df["x=4;y=5"]`. Tracking methods are also available, which call [trackpy](https://github.com/soft-matter/trackpy) inside.
  - `TrackFrame` is quite similar to `MarkerFrame` while it is only retuned when points are linked by particle tracking. It has information of track ID.

- **Others**
  - `DataList` and `DataDict` can apply same method to all the data inside it.
  - `gui` is a controller object that connects console and `napari.Viewer`.

## Image Analysis Tools

`ImgArray` has a lot of member functions for image analysis. Some of them supports multiprocessing with `dask`. ":heavy_check_mark:" indicates (partially) GPU support is available.

- **Drift/Aberration Correction**
  - `track_drift`, `drift_correction`:heavy_check_mark: &rarr; Correction of xy-drift.
  - `affine_correction`:heavy_check_mark: &rarr; Correction of such as chromatic aberration using Affine transformation.

- **2D/3D Deconvolution**
  - `wiener`:heavy_check_mark:, `lucy`:heavy_check_mark: &rarr; Classical Wiener's and Richardson-Lucy's algorithm.
  - `lucy_tv`:heavy_check_mark: &rarr; Richardson-Lucy's algorithm with total variance (TV) regularization.

- **Filters**
  - `mean_filter`:heavy_check_mark:, `meadian_filter`:heavy_check_mark:, `gaussian_filter`:heavy_check_mark: &rarr; Smoothing.
  - `dog_filter`:heavy_check_mark:, `doh_filter`:heavy_check_mark:, `log_filter`:heavy_check_mark: &rarr; Blob detection by DoG, DoH, LoG filter.
  - `edge_filter`, `laplacian_filter`:heavy_check_mark: &rarr; Edge detection.
  - `std_filter`:heavy_check_mark:, `coef_filter`:heavy_check_mark: &rarr; Standard deviation based filtering.
  - `lowpass_filter`:heavy_check_mark:, `highpass_filter`:heavy_check_mark: &rarr; FFT based filtering.
  - `entropy_filter`, `enhance_contrast`, `gabor_filter`:heavy_check_mark: &rarr; Object detection etc.
  - `ncc_filter` Template matching etc.
  - `kalman_filter`:heavy_check_mark:, `wavelet_denoising`, `rof_filter` &rarr; Advanced denoising methods.

- **Morphological Image Processing**
  - `erosion`, `dilation`, `opening`, `closing` &rarr; Basic ones.
  - `area_opening`, `area_closing`, `diameter_opening`, `diameter_closing` &rarr; Advanced ones.
  - `skeletonize`, `fill_hole` &rarr; Binary processing.
  - `count_neighbors` &rarr; For structure detection in binary images.
  - `remove_large_objects`, `remove_fine_objects` `remove_skeleton_structure` &rarr; Detect and remove objects.

- **Single Molecule Detection**
  - `find_sm`, `peak_local_max` &rarr; Return coordinates of single molecules.
  - `centroid_sm`, `gauss_sm`, `refine_sm` &rarr; Return coordinates in subpixel precision.

- **Background/Intensity Correction**
  - `rolling_ball`, `tophat`:heavy_check_mark: &rarr; Background subtraction.
  - `gaussfit`, `gauss_correction` &rarr; Use Gaussian for image correction.
  - `unmix` &rarr; Unmixing of leakage between channels.
  
- **Labeling**
  - `label`, `label_if`, `label_threshold` &rarr; Labeling using binary images.
  - `specify` &rarr; Labeling around coordinates.
  - `append_label` &rarr; Label images.
  - `expand_labels`, `watershed`, `random_walker` &rarr; Adjuct or segment labels.

- **Feature Detection**
  - `hessian_eigval`:heavy_check_mark:, `hessian_eig`:heavy_check_mark: &rarr; Hessian.
  - `structure_tensor_eigval`:heavy_check_mark:, `structure_tensor_eig`:heavy_check_mark: &rarr; Structure tensor.

- **Gradient Orientation Estimation**
  - `edge_grad`

- **Filament Orientation Estimation**
  - `hessian_angle`:heavy_check_mark: &rarr; Using Hessian eigenvector's orientations.
  - `gabor_angle`:heavy_check_mark: &rarr; Using Gabor filter's responses.

- **Property Measurement**
  - `regionprops` &rarr; Measure region properties such as mean intensity, Euler number, centroid, moment etc.
  - `pathprops` &rarr; Measure path properties such as mean intensity.
  - `lineprops`, `pointprops` &rarr; Measure line/point properties.

- **Profiling**
  - `reslice` &rarr; Get scan along a line or path.
  - `radial_profile` &rarr; Radial profiling of n-D images.

- **Others**
  - `focus_map` &rarr; Find focus using variance of Laplacian method. 
  - `stokes` &rarr; Analyze polarization using Stokes parameters.
  - `fft`:heavy_check_mark:, `power_spectra`:heavy_check_mark:, `ifft`:heavy_check_mark: &rarr; Fourier transformation.
  - `threshold` &rarr; Thresholding (many methods included).
  - `crop_center`, `crop_kernel`, `remove_edges`, `rotated_crop` &rarr; Crop image.
  - `clip`, `rescale_intensity` &rarr; Rescale the intensity profile into certain range.
  - `proj` &rarr; Z-projection along any axis.
  - `split`, `split_pixel_unit` &rarr; Split the image.
  - `pad`, `defocus`:heavy_check_mark: &rarr; Padding.
  - `iter`, `for_each_channel`, `for_params` &rarr; Easy iteration.
  - `set_scale` &rarr; set scales of any axes.
  - `imshow` &rarr; visualize 2-D or 3-D image with `matplotlib`.
  - `imsave` &rarr; save image (by default save in the directory that the original image was loaded).

## Correlations

- `fsc`, `fourier_shell_correlation` (alias) &rarr; Estimate resolution.
- `ncc`, `zncc` &rarr; (Zero-)Normalized cross correlation and masked version of it.
- `fourier_ncc`, `fourier_zncc` &rarr; (Zero-)Normalized cross correlation in Fourier space and masked version of it.

## Image I/O

`impy` provides useful I/O functions for effective image analysis.

- `impy.imread`
  
  Load image and convert them into `ImgArray`. Many formats supported:

  1. `>>> ip.imread(r"C:\Users\...\images.tif")` ... read single tif file.
  2. `>>> ip.imread(r"C:\Users\...\xx\*.tif")` ... read all the tif files in a directory.
  3. `>>> ip.imread(r"C:\Users\...\xx\**\*.tif")` ... read all the tif files recursively.
  4. `>>> ip.imread(r"C:\Users\...\images.tif", key="t=0")` ... only read the first time frame (much more efficient for large datasets).
  5. `>>> ip.imread(r"C:\Users\...\condition$i\image-pos$p.tif")` ... read all the tif files in a certain pattern. In this case, paths such as `"...\condition2\image-pos0.tif"` are read and they are arranged into `i`/`p`-axes.
   
- `impy.imread_collection`
  
  Load images into `DataList`. Wildcards are supported (like 2. and 3. in `impy.imread` examples).

  ```python
  imgs = ip.imread_collection(r"C:\Users\...\xx\**\*.tif")
  ip.gui.add(imgs.kalman_filter()) # run Kalman filter for each image stack and view them in napari.
  ```

- `impy.lazy_imread`
  
  Load an image lazily, i.e., image data is acturally read into memory only when it is needed. This function returns `LazyImgArray`, which cannot conduct operations but you can access metadata like those in `ImgArray`, by such as `.axes`, `.shape`, `.dirpath`, `.scale` etc.
  
  ```python
  limg = ip.lazy_imread(r"C:\Users\...\xx\**\*.tif")
  print(limg.gb) # print GB of the image
  limg_center = limg["z=120;y=1000:2000;x=1500:2500"] # get a part of the image
  limg_center.data # get data as ImgArray
  ```

  With preview in `napari`, you can manually select a small region of the large image and read it into ImgArray.

  ```python
  ip.gui.add(limg) # preview LazyImgArray
  ### In the napari window, add a shape layer, draw a rectangle and Ctrl+Shift+X to crop it###
  img = ip.gui.selection[0] # get the selected image layer as ImgArray
  ```

## Napari Interface

`impy.gui` has methods for better interface between images and `napari`.

![](Figs/FFT.gif)

- Add any objects (images, labels, points, ...) to the viewer by `ip.gui.add(...)`.
- Return all the manually selected layers' data by `layers = ip.gui.selection`.
- Run `ImgArray`'s method inside viewers.
- Translate and rescale layers with mouse.
  - `Alt` + mouse drag &rarr; lateral translation
  - `Alt` + `Shift` + mouse drag &rarr; lateral translation restricted in either x- or y-orientation (left button or right button respectively).
  - `Alt` + mouse wheel &rarr; rescaling
  - `Ctrl` + `Shift` + `R` &rarr; reset original states.

- Fast layer selection and manipulation.
  - `Ctrl` + `Shift` + `A` &rarr; Hide non-selected layers. Display all the layers by push again.
  - `Ctrl` + `Shift` + `F` &rarr; Move selected layers to front.
  - `Alt` + `L` &rarr; Convert all the shapes in seleted shape-layers into labels of selected image-layers.
  - `Ctrl` + `Shift` + `D` &rarr; Duplicate selected layers.
  - `Ctrl` + `Shift` + `X` &rarr; Crop selected image-layers with all the rectangles in selected shape-layers. Rotated cropping is also supported!
  - `/` &rarr; Reslice selected image-layers with all the lines and paths in selected shape-layers. Result is stored in `ip.gui.results` for now.
  - `Ctrl` + `P` &rarr; Projection of shape-layers or point-layers to 2D layers.
  - `Ctrl` + `G` / `Ctrl` + `Shift` + `G` &rarr; Link/Unlink layers. Like "grouping" in PowerPoint.
  - `Shift` + `S` / `S` &rarr; Add 2D/nD shape-layer.
  - `Shift` + `P` / `P` &rarr; Add 2D/nD point-layer.
- Show coordinates of selected point-layers or track-layers. You can also copy it to clipboard.
- Note pad in `Window > Note`.
- Call `impy.imread` in `File > imread ...`. Call `impy.imsave` in `File > imsave ...`.

`napari` is now under development itself so I'll add more and more functions (I'm especially looking forward to layer group and text layer).

## Integrating Your Own Functions

`ImgArray` is designed highly extensible. With `impy.bind`, You can easily integrate functions that converts:

- image &rarr; image (image filtering, thresholding etc.)
- image &rarr; scalar (measuring/estimating properties)
- image &rarr; label (feature detection, image segmentation etc.)

Suppose you want to use `imfilter`, a image filtering function that works on float images, for batch processing of multi-dimensional images. Just write

```python
import impy as ip
@ip.bind(indtype="float32")
def imfilter(img, param=None):
    # do something for a 2D or 3D image.
    return out
```

or

```python
ip.bind(imfilter, indtype="float32")
```

or in with-block for temporary usage

```python
with ip.bind(imfilter, indtype="float32"):
    ...
```

and now it's ready to execute batch-`imfilter`!

```python
img = ip.imread(r"...\images\XXX.tif")
img.imfilter(param=3)
```

This function is also accessible inside `napari` viewers.
