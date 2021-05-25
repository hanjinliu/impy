# impy

## More Numpy in Image Analysis! 

![](Figs/Img.png)

[scikit-image](https://github.com/scikit-image/scikit-image) is very useful but sometimes troublesome like ...
1. for multi-dimensional images, you need to check which is time-axis and which is channel axis and so on.
2. you need to consider the output data types and shapes for every iteration of image processing.
3. you need to care about all the images' information such as the names and directories of original images.

With extended `numpy.ndarray` this module solves major problems above that happens when you code image analysis in Python. 

1. **Image axes are automatically read** from Tiff file and arrays support **axis targeted slicing** like `img["t=3;z=5:7"]`.
2. Almost all the image processing functions can **automatically iterate** along all the axes needed.
3. All the information and metadata are inherited to outputs.

This module contains several classes.

- `ImgArray` is an array mainly used for image analysis here. Many `skimage`'s functions are wrapped in this class.
- `PropArray` is an array that contains properties of another array, such as mean intensities of fixed regions of an array. 
- `Label` is also an array type while it is only used for labeling of another image and is always attached to it. 
- `PhaseArray` is an array that contains phase values. Unit (radian or degree) and periodicity are always tagged to itself so that you don't need to care about them. 
- `MarkerFrame` is a subclass of `pandas.DataFrame` and it is specialized in storing coordinates and markers, such as xyz-coordinates of local maxima. This class also supports axis targeted slicing `df["x=4;y=5"]`.
- `TrackFrame` is quite similar to `MarkerFrame` while it is only retuned when points are linked by particle tracking. It has information of track ID.

This module also provides many image analysis tools and seamless interface between [napari](https://github.com/napari/napari), which help you to operate with and visualize n-D images, and [trackpy](https://github.com/soft-matter/trackpy), which enables efficient molecule tracking.


## Image Analysis Tools

`ImgArray` has a lot of member functions for image analysis. Some of them supports multiprocessing.

- **Drift/Aberration Correction**
  - `track_drift`, `drift_correction`
  - `affine_correction` &rarr; Correction of such as chromatic aberration using Affine transformation.

- **3D Deconvolution**
  - `wiener`, `lucy`

- **Filters**
  - `mean_filter`, `meadian_filter`, `gaussian_filter`, `directional_median_filter` &rarr; Smoothing.
  - `dog_filter`, `doh_filter`, `log_filter` &rarr; Blob detection.
  - `sobel_filter`, `laplacian_filter` &rarr; Edge detection.
  - `std_filter`, `coef_filter` &rarr; Standard deviation based filtering.
  - `entropy_filter`, `enhance_contrast`, `gabor_filter` &rarr; Object detection etc.

- **Morphological Image Processing**
  - `erosion`, `dilation`, `opening`, `closing` &rarr; Basic ones.
  - `area_opening`, `area_closing`, `diameter_opening`, `diameter_closing` &rarr; Advanced ones.
  - `skeletonize`, `fill_hole` &rarr; Binary processing.
  - `count_neighbors` &rarr; For structure detection in binary images.
  - `remove_large_objects`, `remove_fine_objects` `remove_skeleton_structure` &rarr; Detect and remove objects.

- **Single Molecule Detection**
  - `find_sm`, `peak_local_max` &rarr; Return coordinates of single molecules.
  - `centroid_sm`, `gauss_sm`, `refine_sm` &rarr; Return coordinates in subpixel precision.

- **Background Correction**
  - `rolling_ball`, `tophat` &rarr; Background subtraction.
  - `gaussfit`, `gauss_correction` &rarr; Use Gaussian for image correction.

- **Labeling**
  - `label`, `label_if`, `label_threshold` &rarr; Labeling using binary images.
  - `specify` &rarr; Labeling around coordinates.
  - `append_label` &rarr; Label images.
  - `expand_labels`, `watershed`, `random_walker` &rarr; Adjuct or segment labels.

- **Feature Detection**
  - `hessian_eigval`, `hessian_eig` &rarr; Hessian.
  - `structure_tensor_eigval`, `structure_tensor_eig` &rarr; Structure tensor.

- **Filament Angle Estimation**
  - `hessian_angle` &rarr; Using Hessian eigenvector's orientations.
  - `gabor_angle` &rarr; Using Gabor filter's responses.

- **Property Measurement**
  - `regionprops` &rarr; Measure region properties such as mean intensity, Euler number, centroid, moment etc.
  - `lineprops`, `pointprops` &rarr; Measure line/point properties.

- **Texture Classification**
  - `lbp`, `glcm`, `glcm_props`

- **Others**
  - `focus_map` &rarr; Find focus using variance of Laplacian method. 
  - `stokes` &rarr; Analyze polarization using Stokes parameters.
  - `fft`, `ifft` &rarr; Fourier transformation.
  - `threshold` &rarr; Thresholding (many methods included).
  - `reslice` &rarr; Get line scan.
  - `crop_center`, `remove_edges` &rarr; Crop image.
  - `clip`, `rescale_intensity` &rarr; Rescale the intensity profile into certain range.
  - `proj` &rarr; Z-projection along any axis.
  - `split`, `split_pixel_unit` &rarr; Split the image.
  - `pad`, `defocus` &rarr; Padding.

## Brief Examples

#### 1. Input/Output and Visualization

```python
import impy as ip
img = ip.imread(r"...\images\XXX.tif")
img.gaussian_filter(sigma=1, update=True)
img.imsave("image_name")
```

```python
img.imshow() # matplotlib based visualization
ip.window.add(img) # send to napari
```

#### 2. Metadata and Axis-Targeted Slicing

Suppose an `np.ndarray` with shape (10, 20, 256, 256), which axis is time and which is z-slice? The file path of original image, what analysis have been applied are also confusing. `ImgArray` retains all the axis information and histories. You can use any character as axis symbols.

```python
img
```

    [Out]
        shape     : 10(t), 20(z), 256(y), 256(x)
      label shape : No label
        dtype     : uint16
      directory   : ...\images
    original image: XXX
       history    : gaussian_filter(sigma=1)

You can also access any parts of image with string that contains axis information.

```python
img1 = img["z=1;t=4,6,8"]
img1.axes = "p*@e"
img2 = img["z=3:6;t=2:15,18:-1"]
```

#### 3. Axis-Targeted Iteration

Usually we want to iterate analysis along random axes. `ImgArray` has `iter` method that simplify this process, which is similar to `groupby` function in `pandas`:

```python
for sl, img2d in img.iter("tzc"): # iterate along t, z and c axis
    # Here, img[sl] == img2d
    print(img2d.range) # do something
```

which is equivalent to something like ...

```C
for (t in t_all) {
    for (z in z_all) {
        for (c in c_all) {
            print(min(img[t,z,c]), max(img[t,z,c]))
        }
    }
}
```

#### 4. Labeling and Measurement

`scikit-image` has a powerful measurement function called `regionprops`. `ImgArray` also has a method that wrapped the `regionprops` function while enables multi-measurement.

```python
img.label_threshold(thr="yen") # Label image using Yen's thresholding
props = img.regionprops(properties=("mean_intensity", "perimeter")) # Measure mean intensity and perimeter for every labeled region
props.perimeter.plot() # Plot results of perimeter
props.perimeter["p=10;t=2"] # Get the perimeter of 10-th label in the slice t=2.
fit_result = props.mean_intensity.curve_fit(func) # curve fitting
```

## Basic Functions in impy

- `imread` = Load an image. `e.g. >>> ip.imread(path)`
- `imread_collection` = Load images recursively as a stack. `e.g. >>> ip.imread_collection(path, ignore_exception=True)`
- `read_meta` = Read metadata of a tiff file.
- `array`, `zeros`, `zeros_like`, `empty`, `empty_like` = similar to those in `numpy` but return `ImgArray`.
- `set_cpu` = Set the numbers of CPU used in image analysis.
- `stack` = Make a image stack from a list of images along any axis. ` e.g. >>> ip.stack(imglist, axis="c")`
- `window` &rarr; interface between `napari`. `ImgArray`, `Label`, `MarkerFrame` and `TrackFrame` can be sent to viewer with a simple code `ip.window.add(X)`.

## Common Attributes and Methods of Arrays

### Attributes

- `name` &rarr; name of the original image.
- `dirpath` &rarr; absolute path to the original image.
- `history` &rarr; history of applied analysis.
- `axes` &rarr; dimensions of image, `ptzcyx`-order.
- `scale` *property* &rarr; scales of each axis.
- `value` *property* &rarr; show the array in numpy format.
- `range` *property* &rarr; return a tuple of min/max of the image.
- `spatial_shape` *property* &rarr; such as `"yx"` or `"zyx"`.

### Basic Functions

- `imshow` &rarr; visualize 2-D or 3-D image.
- `imshow_label` &rarr; visualize 2-D or 3-D image and its labels.
- `imshow_comparewith` &rarr; compare two 2-D images.
- `hist` &rarr; show the histogram of image intensity profile.
- `imsave` &rarr; save image (by default save in the directory that the original image was loaded).
- `set_scale` &rarr; set scales of any axes.

## Data Type Conversion

`uint8`, `uint16`, `bool` and `float32` (sometimes `complex64`) are supported for type conversion.
- `as_uint8` &rarr; convert to `uint8`.
- `as_uint16` &rarr; convert to `uint16`.
- `as_float` &rarr; convert to `float32`.
- `as_img_type` &rarr; convert to any supported types.

## Automatic Saturation and Type Conversion

Overflow, underflow and type conversion is considered for operations `+`, `-`, `*` and `/`.
```python
# img = <uint16 image>

img + 10000     # pixel values larger than 65535 is substituted to 63353

img - 10000     # pixel values smaller than 0 is substituted to 0

img / 10        # output is converted to float32 
```

# References
For deconvolution, function `lucy` from Julia-coded package [Deconvolution.jl](https://github.com/JuliaDSP/Deconvolution.jl) is translated into Python.
