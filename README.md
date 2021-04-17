# impy

## More Numpy in image analysis! 

ImageJ is generally used for image analysis especially in biological backgrounds. However, recent demands for batch analysis, machine learning and high reproducibility usually do not suit for ImageJ. On the other hand, the famous image analysis toolkit, [scikit-image](https://github.com/scikit-image/scikit-image), is not convenient for biological multi-dimensional analysis, although it is the best practice for above-mentioned problems.

Here with `ImgArray`, this module solved major problems that happens when you code image analysis in Python. Because axial information such as xy plane, channels and time are also included in the arrays, many functions can automatically optimize multi-dimensional image analysis such as filtering, background subtraction and deconvolution.

## Brief Examples

#### 1. Input/Output

```python
import impy as ip
img = ip.imread(r"...\images\XXX.tif")
img.gaussian_filter(sigma=1, update=True)
img
```
    [Out]
        shape     : 10(t), 3(c), 512(y), 512(x)
      label shape : No label
        dtype     : uint16
      directory   : ...\images
    original image: XXX
       history    : gaussian_filter(sigma=1)

```python
img.imsave("image_name")
```

#### 2. Visualization

```python
img.imshow()
img.imshow_comparewith(another_img)
```

#### 3. Axis-Targeted Slicing

```python
img_new = img["c=1;t=4,8,12"]
```

#### 4. Axis-Targeted Iteration

```python
for sl, img2d in img.iter("tzc"):
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

#### 5. Labeling and Measurement

```python
img.label_threshold(thr="yen") # Label image using Yen's thresholding
props = img.regionprop(properties=("mean_intensity", "perimeter")) # Measure mean intensity and perimeter for every labeled region
props.perimeter.plot_profile() # Plot results of perimeter
props.perimeter["t=2;p=10"] # Get the perimeter of 10-th label in the slice t=2.
```

## Basic Usage

Load image with `imread()` function. `ImgArray` object is created.

```python
import impy as ip

# load single tif
img = ip.imread(r"C:\Users\...\XXX.tif")

# load tifs recursively from a directory
img = ip.imread_collection(r"C:\Users\...\XX_100nM", ignore_exception=True)
```

Stacking images with `stack()`.

```python
# make stack along channel axis
img = ip.stack([img1, img2], axis="c", dtype="uint16") 
```

Making synthetic three-channel image with `array()`.

```python
img = ip.array(np.random.rand(3*40*30).reshape(3,40,30)*100, name="random noise")
```

## Basic Attributes and Functions of ImgArray

### Attributes

- `name` = name of the original image.
- `dirpath` = absolute path to the original image.
- `history` = history of applied analysis.
- `axes` = dimensions of image, `ptzcyx`-order.
- `value` (property) = show the array in numpy format.
- `range` (property) = return a tuple of min/max.
- `spatial_shape` (property) = such as `"yx"` or `"zyx"`.

### Functions

- `imshow` = visualize 2-D or 3-D image.
- `imshow_label` = visualize 2-D or 3-D image and its labels.
- `imshow_comparewith` = compare two 2-D images.
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

## Axis-Targeted Slicing

When you want to access the first channel and 4-th to 10-th time points, you can do it by:

```python
img["c=1;t=4-10"]       # get items
img["c=1;t=4-10"] = 0   # set items
```

List-like slicing is also supported:

```python
img["t=1,3-6,9"]  # this means [0,2,3,4,5,8] in t-axis
```

You can define your own axes such as:

```python
img.axes = "aoe"
```


## Image Analysis

`ImgArray` has a lot of member functions for image analysis. Some of them supports multiprocessing.

- `drift_correction` (plugin) = automatic drift correction using `phase_cross_correlation` function in skimage.
- `lucy` (plugin) = deconvolution of images.
- `affine_correction` = Correction of such as chromatic aberration using Affine transformation.
- `hessian_eigval`, `hessian_eig` = feature detection using Hessian method.
- `structure_tensor_eigval`, `structure_tensor_eig` = feature detection using structure tensor.
- `dog_filter` = filtering using difference of Gaussian method.
- `mean_filter`, `meadian_filter`, `gaussian_filter` = for 2-D or 3-D smoothing.
- `sobel_filter` = for edge detection.
- `entropy_filter` = for object detection.
- `enhance_contrast` = for higher contrast.
- `erosion`, `dilation`, `opening`, `closing` = for morphological processing.
- `rolling_ball`, `tophat` = for background subtraction.
- `gaussfit`, `gaussfit_particle` = fit the image to 2-D Gaussian (for correction of uneven irradiation or single molecular analysis).
- `distance_map`, `skeletonize` = processing binary images.
- `fft`, `ifft` = Fourier transformation.
- `threshold` = thresholding.
- `peak_local_max` = find maxima.
- `fill_hole` = fill holl-like region.
- `label`, `label_threshold` = labeling images.
- `expand_labels`, `watershed` = adjuct labels.
- `regionprops` =  measure properties on labels.
- `profile_line` = get line scan.
- `crop_center`, `crop_circle` = crop image.
- `clip_outliers`, `rescale_intensity` = rescale the intensity profile into certain range.
- `proj` = Z-projection along any axis.
- `split` = split the image along any axis.

# References
For 3-D PSF generation, [flowdec](https://github.com/hammerlab/flowdec) is imported in this package. For deconvolution, function `lucy` from Julia-coded package [Deconvolution.jl](https://github.com/JuliaDSP/Deconvolution.jl) is translated into Python.
