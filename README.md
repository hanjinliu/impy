[![BSD 3-Clause License](https://img.shields.io/pypi/l/impy-array.svg?color=green)](https://github.com/hanjinliu/impy/blob/main/LICENSE)
[![Python package index download statistics](https://img.shields.io/pypi/dm/impy-array.svg)](https://pypistats.org/packages/impy-array)
[![PyPI version](https://badge.fury.io/py/impy-array.svg)](https://badge.fury.io/py/impy-array)

# impy

`impy` is an all-in-one multi-dimensional image analysis library. The core array,
`ImgArray`, is a subclass of `numpy.ndarray`, tagged with information such as:

- image axes
- scale of each axis
- directory of the original image
- and other image metadata

## Documentation

Documentation is available [here](https://hanjinliu.github.io/impy/).

## Installation

- use pip

``` sh
pip install impy-array
pip install impy-array[tiff]    # with supports for reading/writing .tif files
pip install impy-array[mrc]     # with supports for reading/writing .mrc files
pip install impy-array[napari]  # viewer support
pip install impy-array[all]     # install everything
```

- from source

```
git clone https://github.com/hanjinliu/impy
```

### Code as fast as you speak

Almost all the functions, such as filtering, deconvolution, labeling, single molecule
detection, and even those pure `numpy` functions, are aware of image metadata. They
"know" which dimension corresponds to `"z"` axis, which axes they should iterate along
or where to save the image. As a result, **your code will be very concise**:

```python
import impy as ip
import numpy as np

img = ip.imread("path/to/image.tif")   # Read images with metadata.
img["z=3;t=0"].imshow()                # Plot image slice at z=3 and t=0.
img["y=N//4:N//4*3"].imshow()          # `N` for the size of the axis.
img_fil = img.gaussian_filter(sigma=2) # Paralell batch denoising. No more for loop!
img_prj = np.max(img_fil, axis="z")    # Z-projection (numpy is aware of image axes!).
img_prj.imsave("image_max.tif")        # Save in the same place. Don't spend time on searching for the directory!
```

### Supports many file formats

`impy` automatically chooses the proper reader/writer according to the extension.

- Tiff file (".tif", ".tiff")
- LSM file (".lsm")
- MRC file (".mrc", ".rec", ".st", ".map", ".map.gz")
- Zarr file (".zarr")
- ND2 file (".nd2")
- Other image file (".png", ".jpg")

### Lazy loading

With the `lazy` submodule, you can easily make image processing workflows for large
images.

```python
import impy as ip

img = ip.lazy.imread("path/to/very-large-image.tif")
out = img.gaussian_filter()
out.imsave("image_filtered.tif")
```

### Switch between CPU and GPU

`impy` can internally switches the functions between `numpy` and `cupy`.

```python
img.gaussian_filter()  # <- CPU
with ip.use("cupy"):
    img.gaussian_filter()  # <- GPU
ip.Const["RESOURCE"] = "cupy"  # <- globally use GPU
```

### Seamless interface between `napari`

[napari](https://github.com/napari/napari) is an interactive viewer for multi-dimensional
images. `impy` has a **simple and efficient interface** with it, via the object `ip.gui`.
Since `ImgArray` is tagged with image metadata, you don't have to care about axes or
scales. Just run

```python
ip.gui.add(img)
```

### Extend your function for batch processing

Already have a function for `numpy` and `scipy`? Decorate it with `@ip.bind`

```python
@ip.bind
def imfilter(img, param=None):
    # Your function here.
    # Do something on a 2D or 3D image and return image, scalar or labels
    return out
```

and it's ready for batch processing!

```python
img.imfilter(param=1.0)
```

### Command line usage

`impy` also supports command-line-based image analysis. All methods of `ImgArray` are
available from the command line, such as

```shell
impy path/to/image.tif ./output.tif --method gaussian_filter --sigma 2.0
```

which is equivalent to

```python
import impy as ip
img = ip.imread("path/to/image.tif")
out = img.gaussian_filter(sigma=2.0)
out.imsave("./output.tif")
```

For more complex procedures, it is possible to send images directly to `IPython`

```
impy path/to/image.tif -i
```

```python
thr = img.gaussian_filter().threshold()
```
