[![Downloads](https://pepy.tech/badge/impy-array/month)](https://pepy.tech/project/impy-array)
[![PyPI version](https://badge.fury.io/py/impy-array.svg)](https://badge.fury.io/py/impy-array)

# A numpy extension for efficient and powerful image analysis workflow

`impy` is an all-in-one image analysis library, equipped with parallel processing, GPU support, GUI based tools and so on.

The core array, `ImgArray`, is a subclass of `numpy.ndarray`, tagged with information such as 

- image axes
- scale of each axis
- directory of the original image
- and other image metadata

## Documentation

Documentation is available [here](https://hanjinliu.github.io/impy/).

## Installation

- use pip

```
pip install impy-array
```

- from source

```
git clone https://github.com/hanjinliu/impy
```

### Code as fast as you speak

Almost all the functions, such as filtering, deconvolution, labeling, single molecule detection, and even those pure `numpy` functions, are aware of image metadata. They "know" which dimension corresponds to `"z"` axis, which axes they should iterate along or where to save the image. As a result, **your code will be very concise**:

```python
import impy as ip
import numpy as np

img = ip.imread("path/to/image")       # Read images with metadata.
img["z=3;t=0"].imshow()                # Plot image slice at z=3 and t=0.
img_fil = img.gaussian_filter(sigma=2) # Paralell batch denoising. No more for loop!
img_prj = np.max(img_fil, axis="z")    # Z-projection (numpy is aware of image axes!).
img_prj.imsave(f"Max-{img.name}")      # Save in the same place. Don't spend time on searching for the directory!
```

### Supported file formats

`impy` automatically chooses proper reader/writer according to the extension.

- Tiff file (".tif", ".tiff")
- MRC file (".mrc", ".rec", ".st", ".map", ".map.gz")
- Zarr file (".zarr")
- Other image file (".png", ".jpg")

### Switch between CPU and GPU

`impy` can internally switches the functions between `numpy` and `cupy`.
You can use GPU for calculation very easily.

```python
img.gaussian_filter()  # <- CPU
with ip.use("cupy"):
    img.gaussian_filter()  # <- GPU
ip.Const["RESOURCE"] = "cupy"  # <- globally use GPU
```

### Seamless interface between `napari`

[napari](https://github.com/napari/napari) is an interactive viewer for multi-dimensional images. `impy` has a **simple and efficient interface** with it, via the object `ip.gui`. Since `ImgArray` is tagged with image metadata, you don't have to care about axes or scales. Just run 

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

`impy` also supports command line based image analysis. All method of `ImgArray` is available
from commad line, such as

```powershell
impy path/to/image.tif ./output.tif --method gaussian_filter --sigma 2.0
```

which is equivalent to

```python
import impy as ip
img = ip.imread("path/to/image.tif")
out = img.gaussian_filter(sigma=2.0)
out.imsave("./output.tif")
```

For more complex procedure, it is possible to send image directly to `IPython`

```
impy path/to/image.tif -i
```
```python
thr = img.gaussian_filter().threshold()
```
