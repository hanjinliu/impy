import impy as ip
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import pytest

filters = ["median_filter", 
           "gaussian_filter", 
           "lowpass_filter", 
           "lowpass_conv_filter",
           "highpass_filter",
           "erosion",
           "dilation",
           "opening",
           "closing",
           "tophat",
           "mean_filter",
           "std_filter",
           "coef_filter",
           "diameter_opening",
           "diameter_closing",
           "area_opening",
           "area_closing",
           "entropy_filter",
           "enhance_contrast",
           "laplacian_filter",
           "kalman_filter",
           "dog_filter",
           "doh_filter",
           "log_filter",
           "rolling_ball",
           ]

dtypes = [np.uint8, np.uint16, np.float32]

path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
img_orig = ip.imread(path)

@pytest.mark.parametrize("f", filters)
@pytest.mark.parametrize("dtype", dtypes)
def test_filters(f, dtype, resource):
    with ip.SetConst(RESOURCE=resource):
        img = img_orig["c=1;z=2"].astype(dtype)
        assert img.axes == "tyx"
        getattr(img, f)()

@pytest.mark.parametrize("method", ["dog", "doh", "log", "ncc"])
def test_sm(method):
    img = img_orig["c=1"]
    img.find_sm(method=method, percentile=98)
    img.centroid_sm()


def test_binning():
    np.random.seed(1111)
    
    img = ip.random.normal(size=(120, 120, 120), axes="zyx")
    assert img.binning(4).shape == (30, 30, 30)
    assert img.binning(4, dims="yx").shape == (120, 30, 30)
    
    img = ip.random.normal(size=(120, 122, 123), axes="zyx")
    with pytest.raises(ValueError):
        img.binning(4)
    imgb = img.binning(4, check_edges=False)
    assert imgb.shape == (30, 30, 30)
    assert_allclose(imgb, img[:120, :120, :120].binning(4))
    
    np.random.seed()

def test_tiled():
    np.random.seed(1111)
    
    img = ip.random.normal(size=(120, 120, 120), axes="zyx")
    img.tiled_lowpass_filter(chunks=(40, 50, 50))

def test_drift_correction():
    img = ip.random.normal(size=(5, 10, 3, 120, 120), axes="tzcyx")
    img["z=0;c=0"].drift_correction(along="t")
    img.drift_correction(along="t", ref=img["c=0"], dims="zyx")
    img.drift_correction(along="t")
    img.drift_correction(ref=img["c=0"], along="t")
