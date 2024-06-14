import impy as ip
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import pytest

filters = [
    "median_filter", 
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

binary_filters = [
    "erosion",
    "dilation",
    "opening",
    "closing",
    "tophat",
    "diameter_opening",
    "diameter_closing",
    "area_opening",
    "area_closing",
]

dtypes = [np.uint8, np.uint16, np.float32]

path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
img_orig = ip.imread(path)

@pytest.mark.parametrize("f", filters)
@pytest.mark.parametrize("dtype", dtypes)
def test_filters(f, dtype, resource):
    if resource == "cupy" and f in ("entropy_filter", "enhance_contrast",):
        # Not implemented for GPU yet
        pytest.skip(reason="Not implemented for GPU yet")
    with ip.SetConst(RESOURCE=resource):
        img = img_orig["c=1;z=2"].astype(dtype)
        assert img.axes == "tyx"
        getattr(img, f)()

def test_fourier_filter(resource):
    with ip.SetConst(RESOURCE=resource):
        img = img_orig["c=1;z=2"]
        assert img.axes == "tyx"
        img.gaussian_filter(fourier=True)
        img.dog_filter(fourier=True)

@pytest.mark.parametrize("f", binary_filters)
def test_binary_filters(f, resource):
    thr = np.mean(img_orig["c=1;z=2"].range)
    with ip.SetConst(RESOURCE=resource):
        img = img_orig["c=1;z=2"] > thr
        assert img.axes == "tyx"
        getattr(img, f)()

@pytest.mark.parametrize("method", ["dog", "doh", "log", "ncc"])
def test_sm(method):
    img = img_orig["c=1"]
    img.find_sm(method=method, percentile=98)
    img.centroid_sm()


def test_binning():
    rng = ip.random.default_rng(1111)
    
    img = rng.normal(size=(120, 120, 120), axes="zyx")
    assert img.binning(4).shape == (30, 30, 30)
    assert img.binning(4, dims="yx").shape == (120, 30, 30)
    
    img = rng.normal(size=(120, 122, 123), axes="zyx")
    with pytest.raises(ValueError):
        img.binning(4)
    imgb = img.binning(4, check_edges=False)
    assert imgb.shape == (30, 30, 30)
    assert_allclose(imgb, img[:120, :120, :120].binning(4))
    
    np.random.seed()

def test_tiled(resource):
    with ip.SetConst(RESOURCE=resource):
        rng = ip.random.default_rng(1111)
        
        img = rng.random(size=(120, 120, 120), axes="zyx")
        img.tiled(chunks=(40, 50, 50)).lowpass_filter()
        img.tiled(chunks=(40, 50, 50)).gaussian_filter(sigma=1.0)
        img.tiled(chunks=(40, 50, 50)).gaussian_filter(sigma=1.0, fourier=True)
        img.tiled(chunks=(40, 50, 50)).dog_filter(low_sigma=1.0)
        img.tiled(chunks=(40, 50, 50)).dog_filter(low_sigma=1.0, fourier=True)
        img.tiled(chunks=(40, 50, 50)).log_filter(sigma=1.0)

def test_lazy_tiled(resource):
    with ip.SetConst(RESOURCE=resource):
        rng = ip.lazy.random.default_rng(1111)
        
        img = rng.random(size=(120, 120, 120), axes="zyx")
        img.tiled(chunks=(40, 50, 50)).lowpass_filter()
        img.tiled(chunks=(40, 50, 50)).gaussian_filter(sigma=1.0)
        img.tiled(chunks=(40, 50, 50)).gaussian_filter(sigma=1.0, fourier=True)
        img.tiled(chunks=(40, 50, 50)).dog_filter(low_sigma=1.0)
        img.tiled(chunks=(40, 50, 50)).dog_filter(low_sigma=1.0, fourier=True)
        img.tiled(chunks=(40, 50, 50)).log_filter(sigma=1.0)

@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.int16, np.uint8, np.uint16])
def test_tiled_dtype(resource, dtype):
    with ip.SetConst(RESOURCE=resource):
        rng = ip.random.default_rng(1111)
        
        img = rng.random(size=(120, 120, 120), axes="zyx").astype(dtype)
        out = img.tiled(chunks=(40, 50, 50)).lowpass_filter()
        assert out.dtype == dtype
        
        rng = ip.lazy.random.default_rng(1111)
        
        img = rng.random(size=(120, 120, 120), axes="zyx").astype(dtype)
        out = img.tiled(chunks=(40, 50, 50)).lowpass_filter()
        assert out.dtype == dtype
        

@pytest.mark.parametrize("order", [1, 3])
def test_drift_correction(order: int):
    img = ip.random.normal(size=(5, 10, 3, 120, 120), axes="tzcyx")
    img["z=0;c=0"].drift_correction(along="t", order=order)
    img.drift_correction(along="t", ref=img["c=0"], dims="zyx")
    img.drift_correction(along="t")
    img.drift_correction(ref=img["c=0"], along="t")

def test_hessian_angle():
    img = ip.asarray(np.eye(32, dtype=np.float32)).gaussian_filter(3)
    ang = img.hessian_angle(deg=True)
    assert abs(ang[15, 15] + 45) < 1e-3
    assert abs(ang[16, 16] + 45) < 1e-3

@pytest.mark.parametrize("method", ["sobel", "farid", "scharr", "prewitt"])
def test_edge_grad(method):
    img = ip.asarray(np.eye(32, dtype=np.float32)).gaussian_filter(3)
    ang = img.edge_grad(method=method, deg=True)
    assert abs(ang[14, 12] - 45) < 1e-3
    assert abs(ang[18, 16] - 45) < 1e-3
    assert abs(ang[12, 14] + 135) < 1e-3
    assert abs(ang[16, 18] + 135) < 1e-3

def test_labeling():
    img = ip.zeros((32, 32), dtype=np.uint8)
    img[20:25, 18:28] = 1
    img[4:12, 8:16] = 1
    img[10:22, 14:26] = 2
    
    img0 = img.copy()
    img0.label()
    assert img0.labels is not None
    assert img0.labels.max() == 3
    
    img0 = img.copy()
    lbl = img0 > 0.5
    img0.label(lbl)
    assert img0.labels is not None
    assert img0.labels.max() == 1

def test_smooth_mask():
    img = ip.circular_mask(4, (19, 19))
    count = np.sum(img)
    assert_allclose(img.smooth_mask(sigma=0, dilate_radius=0), img)
    assert np.sum(img.smooth_mask(sigma=1, dilate_radius=1) > 0.5) > count
    assert np.sum(img.smooth_mask(sigma=0, dilate_radius=-1) > 0.5) < count
