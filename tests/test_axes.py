import pytest
import impy as ip
import numpy as np

def test_axes():
    img = ip.random.random_uint8((10, 10, 10))
    img.axes = "zyx"
    img.set_scale(z=0.3)
    assert str(img.axes) == "zyx"
    assert "".join(img.scale.keys()) == "zyx"
    assert img.scale.z == 0.3
    
    img1 = img.gaussian_filter()
    assert str(img1.axes) == "zyx"
    assert "".join(img1.scale.keys()) == "zyx"
    assert img1.scale.z == 0.3
    
    img1.axes.replace("z", "t")
    assert img1.axes == "tyx"
    assert img.axes == "zyx"
        
    img2 = img.proj("y")
    assert str(img2.axes) == "zx"
    assert "".join(img2.scale.keys()) == "zx"
    assert img2.scale.z == 0.3

    
def test_set_scale():
    img = ip.random.random_uint8((10, 10, 10), axes="zyx")
    assert img.scale.z == img.scale.y == img.scale.x == 1
    img.scale = {"z": 0.5, "y": 0.4, "x": 0.4}
    assert img.scale.z == 0.5
    assert img.scale.y == img.scale.x == 0.4
    
    with pytest.raises(Exception):
        img.scale["t"] = 1
    with pytest.raises(Exception):
        img.scale.t = 1
    with pytest.raises(ValueError):
        img.scale["z"] = 0
    
    img.scale.z = 0.3
    assert img.scale.z == 0.3
        
    img1 = img.gaussian_filter()
    img1.scale.z = 0.4
    assert img.scale.z == 0.3
    assert img1.scale.z == 0.4
    
    img2 = img.binning(2)
    assert img2.scale.y == img2.scale.x == 0.8


def test_numpy():
    img = ip.random.random_uint8((10, 10, 10), axes="zyx")
    assert np.all(np.array(img.scale) == np.ones(3))

def test_slicing():
    img = ip.random.random_uint8((10, 10, 10, 10), axes="tzyx")
    assert img.axes == "tzyx"
    assert img[0].axes == "zyx"
    assert img[0, 0].axes == "yx"
    assert img[1, 1, 2].axes == "x"
    assert img[1, :, 2].axes == "zx"
    assert img[1, :, 5:7].axes == "zyx"
    assert img[:, 0].axes == "tyx"
    assert img[[1, 3, 5]].axes == "tzyx"
    assert img[5, [1, 3, 5]].axes == "zyx"
    
    # test array slicing
    sl = img[:, 0, 0, 0].value > 120
    assert img[sl].axes == "tzyx"
    assert img[0, sl].axes == "zyx"
    
    sl = img[:, :, 0, 0].value > 120
    assert img[sl].axes == "#yx"
    
    sl = img[:, :, :, 0].value > 120
    assert img[sl].axes == "#x"
    
    # test ellipsis
    assert img[..., 0].axes == "tzy"
    assert img[..., 0, :].axes == "tzx"
    assert img[0, ..., 0].axes == "zy"
    