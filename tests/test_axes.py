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