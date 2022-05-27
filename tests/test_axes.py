import pytest
import impy as ip
import numpy as np
from impy.axes import ImageAxesError, broadcast

@pytest.mark.parametrize("axes", [["t", "z", "y", "x"], ["time", "z", ":y", ":x"]])
def test_axes(axes):
    img = ip.random.random_uint8((10, 10, 10))
    img.axes = axes[1:]
    
    tyx = axes[0:1] + axes[2:]
    zyx = axes[1:]
    zx = axes[1:2] + axes[3:4]
    
    img.set_scale({axes[1]: 0.3})
    assert img.axes == zyx
    assert list(img.scale.keys()) == zyx
    assert img.axes[axes[1]].scale == 0.3
    
    img1 = img.gaussian_filter(dims=axes[1:])
    assert img1.axes == zyx
    assert list(img1.scale.keys()) == zyx
    assert img1.axes[axes[1]].scale == 0.3
    
    img1.axes.replace(axes[1], axes[0])
    assert img1.axes == tyx
    assert img.axes == zyx
        
    img2 = img.proj(axes[2:3])
    assert img2.axes == zx
    assert list(img2.scale.keys()) == zx
    assert img2.axes[axes[1]].scale == 0.3

@pytest.mark.parametrize("axes", [["t", "z", "y", "x"], ["time", "z", ":y", ":x"]])
def test_set_axes(axes):
    img = ip.random.random_uint8((10, 10, 10), axes="zyx")
    tyx = axes[0:1] + axes[2:]
    img.axes = tyx
    assert img.axes == tyx
    with pytest.raises(ImageAxesError):
        img.axes = axes[-1:]
    assert img.axes == tyx
    img.axes = None
    assert str(img.axes) == "###"

    
def test_set_scale():
    img = ip.random.random_uint8((10, 10, 10), axes="zyx")
    assert img.scale.z == img.scale.y == img.scale.x == 1
    img.scale = {"z": 0.5, "y": 0.4, "x": 0.4}
    assert img.scale.z == 0.5
    assert img.scale.y == img.scale.x == 0.4
    
    with pytest.raises(Exception):
        img.scale["t"] = 1  # cannot set scale to an axis that image does not have.
    with pytest.raises(Exception):
        img.scale.t = 1  # cannot set scale to an axis that image does not have.
    with pytest.raises(ValueError):
        img.scale["z"] = 0  # cannot set zero
    
    img.scale.z = 0.3
    assert img.scale.z == 0.3
        
    img1 = img.gaussian_filter()
    img1.scale.z = 0.4
    assert img.scale.z == 0.3
    assert img1.scale.z == 0.4
    
    img2 = img.binning(2)
    assert img2.scale.y == img2.scale.x == 0.8

@pytest.mark.parametrize("axes", [["z", "y", "x"], [":z", ":y", ":x"]])
def test_numpy(axes):
    img = ip.random.random_uint8((10, 10, 10), axes=axes)
    assert np.all(np.array(img.scale) == np.ones(3))

@pytest.mark.parametrize("axes", [["t", "z", "y", "x"], ["time", ":z", ":y", ":x"]])
def test_slicing(axes):
    img = ip.random.random_uint8((10, 10, 10, 10), axes=axes)
    tzyx = axes
    tyx = axes[0:1] + axes[2:]
    zyx = axes[1:]
    yx = axes[2:]
    zx = axes[1:2] + axes[3:4]
    _yx = ["#"] + yx
    _zx = ["#"] + zx
    
    assert img.axes == tzyx
    assert img[0].axes == zyx
    assert img[0, 0].axes == yx
    assert img[1, 1, 2].axes == axes[3]
    assert img[1, :, 2].axes == zx
    assert img[1, :, 5:7].axes == zyx
    assert img[:, 0].axes == tyx
    assert img[[1, 3, 5]].axes == tzyx
    assert img[5, [1, 3, 5]].axes == zyx
    assert img[[1, 2, 3], [1, 2, 3]].axes == _yx
    assert img[[1, 2, 3], :, [1, 2, 3]].axes == _zx
    assert img[:, [1, 2, 3], :, [1, 2, 3]].axes == tzyx[0:1] + ["#"] + tzyx[2:3]
    assert img[[1, 3, 5], [1, 2, 3], :, [1, 2, 3]].axes == ["#"] + tzyx[2:3]
    
    # test new axis
    assert img[np.newaxis].axes == ["#"] + tzyx
    assert img[:, :, np.newaxis].axes == axes[0:2] + ["#"] + axes[2:]
    assert img[np.newaxis, :, np.newaxis].axes == ["#", axes[0], "#"] + zyx
    
    # test array slicing
    sl = img[:, 0, 0, 0].value > 120
    assert img[sl].axes == tzyx
    assert img[0, sl].axes == zyx
    
    sl = img[:, :, 0, 0].value > 120
    assert img[sl].axes == _yx
    
    sl = img[:, :, :, 0].value > 120
    assert img[sl].axes == ["#"] + tzyx[3:4]
    
    # test ellipsis
    assert img[..., 0].axes == tzyx[:-1]
    assert img[..., 0, :].axes == tzyx[:2] + tzyx[3:4]
    assert img[0, ..., 0].axes == tzyx[1:3]

def test_axis_labels():
    img = ip.zeros((4, 10, 10), axes="cyx")
    with pytest.raises(ValueError):
        img.set_axis_label(c=["c0", "c1", "c2"])
    img.set_axis_label(c=["c0", "c1", "c2", "c3"])
    assert img.axes["c"].labels == ("c0", "c1", "c2", "c3")
    assert img[:2].axes["c"].labels == ("c0", "c1")
    assert img[2:].axes["c"].labels == ("c2", "c3")
    assert img[[0, 2]].axes["c"].labels == ("c0", "c2")

def test_broadcast():
    assert broadcast("zyx", "tzyx") == "tzyx"
    assert broadcast("yx", "tzyx") == "tzyx"
    assert broadcast("z", "tzyx") == "tzyx"
    assert broadcast("tzcyx", "tyx") == "tzcyx"
    assert broadcast("tzyx", "tcyx") == "tzcyx"
    assert broadcast("yx", "xy") == "yx"
    assert broadcast("tyx", "xy") == "tyx"
    assert broadcast("y", "x") == "yx"
    assert broadcast("z", "y", "x") == "zyx"
    assert broadcast("dz", "dy", "dx") == "dzyx"
    assert broadcast("tzyx", "tyx", "tzyx", "yzx") == "tzyx"
