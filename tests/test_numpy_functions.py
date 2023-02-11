import pytest
import impy as ip
from impy.axes import ImageAxesError
import numpy as np
from numpy.testing import assert_allclose

def test_numpy_function():
    np.random.seed(1234)
    img = ip.random.random_uint16((10, 100, 100))
    assert img.axes == "tyx"
    assert isinstance(np.mean(img), float)
    proj = np.mean(img, axis="y")
    assert_allclose(proj, np.mean(img.value, axis=1))
    np.random.seed()

def test_squeeze():
    img0 = ip.zeros((3, 3, 1), axes="zyx")
    out = np.squeeze(img0)
    assert out.axes == ["z", "y"]
    assert out.shape == (3, 3)
    
def test_stack():
    img0 = ip.zeros((3, 3))
    img1 = ip.zeros((3, 3))
    
    out = np.stack([img0, img1], axis=0)
    assert out.axes == ["#", "y", "x"]
    assert out.shape == (2, 3, 3)
    
    out = np.stack([img0, img1], axis=1)
    assert out.axes == ["y", "#", "x"]
    assert out.shape == (3, 2, 3)
    
    out = np.stack([img0, img1], axis="p")
    assert out.axes == ["p", "y", "x"]
    assert out.shape == (2, 3, 3)
    
    with pytest.raises(ImageAxesError):
        np.stack([img0, img1], axis="y")

def test_concatenate():
    img0 = ip.zeros((3, 2))
    img1 = ip.zeros((3, 2))
    
    out = np.concatenate([img0, img1], axis=0)
    assert out.axes == ["y", "x"]
    assert out.shape == (6, 2)
    
    out = np.concatenate([img0, img1], axis=1)
    assert out.axes == ["y", "x"]
    assert out.shape == (3, 4)
    
    out = np.concatenate([img0, img1], axis="y")
    assert out.axes == ["y", "x"]
    assert out.shape == (6, 2)
    
    with pytest.raises(ImageAxesError):
        np.concatenate([img0, img1], axis="z")
    
def test_expand_dims():
    img0 = ip.zeros((3, 3))
    out = np.expand_dims(img0, axis=0)
    assert out.axes == ["#", "y", "x"]
    assert out.shape == (1, 3, 3)
    
    out = np.expand_dims(img0, axis=1)
    assert out.axes == ["y", "#", "x"]
    assert out.shape == (3, 1, 3)
    
    out = np.expand_dims(img0, axis="z")
    assert out.axes == ["z", "y", "x"]
    assert out.shape == (1, 3, 3)
    
    with pytest.raises(ImageAxesError):
        np.expand_dims(img0, axis="y")

def test_transpose():
    img0 = ip.zeros((2, 3, 4), axes="zyx")
    
    out = np.transpose(img0, "zxy")
    assert out.axes == ["z", "x", "y"]
    assert out.shape == (2, 4, 3)
    
    out = np.transpose(img0, "yxz")
    assert out.axes == ["y", "x", "z"]
    assert out.shape == (3, 4, 2)
    
    out = np.transpose(img0)
    assert out.axes == ["x", "y", "z"]
    assert out.shape == (4, 3, 2)

def test_broadcast_to():
    img0 = ip.zeros((2, 3), axes="yx")
    
    out = np.broadcast_to(img0, (4, 2, 3))
    assert out.axes == ["#", "y", "x"]
    assert out.shape == (4, 2, 3)
    
    out = np.broadcast_to(img0, (6, 4, 2, 3))
    assert out.axes == ["#", "#", "y", "x"]
    assert out.shape == (6, 4, 2, 3)

def test_moveaxis():
    img0 = ip.zeros((4, 3, 2, 5), axes="tzyx")
    
    out = np.moveaxis(img0, 0, 2)
    assert out.axes == ["z", "y", "t", "x"]
    assert out.shape == np.moveaxis(img0.value, 0, 2).shape
    
    out = np.moveaxis(img0, [0, 2], [1, 3])
    assert out.axes == ["z", "t", "x", "y"]
    assert out.shape == np.moveaxis(img0.value, [0, 2], [1, 3]).shape

def test_swapaxes():
    img0 = ip.zeros((4, 3, 2, 5), axes="tzyx")
    
    out = np.swapaxes(img0, 0, 2)
    assert out.axes == ["y", "z", "t", "x"]
    assert out.shape == np.swapaxes(img0.value, 0, 2).shape
    
    out = np.swapaxes(img0, "x", "z")
    assert out.axes == ["t", "x", "y", "z"]
    assert out.shape == np.swapaxes(img0.value, 3, 1).shape

def test_indices():
    inds = ip.indices((4, 5, 6), axes="zyx")
    inds_np = np.indices((4, 5, 6))
    
    assert_allclose(inds.z , inds_np[0])
    assert_allclose(inds.y , inds_np[1])
    assert_allclose(inds.x , inds_np[2])

@pytest.mark.parametrize("shape", [(3, 4, 5), (5,)])
def test_arg(shape):
    img = ip.zeros(shape)
    np.argmax(img, axis=0)
    np.argmax(img, axis="x")
    np.argmin(img, axis=0)
    np.argmin(img, axis="x")
