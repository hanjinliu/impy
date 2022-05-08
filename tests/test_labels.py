import numpy as np
from numpy.testing import assert_equal
import impy as ip
import pytest
from impy.axes import ImageAxesError

def test_inherit_label():
    img = ip.zeros((10, 10, 10), axes="zyx")
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[:4] = 1
    label[4:6] = 2
    img.labels = label
    
    out = img.gaussian_filter()
    assert_equal(out.labels, img.labels)
    assert out.labels is not img.labels
    assert out.labels.axes == "zyx"
    
    out = img + 1
    assert_equal(out.labels, img.labels)
    assert out.labels is not img.labels
    assert out.labels.axes == "zyx"
    
    out = img.proj("z")
    assert out.labels is None
    
def test_lower_dim():
    img = ip.zeros((10, 10, 10), axes="zyx")
    label = np.zeros((10, 10), dtype=np.uint8)
    label[:4] = 1
    label[4:6] = 2
    img.labels = label
    
    assert img.labels.axes == "yx"
    
    out = img.gaussian_filter()
    assert_equal(out.labels, img.labels)
    assert out.labels is not img.labels
    assert out.labels.axes == "yx"
    
    out = img + 1
    assert_equal(out.labels, img.labels)
    assert out.labels is not img.labels
    assert out.labels.axes == "yx"
    
    out = img.proj("z")
    assert_equal(out.labels, img.labels)
    assert out.labels is not img.labels
    assert out.labels.axes == "yx"
    
    out = img.proj("y")
    assert out.labels is None

def test_getitem():
    img = ip.zeros((10, 10, 10), axes="zyx")
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[:4] = 1
    label[4:6] = 2
    label[:, 3:8] = 3
    img.labels = label
    
    out = img[2:7, 1:-1, 3:6]
    assert out.labels.axes == out.axes == "zyx"
    assert out.labels.shape == out.shape
    assert_equal(out.labels, img.labels[2:7, 1:-1, 3:6])
    
    
    out = img[2:7, [0, 2, 4, 6, 9]]
    assert out.labels.axes == out.axes == "zyx"
    assert out.labels.shape == out.shape
    assert_equal(out.labels, img.labels[2:7, [0, 2, 4, 6, 9]])

    
def test_getitem_lower_dim():
    img = ip.zeros((10, 10, 10), axes="zyx")
    label = np.zeros((10, 10), dtype=np.uint8)
    label[:4] = 1
    label[4:6] = 2
    label[:, 3:8] = 3
    img.labels = label
    
    out = img[2:7, 1:-1, 3:6]
    assert out.labels.axes == "yx"
    assert out.labels.shape == out.shape[1:]
    assert_equal(out.labels, img.labels[1:-1, 3:6])
    
    
    out = img[2:7, [0, 2, 4, 6, 9]]
    assert out.labels.axes == "yx"
    assert out.labels.shape == out.shape[1:]
    assert_equal(out.labels, img.labels[[0, 2, 4, 6, 9]])


def test_error():
    img = ip.zeros((10, 10, 10), axes="zyx")
    with pytest.raises(ValueError):
        img.labels = img
    with pytest.raises(TypeError):
        img.labels = np.zeros((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        img.labels = np.zeros((10, 10, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        img.labels = np.zeros((10, 8), dtype=np.uint8)
    with pytest.raises(ImageAxesError):
        img.labels = np.zeros((2, 10, 10, 10), dtype=np.uint8)