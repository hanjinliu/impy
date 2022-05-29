import numpy as np
import impy as ip

def test_pointprops():
    img2d = ip.zeros((20, 20), axes="yx")
    img = ip.zeros((10, 20, 20), axes="tyx")
    assert img2d.pointprops([3, 3]) == 0.
    assert img.pointprops([3, 3]).shape == (10,)
    out = img.pointprops([[3, 3], [6, 3], [3, 6], [6, 6]])
    assert out.shape == (4, 10)  # N, t

def test_reslice():
    img = ip.zeros((4, 10, 12, 14), axes="tzyx")
    coords = np.array([[2, 2], [4, 4], [6, 4]])  # length = 4.84 => 5 sample points
    out = img.reslice(coords)
    assert out.shape == (4, 10, 5)
    assert out.axes == ["t", "z", "s"]
    
    coords = np.array([[3, 2, 2], [3, 4, 4], [3, 6, 4]])
    out = img.reslice(coords)
    assert out.shape == (4, 5)
    assert out.axes == ["t", "s"]
