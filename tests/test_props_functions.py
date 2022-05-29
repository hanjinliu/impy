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

def test_pathprops():
    img = ip.zeros((4, 10, 12, 14), axes="tzyx")
    
    # 2-D
    coords = [
        np.array([[2, 2], [3.4, 4.2], [7.5, 8.7]]),
        np.array([[1, 2.1], [5.4, 3.2]]),
    ]
    
    out = img.pathprops(coords, [np.mean, np.std])
    assert len(out) == 2
    out["mean"].shape == (2, 4, 10)
    out["std"].shape == (2, 4, 10)
    
    # 3-D
    coords = [
        np.array([[1.2, 2, 2], [1.4, 3.4, 4.2], [5.5, 7.5, 8.7]]),
        np.array([[3.4, 1, 2.1], [3, 5.4, 3.2]]),
    ]
    
    out = img.pathprops(coords, [np.mean, np.std])
    assert len(out) == 2
    out["mean"].shape == (2, 10)
    out["std"].shape == (2, 10)
