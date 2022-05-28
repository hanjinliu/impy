import numpy as np
import impy as ip
from impy.axes import ImageAxesError
import pytest

def test_operators():
    arr = ip.random.normal(size=(3, 4, 10), axes="tyx")
    fit = ip.asarray(np.arange(1, 41).reshape(4, 10), axes="yx")
    fit0 = np.stack([fit]*3, axis="t")
    assert np.all(arr/fit == arr/fit0)
    assert np.all(arr[fit>12] == arr[fit0>12])

def test_mismatch():
    img0 = ip.zeros((3, 3), axes="yx")
    img1 = ip.zeros((3, 3), axes="zy")
    
    with pytest.raises(ImageAxesError):
        img0 + img1

def test_broadcasting_arrays():
    out = ip.broadcast_arrays(
        ip.zeros((3, 4, 5), axes="zyx"),
        ip.zeros((4, 5), axes="yx"),
        ip.zeros((3, 4), axes="zy"),
    )
    for i in range(3):
        assert out[i].shape == (3, 4, 5)
        assert out[i].axes == ["z", "y", "x"]
    
    out = ip.broadcast_arrays(
        ip.zeros((3,), axes="z"),
        ip.zeros((4,), axes="y"),
        ip.zeros((5,), axes="x"),
    )
    for i in range(3):
        assert out[i].shape == (3, 4, 5)
        assert out[i].axes == ["z", "y", "x"]
    
    out = ip.broadcast_arrays(
        ip.zeros((3, 3), axes="dz"),
        ip.zeros((3, 4), axes="dy"),
        ip.zeros((3, 5), axes="dx"),
    )
    for i in range(3):
        assert out[i].shape == (3, 3, 4, 5)
        assert out[i].axes == ["d", "z", "y", "x"]

def test_error():
    with pytest.raises(Exception):
        ip.broadcast_arrays(
            ip.zeros((3, 4, 5), axes="zyx"),
            ip.zeros((4, 6), axes="yx"),
        )

    with pytest.raises(Exception):
        ip.broadcast_arrays(
            ip.zeros((3, 4, 5), axes="zyx"),
            ip.zeros((5, 5), axes="yt"),
        )
        
    with pytest.raises(Exception):
        ip.broadcast_arrays(
            ip.zeros((3, 4, 5), axes="zyx"),
            ip.zeros((4, 5), axes="zy"),
        )