import impy as ip
import numpy as np
from numpy.testing import assert_equal
import pytest

def test_str_slicing():
    img = ip.random.normal(size=(10, 2, 30, 40), axes = "tcyx")
    assert_equal(img["t=4;c=0"], img.value[4,0])
    assert_equal(img["c=0;x=10:30"], img.value[:,0,:,10:30])
    assert_equal(img["y=3,6,20,26;x=7,3,4,13"], img.value[:,:,[3,6,20,26],[7,3,4,13]])
    assert_equal(img["t=::-1;x=2:-1:3"], img.value[::-1,:,:,2:-1:3])

def test_slicer_slicing():
    img = ip.random.normal(size=(10, 2, 30, 40), axes = "tcyx")
    assert_equal(img[ip.slicer.t[4].c[0]], img.value[4,0])
    assert_equal(img[ip.slicer.c[0].x[10:30]], img.value[:,0,:,10:30])
    assert_equal(img[ip.slicer.y[3, 6, 20, 26].x[7, 3, 4, 13]], img.value[:,:,[3,6,20,26],[7,3,4,13]])
    assert_equal(img[ip.slicer.t[::-1].x[2:-1:3]], img.value[::-1,:,:,2:-1:3])

def test_formatter():
    img = ip.random.normal(size=(10, 2, 30, 40))
    fmt = ip.slicer.y[4].get_formatter("tx")
    
    with pytest.raises(Exception):
        img[fmt]
    
    # test repr
    repr(ip.slicer)
    repr(fmt)
    
    assert_equal(img[fmt[0, 0]], img["t=0;y=4;x=0"])
    assert_equal(img[fmt[0, 3:6]], img["t=0;y=4;x=3:6"])
    assert_equal(img[fmt[:5][:6]], img["t=:5;y=4;x=:6"])
    
    fmt = ip.slicer.get_formatter("tx")
    with pytest.raises(Exception):
        img[fmt]
    
    # test repr
    repr(ip.slicer)
    repr(fmt)
    
    assert_equal(img[fmt[0, 0]], img["t=0;x=0"])
    assert_equal(img[fmt[0, 3:6]], img["t=0;x=3:6"])
    assert_equal(img[fmt[:5][:6]], img["t=:5;x=:6"])

def test_scale():
    img = ip.zeros((10, 10))
    img.set_scale(xy=0.2)
    assert img[0].axes[-1].scale == img.axes[-1].scale
    assert img[0, 2:4].axes[-1].scale == img.axes[-1].scale
    assert img[0, ::-1].axes[-1].scale == img.axes[-1].scale
    assert img[0, ::2].axes[-1].scale == img.axes[-1].scale * 2
    assert img[0, ::-3].axes[-1].scale == img.axes[-1].scale * 3
    assert img.binning(2).axes[-1].scale == img.axes[-1].scale * 2
    assert img.rescale(1/4).axes[-1].scale == img.axes[-1].scale * 4    

def test_dataframe_slicing():
    from impy.frame import AxesFrame
    
    df = AxesFrame({
        "t": [0, 0, 1, 1, 2], 
        "y": [8, 9, 7, 5, 3],
        "x": [2, 3, 5 ,7 ,10],
    })
    
    assert df.col_axes == ["t", "y", "x"]
    assert (df["t=0"]["t"] == 0).all()
    assert (df["t=1"]["t"] == 1).all()
    assert_equal(df["t=0"].values, np.array([[0, 8, 2], [0, 9, 3]]))
    assert_equal(df["t=1"].values, np.array([[1, 7, 5], [1, 5, 7]]))
    assert_equal(df["t=0:2"].values, np.array([[0, 8, 2], [0, 9, 3], [1, 7, 5], [1, 5, 7]]))
