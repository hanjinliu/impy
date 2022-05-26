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
