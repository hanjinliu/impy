import impy as ip
import numpy as np

def test_str_slicing():
    img = ip.random.normal(size=(10, 2, 30, 40))
    img.axes = "tcyx"
    assert np.all(img["t=4;c=0"] == img.value[4,0])
    assert np.all(img["c=0;x=10:30"] == img.value[:,0,:,10:30])
    assert np.all(img["y=3,6,20,26;x=7,3,4,13"] == img.value[:,:,[3,6,20,26],[7,3,4,13]])
    assert np.all(img["t=::-1;x=2:-1:3"] == img.value[::-1,:,:,2:-1:3])

def test_slicer_slicing():
    img = ip.random.normal(size=(10, 2, 30, 40))
    img.axes = "tcyx"
    assert np.all(img[ip.slicer.t[4].c[0]] == img.value[4,0])
    assert np.all(img[ip.slicer.c[0].x[10:30]] == img.value[:,0,:,10:30])
    assert np.all(img[ip.slicer.y[3, 6, 20, 26].x[7, 3, 4, 13]] == img.value[:,:,[3,6,20,26],[7,3,4,13]])
    assert np.all(img[ip.slicer.t[::-1].x[2:-1:3]] == img.value[::-1,:,:,2:-1:3])
    