import numpy as np
import pytest
import impy as ip

def test_radon_2d(resource):
    ndegs = 61
    degrees = np.linspace(-160, 160, ndegs)
    with ip.SetConst(RESOURCE=resource):
        img = ip.gaussian_kernel((15, 18), sigma=[2, 3])
        sino = img.radon(degrees)
        assert sino.axes == ["degree", "x"]
        assert sino.shape == (ndegs, 18)
        
        img2 = sino.iradon(degrees, height=15)
        assert img2.shape == (15, 18)
        assert ip.zncc(img, img2, dims="yx") > 0.85

def test_radon_3d(resource):
    ndegs = 61
    degrees = np.linspace(-160, 160, ndegs)
    with ip.SetConst(RESOURCE=resource):
        img = ip.gaussian_kernel((15, 18, 21), sigma=[2, 3, 3.5], axes="zyx")
        with pytest.raises(ValueError):
            img.radon(degrees)
        sino = img.radon(degrees, central_axis="y")
        assert sino.axes == ["degree", "y", "x"]
        assert sino.shape == (ndegs, 18, 21)

        img2 = sino.iradon(degrees, central_axis="y", height=15)
        assert img2.shape == (15, 18, 21)
        assert ip.zncc(img, img2, dims="zyx") > 0.85
