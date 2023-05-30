import impy as ip
from pathlib import Path
import numpy as np
from impy.array_api import xp
from numpy.testing import assert_allclose

path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
img_orig = ip.imread(path)

def test_precision(resource):
    with ip.SetConst(RESOURCE=resource):
        img = img_orig["c=1;t=0"].as_float()
        
        assert_allclose(img.fft(shift=False),
                        xp.asnumpy(xp.fft.fftn(xp.asarray(img.value)))
                        )
        
        assert_allclose(img.fft(shift=False, double_precision=True),
                        xp.asnumpy(xp.fft.fftn(xp.asarray(img.value.astype(np.float64))).astype(np.complex64))
                        )
        
        assert_allclose(img.fft().ifft(), img, rtol=1e-6)
        assert_allclose(img.fft(double_precision=True).ifft(double_precision=True), img)
        
        assert_allclose(np.fft.fftshift(img.local_dft()).ifft(), img, rtol=1e-6)
        assert_allclose(np.fft.fftshift(img.local_dft(double_precision=True)).ifft(double_precision=True), img)

def test_iteration(resource):
    if resource == "cupy":
        rtol = 1e-4
        atol = 1e-4
    else:
        rtol = 1e-6
        atol = 1e-6
    with ip.SetConst(RESOURCE=resource):
        img = img_orig["c=1;t=0"].as_float()
        
        ft0 = img.fft(dims="zx")
        fmt = ip.slicer.get_formatter("y")
        ft1 = np.stack([img[fmt[i]].fft(dims="zx") for i in range(img.shape.y)], axis="y")
        assert_allclose(ft0, ft1, rtol=rtol, atol=atol)
        
        ft0 = img.ifft(dims="zx")
        ft1 = np.stack([img[fmt[i]].ifft(dims="zx") for i in range(img.shape.y)], axis="y")
        assert_allclose(ft0, ft1, rtol=rtol, atol=atol)

def test_local_dft(resource):
    with ip.SetConst(RESOURCE=resource):
        img = img_orig["c=1"].as_float()
        img.local_dft(key="y=2:5", upsample_factor=8)
        img.local_dft(key="y=-2:3", upsample_factor=8)
        img.local_dft(key="y=-2:3;x=2:4", upsample_factor=8)
        img.local_dft(key=ip.slicer.y[:2].x[:3], upsample_factor=8)
