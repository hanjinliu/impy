import impy as ip
from pathlib import Path
import numpy as np
from impy.array_api import xp
from numpy.testing import assert_allclose

def test_precision(resource):
    with ip.SetConst(RESOURCE=resource):
        path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
        img = ip.imread(path)["c=1;t=0"].as_float()
        
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
    with ip.SetConst(RESOURCE=resource):
        path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
        img = ip.imread(path)["c=1;t=0"].as_float()
        
        ft0 = img.fft(dims="zx")
        ft1 = np.stack([img[f"y={i}"].fft(dims="zx") for i in range(img.shape.y)], axis="y")
        assert_allclose(ft0, ft1)
        
        ft0 = img.ifft(dims="zx")
        ft1 = np.stack([img[f"y={i}"].ifft(dims="zx") for i in range(img.shape.y)], axis="y")
        assert_allclose(ft0, ft1)