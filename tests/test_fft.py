import impy as ip
from pathlib import Path
import numpy as np
from scipy.fft import fftn
from numpy.testing import assert_allclose
ip.Const["SHOW_PROGRESS"] = False

def test_precision():
    path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
    img = ip.imread(path)["c=1;t=0"].as_float()
    
    assert_allclose(img.fft(shift=False),
                    fftn(img.value)
                    )
    
    assert_allclose(img.fft(shift=False, double_precision=True),
                    fftn(img.value.astype(np.float64)).astype(np.complex64)
                    )
    
    assert_allclose(img.fft().ifft(), img, rtol=1e-6)
    assert_allclose(img.fft(double_precision=True).ifft(), img)
    
    assert_allclose(np.fft.fftshift(img.local_dft()).ifft(), img, rtol=1e-6)
    assert_allclose(np.fft.fftshift(img.local_dft(double_precision=True)).ifft(), img)