import numpy as np
import impy as ip
from impy.utils.gauss import GaussianBackground

rng = np.random.default_rng(12345)
mu = np.array([30.5, 26.3])
sg = np.array([22.1, 19.6])
a = 1.5
b = 0.2
gs = GaussianBackground([mu, sg, a, b])
img = ip.asarray(gs.generate((56, 56))) + rng.normal(scale=0.1, size=(56, 56))

def test_2d():
    fit = img.gaussfit(scale=0.5)
    assert "GaussianParameters" in fit.metadata
    assert "GaussianParameters" not in img.metadata
    assert np.all(np.abs(mu - fit.metadata["GaussianParameters"]["mu"]) < 1)
    assert np.all(np.abs(sg - fit.metadata["GaussianParameters"]["sigma"]) < 1)
    assert abs(a - fit.metadata["GaussianParameters"]["A"]) < 0.2
    assert abs(b - fit.metadata["GaussianParameters"]["B"]) < 0.04
    
def test_3d():
    img0 = ip.asarray(gs.generate((56, 56))) + rng.normal(scale=0.1, size=(56, 56))
    a0 = [(1.2, 0), (1.0, 0.6), (2.2, 0.3)]
    img: ip.ImgArray = np.stack([a*img0 + b for a, b in a0], axis="c")
    fit = img.gaussfit(scale=0.5)
    assert fit.axes == img.axes
    
    assert "GaussianParameters" in fit.metadata
    params = fit.metadata["GaussianParameters"]
    for c in range(3):
        _a, _b = a0[c]
        assert np.all(np.abs(mu - params[c]["mu"]) < 1)
        assert np.all(np.abs(sg - params[c]["sigma"]) < 1)
        assert abs(_a * a - params[c]["A"]) < 0.2 * _a
        assert abs(_a * b + _b - params[c]["B"]) < 0.1 * _a

def test_2d_with_mask():
    mask = np.zeros_like(img.value, dtype=bool)
    mask[4:6, 5:7] = True
    mask[18:20, 8:10] = True
    mask[22:23, 23:24] = True
    mask[15:17, 16:18] = True
    img0 = img.copy()
    img0[mask] = 20.0  # outliers
    
    fit_no_mask = img0.gaussfit(scale=1, mask=None)
    fit_with_mask = img0.gaussfit(scale=1, mask=mask)
    param_no_mask = fit_no_mask.metadata["GaussianParameters"]
    param_with_mask = fit_with_mask.metadata["GaussianParameters"]
    assert np.all(np.abs(mu - fit_with_mask.metadata["GaussianParameters"]["mu"]) < 1)
    
    assert abs(param_no_mask["A"] - a) > 0.25
    assert abs(param_no_mask["B"] - b) > 0.1
    
    assert abs(param_with_mask["A"] - a) < 0.2 
    assert abs(param_with_mask["B"] - b) < 0.05
    