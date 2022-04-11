import impy as ip
from pathlib import Path
import numpy as np

def test_bind_unbind():
    path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
    img = ip.imread(path)
    
    @ip.bind(indtype=np.float32, outdtype=np.float32)
    def normalize(img: np.ndarray):
        min_, max_ = img.min(), img.max()
        return (img - min_)/(max_ - min_)

    out = img.normalize()
    assert out.dtype == np.float32

    def normalize(img: np.ndarray):
        min_, max_ = img.min(), img.max()
        return (img - min_)/(max_ - min_)
    
    with ip.bind(normalize, indtype=np.float32, outdtype=np.float32):
        out = img.normalize()
        assert out.dtype == np.float32
    assert not hasattr(img, "normalize")

    ip.bind(np.mean, "calc_mean", outdtype=np.float32, kind="property")
    img.calc_mean()

    assert ip.bind.bound == set(["calc_mean"])