import impy as ip
from pathlib import Path
from numpy.testing import assert_allclose
ip.Const["SHOW_PROGRESS"] = False

def test_filters_and_slicing():
    path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
    img = ip.lazy_imread(path)
    assert img.axes == "tzcyx"
    sl = "y=20:40;x=30:50;c=0;z=2,4"
    assert_allclose(img[sl].as_imgarray(), img.as_imgarray()[sl])
    assert_allclose(img.median_filter().as_imgarray(),
                    img.as_imgarray().median_filter()
                    )
    assert_allclose(img.mean_filter().as_imgarray(),
                    img.as_imgarray().mean_filter()
                    )
    assert_allclose(img.opening().as_imgarray(),
                    img.as_imgarray().opening()
                    )
    assert_allclose(img.gaussian_filter().as_imgarray(),
                    img.as_imgarray().gaussian_filter()
                    )
    assert_allclose(img.affine(translation=[1, 50, 50]).as_imgarray(),
                    img.as_imgarray().affine(translation=[1, 50, 50])
                    )