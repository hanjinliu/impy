import impy as ip
from pathlib import Path

ip.Const["SHOW_PROGRESS"] = False

def test_filters():
    filters = ["median_filter", 
               "gaussian_filter", 
               "lowpass_filter", 
               "lowpass_conv_filter",
               "highpass_filter",
               "erosion",
               "dilation",
               "opening",
               "closing",
               "tophat",
               "mean_filter",
               "std_filter",
               "coef_filter",
               "diameter_opening",
               "diameter_closing",
               "area_opening",
               "area_closing",
               "entropy_filter",
               "enhance_contrast",
               "laplacian_filter",
               "kalman_filter",
               "dog_filter",
               "doh_filter",
               "log_filter",
               "rolling_ball",
               "rof_filter",
               ]
    
    path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
    img = ip.imread(path)["c=1;z=2"]
    for f in filters:
        try:
            getattr(img.as_uint8(), f)()
            getattr(img.as_uint16(), f)()
            getattr(img.as_float(), f)()
        except Exception as e:
            e.args = (str(e) + f". Caused by {f}.",)
            raise e

def test_sm():
    path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
    img = ip.imread(path)["c=1"]
    for method in ["dog", "doh", "log", "ncc"]:
        try:
            img.find_sm(method=method, percentile=98)
        except Exception as e:
            e.args = (str(e) + f". Caused by method={method}.",)
            raise e
    img.centroid_sm()