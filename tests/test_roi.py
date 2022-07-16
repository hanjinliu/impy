import impy as ip
from impy.roi import LineRoi
from pathlib import Path
import tempfile

def test_roi_slicing():
    roi = LineRoi(
        [[3, 4],
         [6, 10]],
        "tyx", 
        multi_dims=[2]
    )
    assert roi._slice_by((2, slice(1, 6, 1), slice(3, 8, 1))) == LineRoi([[2, 1], [5, 7]], "yx", multi_dims=None)
    assert roi._slice_by((2, slice(1, 6, 2), slice(3, 12, -1))) == LineRoi([[1, 8], [2.5, 2]], "yx", multi_dims=None)
    assert roi._slice_by((1, slice(1, 6, 2), slice(3, 12, -1))) is None
    assert roi._slice_by((2, 1, slice(3, 12, -1))) is None
    assert roi._slice_by((2, slice(1, 6, 2), 1)) is None

def test_multi_dims():
    roi = LineRoi(
        [[3, 4],
         [6, 10]],
        "tzcyx", 
        multi_dims=[2, 14, 1]
    )
    assert roi._slice_by((2, 14, 1, slice(1, 6, 1), slice(3, 8, 1))) is not None
    assert roi._slice_by((2, 14, slice(0, 3, 1), slice(1, 6, 1), slice(3, 8, 1))) is not None
    assert roi._slice_by((2, slice(0, 3, 1), 1, slice(1, 6, 1), slice(3, 8, 1))) is not None
    assert roi._slice_by((slice(0, 3, 1), 14, 1, slice(1, 6, 1), slice(3, 8, 1))) is not None
    assert roi._slice_by((slice(0, 3, 1), 14, slice(0, 3, 1), slice(1, 6, 1), slice(3, 8, 1))) is not None
    assert roi._slice_by((slice(0, 3, 1), slice(0, 3, 1), 1, slice(1, 6, 1), slice(3, 8, 1))) is not None
    assert roi._slice_by((1, 14, 1, slice(1, 6, 1), slice(3, 8, 1))) is None
    assert roi._slice_by((2, 13, 1, slice(1, 6, 1), slice(3, 8, 1))) is None
    assert roi._slice_by((2, 14, 3, slice(1, 6, 1), slice(3, 8, 1))) is None
    assert roi._slice_by((1, 14, slice(0, 3, 1), slice(1, 6, 1), slice(3, 8, 1))) is None
    assert roi._slice_by((1, slice(0, 3, 1), 1, slice(1, 6, 1), slice(3, 8, 1))) is None
    assert roi._slice_by((slice(0, 3, 1), 1, 1, slice(1, 6, 1), slice(3, 8, 1))) is None
    assert roi._slice_by((slice(0, 3, 1), 1, slice(0, 3, 1), slice(1, 6, 1), slice(3, 8, 1))) is None
    assert roi._slice_by((slice(0, 3, 1), slice(0, 3, 1), 3, slice(1, 6, 1), slice(3, 8, 1))) is None
        

def test_drop():
    roi = LineRoi(
        [[3, 4],
         [6, 10]],
        "tzcyx", 
        multi_dims=[2, 14, 1]
    )
    
    assert roi.drop(0) == LineRoi([[3, 4], [6, 10]], "zcyx", multi_dims=[14, 1])
    assert roi.drop(2) == LineRoi([[3, 4], [6, 10]], "tzyx", multi_dims=[2, 14])
    

def test_covariance():
    img = ip.zeros((5, 10, 10), axes="tyx")
    img.rois = [
        LineRoi([[1, 1], [2, 3]], axes="yx"),
        LineRoi([[4, 6], [2, 3]], axes="tyx", multi_dims=1),
    ]
    
    img0 = img[1]
    assert len(img0.rois) == 2
    assert img0.rois[0] == LineRoi([[1, 1], [2, 3]], axes="yx")
    assert img0.rois[1] == LineRoi([[4, 6], [2, 3]], axes="yx")
    
    img0 = img[1, 2:8]
    assert len(img0.rois) == 2
    assert img0.rois[0] == LineRoi([[-1, 1], [0, 3]], axes="yx")
    assert img0.rois[1] == LineRoi([[2, 6], [0, 3]], axes="yx")
    
    img0 = img[0, 3:8]
    assert len(img0.rois) == 1
    assert img0.rois[0] == LineRoi([[-2, 1], [-1, 3]], axes="yx")

def test_consistency():
    path = Path(__file__).parent / "_test_images" / "RoiSet.zip"
    rois = ip.roiread(path)
    
    assert len(rois) == 6
    
    with tempfile.TemporaryDirectory() as tmpdir:
        _path = Path(tmpdir) / "RoiSet-test.zip"
        rois.tofile(_path)
        rois2 = ip.roiread(_path)
        assert len(rois2) == 6
        for roi1, roi2 in zip(rois, rois2):
            assert roi1 == roi2
