from impy import DataList, DataDict
import pytest

def test_empty_list():
    l0 = DataList()
    l0 + [1]
    DataList([1]) + l0
    l0.append(1)
    
def test_list():
    l = DataList(["a", "b"])
    
    l1 = l + DataList(["c"])
    with pytest.raises(TypeError):
        l1.append(2)
    
    assert l1[[0, 2]]._list == ["a", "c"]
    assert l.endswith("a")._list == [True, False]
    del l[0]

def test_empty_dict():
    d0 = DataDict()
    d0["a"] = 0

def test_dict():
    d = DataDict(a="A", b="B")
    
    with pytest.raises(TypeError):
        d[3] = 4
    
    out = d.endswith("A")
    assert out["a"] == True
    assert out["b"] == False