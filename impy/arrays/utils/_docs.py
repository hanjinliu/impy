import re

shared_docs = dict(
    dims = """
        dims : str or int, optional
            Spatial dimensions. If string is given, each symbol is interpeted as an axis name of spatial dimensions. If an integer is given, it is interpreted as the number of spatial dimensions. 
            For instance, `dims="yx"` means axes "y" and "x" are spatial dimensions and function is applied to other axes, say, "t" and/or "c". `dims=3` is equivalent to `dims="zyx"`."""
    ,
        
    update = """
        update : bool, default is False
            If True, input itself is updated to the output."""
        ,
        
    radius = """
        radius : float, optional
            Radius of kernel structure. For instance, if input has two spatial dimensions, `radius=1` gives a structure:
                [[0, 1, 0], 
                 [1, 1, 1],
                 [0, 1, 0]]
            and `radius=1.8` gives a structure:
                [[0, 1, 1, 0], 
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [0, 1, 1, 0]]   """
        ,
        
    sigma = """
        sigma : float or array of float, optional
            Standard deviation(s) of Gaussian filter. If a scalar value is given, same standard deviation will applied to all the spatial dimensions."""
        ,
        
    order = """
        order : int, default is 1
            Spline interpolation order. For more details see https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp."""
        ,
        
    squeeze = """
        squeeze : bool, default is True
            If True, the redundant axis will be deleted. Array with sinl0gle value will be converted to a scalar."""
    ,
)

def write_docs(func):
    doc = func.__doc__
    if doc is not None:
        try:
            summary, params, rest = split_doc(doc)
            for key, value in shared_docs.items():
                params = re.sub("{"+key+"}", value, params)
            doc = merge_doc(summary, params, rest)
            func.__doc__ = doc
        except ValueError as e:
            print(func.__name__, ":", e)
    return func

def split_doc(doc:str):
    summary, other = doc.split("Parameters\n")
    params, rest = other.split("Returns\n")
    return summary, params, rest

def merge_doc(summary, params, rest):
    return summary + \
           "Parameters\n" + \
           params + \
           "Returns\n" + \
           rest

