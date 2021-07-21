shared_docs = dict(
    dims = """
        dims : str or int, optional
            Spatial dimensions. If string is given, each symbol is interpeted as an axis name of spatial dimensions. If an integer is given, it is interpreted as the number of spatial dimensions. 
            For instance, `dims="yx"` means axes "y" and "x" are spatial dimensions and function is applied to other axes, say, "t" and/or "c". `dims=3` is equivalent to `dims="zyx"`.
        """,
        
    update = """
        update : bool, default is False
            If True, input itself is updated to the output.
        """,
        
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
                 [0, 1, 1, 0]]
        """,
        
    sigma = """
        sigma : float or array of float, optional
            Standard deviation(s) of Gaussian filter. If a scalar value is given, same standard deviation will applied to all the spatial dimensions.
        """,
        
    order = """
        order : int, default is 1
            Spline interpolation order.
        """,
)

def write_docs(func):
    doc = func.__doc__
    if doc is not None:
        doc = doc.format(**shared_docs)
        func.__doc__ = doc
    return func
