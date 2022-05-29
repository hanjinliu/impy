import re

__all__ = ["write_docs", "copy_docs"]

shared_docs = dict(
    dims = \
        """
        dims : str or int, optional
            Spatial dimensions. If string is given, each symbol is interpeted as an axis name of spatial dimensions. If an integer is given, it is interpreted as the number of spatial dimensions. 
            For instance, ``dims="yx"`` means axes ``"y"`` and ``"x"`` are spatial dimensions and function is applied to other axes, say, ``"t"`` and/or ``"c"``. ``dims=3`` is equivalent to ``dims="zyx"``.
            """
    ,
        
    update = \
        """
        update : bool, default is False
            If True, input itself is updated to the output.
            """
        ,
        
    radius = \
        """
        radius : float, optional
            Radius of kernel structure. For instance, if input has two spatial dimensions, ``radius=1`` gives a structure

                .. code-block:: python
                
                    [[0, 1, 0], 
                     [1, 1, 1],
                     [0, 1, 0]]
            
            and ``radius=1.8`` gives a structure 
            
                .. code-block:: python
                
                    [[0, 1, 1, 0], 
                     [1, 1, 1, 1],
                     [1, 1, 1, 1],
                     [0, 1, 1, 0]]
            
            """
        ,
        
    sigma = \
        """
        sigma : float or array of float, optional
            Standard deviation(s) of Gaussian filter. 
            If a scalar value is given, same standard deviation will applied to all the spatial dimensions.
            """
        ,
        
    order = \
        """
        order : int, default is 1
            Spline interpolation order. 
            For more details see `here <https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp>`_.
            """
        ,
        
    squeeze = \
        """
        squeeze : bool, default is True
            If True, the redundant axis will be deleted. 
            Array with sinl0gle value will be converted to a scalar.
            """
    ,
    
    connectivity = \
        """
        connectivity : int, optional
            Connectivity of pixels. See ``skimage.measure.label``. 
            """
    ,
    
    double_precision = \
        """
        double_precision : bool, default is False
            If True, FFT will be calculated using 64-bit float and 128-bit complex.
            """
    ,
    
    inputs_of_correlation = \
        """
        img0 : ImgArray
            First image.
        img1 : ImgArray
            Second image.
            """
    ,
    
    mode = \
        """
        mode : {"reflect", "constant", "nearest", "mirror", "wrap"}
            Edge padding mode.
            """
    ,
    cval = \
        """
        cval : float, default is 0.0
            Constant value for constant padding mode.
        """

)

def write_docs(func):
    doc = func.__doc__
    if doc is not None:
        summary, params, rest = _split_doc(doc)
        for key, value in shared_docs.items():
            value = value.rstrip()
            params = re.sub("{"+key+"}", value, params)
        doc = _merge_doc(summary, params, rest)
        func.__doc__ = doc
    return func

def _split_doc(doc:str):
    summary, other = doc.split("Parameters\n")
    params, rest = other.split("Returns\n")
    return summary, params, rest

def _merge_doc(summary, params, rest):
    return summary + \
           "Parameters\n" + \
           params + \
           "Returns\n" + \
           rest

def copy_docs(original=None):
    def _copy_docs(func):
        if original is not None:
            fstr = str(original).lstrip("<function ").split(" at")[0]
            classname, funcname = fstr.split(".")
            doc = f"Copy of {fstr}. This function returns the same result but the " \
                "value is evaluated lazily as an dask array.\n"
            doc += original.__doc__
            returns = fr"Returns[\s]*-*[\s]*{classname}"
            try:
                st = re.findall(returns, doc)[0]
                st = re.sub(classname, "LazyImgArray", st)
                doc = re.sub(returns, st, doc)
            except IndexError:
                pass
            func.__doc__ = doc
        return func
    return _copy_docs