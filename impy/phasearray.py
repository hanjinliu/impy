from __future__ import annotations
import numpy as np
from .labeledarray import LabeledArray
from ._process import *
from .deco import *
from .func import *

class PhaseArray(LabeledArray):
    @record()
    def deg2rad(self, *, update=False):
        return np.deg2rad(self.value)
    
    @record()
    def rad2deg(self, *, update=False):
        return np.rad2deg(self.value)
    
    @record()
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    def mean_filter(self, radius:float=1, *, periodicity:float=np.pi*2, dims=None, update:bool=False) -> PhaseArray:
        """
        Mean filter using phase averaging method, which is:
        arg(sum(e^j(X0 + X1 + ...)))

        Parameters
        ----------
        radius : float, default is 1
            Radius of kernel.
        periodicity : float
            Periodicity of phase. 
        dims : str or int, optional
            Spatial dimensions.
        update : bool, default is False
            If update self to filtered image.

        Returns
        -------
        PhaseArray
            Filtered image.
        """        
        disk = ball_like(radius, len(dims))
        a = 2*np.pi/periodicity
            
        out = self.parallel(phase_mean_, complement_axes(dims, self.axes), disk, a, outdtype=self.dtype)
        return out