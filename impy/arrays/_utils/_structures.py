import numpy as np
from ...array_api import xp

def circle(radius, shape, dtype="bool"):
    x = xp.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2 + 1)
    y = xp.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2 + 1)
    dx, dy = xp.meshgrid(x, y)
    return xp.array((dx ** 2 + dy ** 2) <= radius ** 2, dtype=dtype)

def ball_like(radius, ndim:int):
    half_int = int(2*radius)/2
    L = xp.arange(-half_int, half_int + 1)
    
    if ndim == 1:
        return xp.ones(int(radius)*2+1, dtype=np.uint8)
    elif ndim == 2:
        X, Y = xp.meshgrid(L, L)
        s = X**2 + Y**2
        return xp.array(s <= radius**2, dtype=np.uint8)
    elif ndim == 3:
        Z, Y, X = xp.meshgrid(L, L, L)
        s = X**2 + Y**2 + Z**2
        return xp.array(s <= radius**2, dtype=np.uint8)
    else:
        raise ValueError(f"dims must be 1 - 3, but got {ndim}")

def ball_like_odd(radius, ndim):
    """
    In ball_like, sometimes shapes of output will be even length, such as:
    [[0, 1, 1, 0],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [0, 1, 1, 0]]
    This is not suitable for specify().
    """    
    xc = int(radius)
    l = xc*2 + 1
    coords = xp.indices((l,)*ndim)
    s = xp.sum((a-xc)**2 for a in coords)
    return xp.array(s <= radius*radius, dtype=bool)