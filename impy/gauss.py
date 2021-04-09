import numpy as np
from scipy import optimize as opt

def square(params, func, r, z):
    """
    calculate ||z - func(x, y, *params)||^2
    where x and y are determine by z.shape
    """
    z_guess = func(r, *params)
    return np.mean((z - z_guess)**2)

def diagonal_gaussian(r, *params):
    a, b = params[-2:]
    ndim = len(params[:-2])//2
    mu = params[:ndim]
    sg = params[ndim:-2]
    z_value = np.array([(x0 - mu0)/sg0 for x0, mu0, sg0 in zip(r, mu, sg)])
    return a * np.exp(-np.sum(z_value**2, axis=0) / 2) + b
    
class Gaussian:
    pass

class DiagonalGaussian(Gaussian):
    def __init__(self, params=None):
        if params is None:
            self.mu = self.sg = self.a = self.b = None
        else:
            self.mu, self.sg, self.a, self.b = params
    
    @property    
    def params(self):
        return tuple(self.mu) + tuple(self.sg) + (self.a, self.b)
    
    @params.setter
    def params(self, params:tuple):
        self.a, self.b = params[-2:]
        self.mu = params[:self.ndim]
        self.sg = params[self.ndim:-2]
    
    @property
    def ndim(self):
        return self.mu.size
    
    def fit(self, data:np.ndarray) -> opt.OptimizeResult:
        if not all((self.mu, self.sg, self.a, self.b)):
            self._estimate_params(data)
        r = np.indices(data.shape)
        result = opt.minimize(square, self.params, args=(diagonal_gaussian, r, data))
        self.params = result.x
        
        return result
            
    def rescale(self, scale:float):
        self.mu *= scale
        self.sg *= scale
        return None
    
    def shift(self, dxdy):
        self.mu += np.array(dxdy)
        return None
    
    def generate(self, shape:tuple) -> np.ndarray:
        r = np.indices(shape)
        return diagonal_gaussian(r, *self.params)
    
    def _estimate_params(self, data:np.ndarray):
        pass

class GaussianParticle(DiagonalGaussian):
    def _estimate_params(self, data:np.ndarray):
        # n-dim argmax
        self.mu = np.array(np.unravel_index(np.argmax(data), data.shape), dtype="float32")
        self.sg = np.full(data.ndim, 1, dtype="float32")
        self.b, p95 = np.percentile(data, [5, 95])
        self.a = p95 - self.b
        return None

class GaussianBackground(DiagonalGaussian):
    def _estimate_params(self, data:np.ndarray):
        # n-dim argmax
        self.mu = np.array(np.unravel_index(np.argmax(data), data.shape), dtype="float32")
        self.sg = np.array(data.shape, dtype="float32")
        self.b, p95 = np.percentile(data, [5, 95])
        self.a = p95 - self.b
        return None