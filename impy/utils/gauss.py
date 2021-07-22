import numpy as np
import scipy

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
    def mu_inrange(self, low, high):
        return np.logical_and(low<=self.mu, self.mu<=high).all()
    
    def sg_inrange(self, low, high):
        sg_ = np.abs(self.sg)
        return np.logical_and(low<=sg_, sg_<=high).all()

class DiagonalGaussian(Gaussian):
    def __init__(self, params=None):
        if params is None:
            self.mu = self.sg = self.a = self.b = None
        else:
            self.mu, self.sg, self.a, self.b = params
    
    @property
    def mu(self):
        return self._mu
    
    @mu.setter
    def mu(self, value):
        if value is None:
            self._mu = None
        else:
            self._mu = np.asarray(value)
        
    @property
    def sg(self):
        return self._sg
    
    @sg.setter
    def sg(self, value):
        if value is None:
            self._sg = None
        else:
            self._sg = np.asarray(value)
        
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
    
    def fit(self, data:np.ndarray, method="Powell"):
        if self.mu is None or self.sg is None or self.a is None or self.b is None:
            self._estimate_params(data)
        r = np.indices(data.shape)
        result = scipy.optimize.minimize(square, self.params, args=(diagonal_gaussian, r, data),
                                         method=method)
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
    def __init__(self, params=None, initial_sg=1):
        super().__init__(params)
        self.initial_sg = initial_sg
        
    def _estimate_params(self, data:np.ndarray):
        # n-dim argmax
        self.mu = np.array(np.unravel_index(np.argmax(data), data.shape), dtype="float32")
        self.sg = np.full(data.ndim, self.initial_sg, dtype="float32")
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