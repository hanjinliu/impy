import numpy as np
from scipy import optimize as opt

def square(params, func, z):
    """
    calculate ||z - func(x, y, *params)||^2
    where x and y are determine by z.shape
    """
    x, y = np.indices(z.shape)
    z_guess = func(x, y, *params)
    return np.mean((z - z_guess)**2)

def diagonal_gaussian(r, ndim, mu, sg, a, b):
    z_value = np.array([(r[i] - mu[i])/sg[i] for i in range(ndim)])
    return  a * np.exp(-np.sum(z_value**2)/2) + b
    
class Gaussian:
    pass

class DiagonalGaussian(Gaussian):
    model :callable
    def __init__(self, params=None):
        if params is None:
            self.mu = self.sg = self.a = self.b = None
        self.mu, self.sg, self.a, self.b = params
    
    def fit(self, data):
        if not all((self.mu, self.sg, self.a, self.b)):
            self._estimate_params(data)
        result = opt.minimize(square, self.params, args=(self.model, data))
        self.params = result.x
        
        return result
            
    def rescale(self, scale):
        self.mu /= scale
        self.sg /= scale
        return None
    
    def _estimate_params(self, data):
        pass

class GaussianParticle(DiagonalGaussian):
    def __init__(self, params=None, ndim=2):
        super().__init__(params)
        self.ndim = ndim
        def model_gaussian(r, mu, sg, a, b):
            return diagonal_gaussian(r, self.ndim, mu, sg, a, b)
        self.model = model_gaussian
        
    def _estimate_params(self, data):
        # 2-dim argmax
        self.mu = np.unravel_index(np.argmax(data), data.shape)
        self.sg = np.full(self.ndim, 1, dtype="float32")
        self.b, p95 = np.percentile(data, [5, 95])
        self.a = p95 - self.b
        return None

class GaussianBackground(DiagonalGaussian):
    def __init__(self, params=None, ndim=2):
        super().__init__(params)
        self.ndim = ndim
        def model_gaussian(r, mu, sg, a, b):
            return diagonal_gaussian(r, 2, mu, sg, a, b)
        self.model = model_gaussian
        
    def _estimate_params(self, data):
        # 2-dim argmax
        self.mu = np.unravel_index(np.argmax(data), data.shape)
        self.sg = np.array(self.shape, dtype="float32")
        self.b, p95 = np.percentile(data, [5, 95])
        self.a = p95 - self.b
        return None