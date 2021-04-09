import numpy as np
from skimage.measure.fit import _dynamic_max_trials
from skimage._shared.utils import check_random_state
from skimage.measure import ransac
from scipy import optimize as opt

def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None):

    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = check_random_state(random_state)

    # in case data is not pair of input and output, male it like it
    
    # <---- HERE --->
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])
    # <---- END --->

    if not (0 < min_samples < num_samples):
        raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), num_samples))

    # TODO: 
    # spl_idxs should be 2D
    # 
    
    # for the first run use initial guess of inliers
    spl_idxs = (initial_inliers if initial_inliers is not None
                else random_state.choice(num_samples, min_samples, replace=False))

    for num_trials in range(max_trials):
        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]
        # for next iteration choose random sample set and be sure that no samples repeat
        spl_idxs = random_state.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        if (
            # more inliers
            sample_inlier_num > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (sample_inlier_num == best_inlier_num
                and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     num_samples,
                                                     min_samples,
                                                     stop_probability)
            if (best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
                or num_trials >= dynamic_max_trials):
                break

    # estimate final model using all inliers
    if best_inliers is not None and any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)
    else:
        best_model = None
        best_inliers = None

    return best_model, best_inliers



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