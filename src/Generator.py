# -*- coding: utf-8 -*-
from scipy.stats import levy, expon, norm, uniform, powerlaw
import numpy as np
import random
import torch
    
    
def gamma(H, k):
    """
    Parameters
    ----------
    H : float
        the hurst exponent
    k : int
        the amount of steps taken from the start
    Returns
    -------
    float
        the value of the covariance of the fractional brownian noise for the current E(X_1,X_k+1)
    """
    # term1 = (k+1)**(2*H)
    # term2 = np.abs(k-1)**(2*H)
    # term3 = 2*(k**(2*H))
    term1 = (k+1)**(2*H)
    term2 = np.abs(k-1)**(2*H)
    term3 = 2*(np.abs(k)**(2*H))
    return 0.5 * (term1 + term2 - term3)


def calc_transformation(N, H):
    """
    Parameters
    ----------
    N : int
        the amount of steps the will be sampled
    H : float
        the hurst exponent
    Returns
    -------
    numpy.array
        a matrix that transforms normal gaussian noise to fractinal gaussian noise
    """
    # initialize the L matrix from the choleski decomposition
    L = np.zeros((N, N))
    # initialize the first column of the matrix
    for i in range(N):
        L[i, 0] = gamma(H, i)
    for i in range(1, N):
        # iterating over the rows and setting the columns without the diagonal
        for j in range(1, i):
            gam = gamma(H, (i - j))
            summation = 0
            for k in range(j):
                summation += (L[i,k] * L[j,k])
            L[i, j] = (1/L[j,j]) * (gam - summation)
        # setting the diagonal
        summation = 0
        for k in range(i):
            summation += L[i,k]**2
        L[i,i] = np.sqrt(1 - summation)
    return L


class FractionalGaussianNoise():
    def __init__(self, steps, hurst):
        """
        initializes the Gaussian noise by calculating the lower triangular matrix of the choleski decomposition
        @:param steps = the amount of steps N in the process
        @:param hurst = the hurst parameter
        """    
        self.N = steps
        self.H = hurst
        self.normal_dist = norm(loc=0, scale=1)
        self.L = calc_transformation(steps, hurst)
        
    def temp_walk_generation(self):
        Z = self.normal_dist.rvs(size=self.N)
        Z[0] = 0
        X = self.L.dot(Z)
        return X

    def generate_tensor(self, shape, torch_tensor=True):
        pass
        

class LevyGenerator():
    def __init__(self, scale=1):
        self.frozen_levy = levy(loc=0.001, scale=scale)

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a levy distribution
        @:param shape = the shape of the tensor of levy variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        @:param scale = the scale of the levy distribution the numbers are taken from (-> y = x*scale)
        """
        levy_tensor = self.frozen_levy.rvs(size=shape)
        if torch_tensor:
            levy_tensor = torch.from_numpy(levy_tensor)
        return levy_tensor
    
    def generate_variate(self):
        return self.frozen_levy.rvs()


class ExponGenerator:
    def __init__(self, lamda=10):
        """
        Parameters
        ----------
        lamda : float, optional
            the scaling factor of the exponent. f(x) = lamda * exp(-lamda * x). The default is 10.
            <f(x)> = 1/lamda
        """
        self.exp = expon(scale=(1/lamda))

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a exponential distribution
        @:param shape = the shape of the tensor of variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        """
        exp_tensor = self.exp.rvs(size=shape)
        if torch_tensor:
            exp_tensor = torch.from_numpy(exp_tensor)
        return exp_tensor

    def generate_variate(self):
        return self.exp.rvs()
 
    
class SymmetricExponGenerator:
    def __init__(self, lamda=1/np.sqrt(2)):
        """
        pdf(x) = 0.5 * lamda * exp(-lamda * x)  for x >= 0
        pdf(x) = -0.5 * lamda * exp(lamda * x) for x < 0
        mean = 0, variance = 0.5 * (1 / lamda^2)
        """
        self.exp = expon(scale=(1/lamda))

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a exponential distribution
        @:param shape = the shape of the tensor of variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        """
        exp_tensor = 0.5 * np.multiply(self.exp.rvs(size=shape), np.random.choice([-1, 1], shape))
        #exp_tensor = exp_tensor if random.random() <= 0.5 else (-1 * exp_tensor)
        if torch_tensor:
            exp_tensor = torch.from_numpy(exp_tensor)
        return exp_tensor

    def generate_variate(self):
        return self.exp.rvs() * np.random.choice([-1/2, 1/2])


class NormalGenerator:
    def __init__(self, mean=0, std=1):
        """
        Parameters
        ----------
        mean : the mean of the normal distribution
        std : the standard deviation of the normal distribution
        """
        self.norm = norm(loc=mean, scale=std)

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a normal distribution
        @:param shape = the shape of the tensor of variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        """
        norm_tensor = self.norm.rvs(size=shape)
        if torch_tensor:
            norm_tensor = torch.from_numpy(norm_tensor)
        return norm_tensor

    def generate_variate(self):
        return self.norm.rvs()
    
    
class UniformGenerator:
    def __init__(self, low=0, high=1):
        """
        Parameters
        ----------
        mean : the mean of the normal distribution
        std : the standard deviation of the normal distribution
        """
        self.unif = uniform(loc=low, scale=(high - low))

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a normal distribution
        @:param shape = the shape of the tensor of variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        """
        unif_tensor = self.unif.rvs(size=shape)
        if torch_tensor:
            unif_tensor = torch.from_numpy(unif_tensor)
        return unif_tensor

    def generate_variate(self):
        return self.unif.rvs()
    
    def update(self, a):
        pass



# class PowerLawGenerator:
#     """
#     implements the distribution f(x,a) = x^(-a),   a>0
#     """
#     def __init__(self, a, xmin):
#         """
#         Parameters
#         ----------
#         a = the exponent to use
#         xmin = the minimal value for powerlaw to prevent divergent. should be the time delta
#         """
#         self.a = a
#         self.xmin = xmin
#         self.dist = uniform(loc=0, scale=(1 - 0))

#     def generate_tensor(self, shape, torch_tensor=True):
#         """
#         generates a tensor of random numbers taken from a normal distribution
#         @:param shape = the shape of the tensor of variates as a list of dimensions
#         @:param torch_tensor = convert the numpy array into a pytorch tensor
#         """
#         # sample from uniform distribution
#         powerlaw_tensor = self.dist.rvs(size=shape)
#         # transform the uniform distribution into powerlaw with alpha = a
#         powerlaw_tensor = self.xmin * (1 - powerlaw_tensor) ** (-1 / (self.a - 1))
#         if torch_tensor:
#             powerlaw_tensor = torch.from_numpy(powerlaw_tensor)
#         return powerlaw_tensor

#     def generate_variate(self):
#         x = self.dist.rvs()
#         return self.xmin * (1 - x) ** (-1 / (self.a - 1))
    
#     def update(self, a):
#         self.a = a


# class PowerLawGenerator:
#     """
#     implements the distribution f(t,a) = (1/d) t^(-a),   a > 1, d=scaling factor (for a<1 the distribution is not normalized)
#     the transformation from uniform deviate x~U[0,1] used is t = [(1/delta)(1-a)(x-1)] ^ (1/(1-a))
#     this transformation takes into account a minimal value x_0 that is required to keep the distribution normalized
#     it is taken from numerical recipes chapter 7
#     """
#     def __init__(self, a, delta=1):
#         """
#         Parameters
#         ----------
#         a = the exponent to use
#         """
#         self.a = a
#         self.delta = delta
#         self.dist = uniform(loc=0, scale=(1 - 0))

#     def generate_tensor(self, shape, torch_tensor=True):
#         """
#         generates a tensor of random numbers taken from a normal distribution
#         @:param shape = the shape of the tensor of variates as a list of dimensions
#         @:param torch_tensor = convert the numpy array into a pytorch tensor
#         """
#         # sample from uniform distribution
#         powerlaw_tensor = self.dist.rvs(size=shape)
#         # transform the uniform distribution into powerlaw with alpha = a
#         powerlaw_tensor = self.transform(powerlaw_tensor)
#         if torch_tensor:
#             powerlaw_tensor = torch.from_numpy(powerlaw_tensor)
#         return powerlaw_tensor

#     def generate_variate(self):
#         x = self.dist.rvs()
#         return self.transform(x)
    
#     def update(self, a):
#         self.a = a
        
#     def transform(self, x):
#         return (((1/self.delta) * (1 - self.a) * (x - 1)) ** (1/(1-self.a)))

class PowerLawGenerator:
    """
    implements the distribution f(t,a) = (1/d) t^(-a),   a > 1, d=scaling factor (for a<1 the distribution is not normalized)
    the transformation from uniform deviate x~U[0,1] used is t = [(1/delta)(1-a)(x-1)] ^ (1/(1-a))
    this transformation takes into account a minimal value x_0 that is required to keep the distribution normalized
    it is taken from numerical recipes chapter 7
    """
    def __init__(self, a, delta=1e-5):
        """
        Parameters
        ----------
        a = the exponent to use
        a_range = for variable exponent rate, defines the range of the variable exponent
        """
        variable = type(a) == list
        self.a = uniform(loc=a, scale=0) if not variable else uniform(loc=a[0], scale=(a[1]-a[0]))
        self.t_0 = delta
        self.dist = uniform(loc=0, scale=(1 - 0))

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a normal distribution
        @:param shape = the shape of the tensor of variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        """
        # sample from uniform distribution
        powerlaw_tensor = self.dist.rvs(size=shape)
        # transform the uniform distribution into powerlaw with alpha = a
        powerlaw_tensor = self.transform(powerlaw_tensor)
        if torch_tensor:
            powerlaw_tensor = torch.from_numpy(powerlaw_tensor)
        return powerlaw_tensor

    def generate_variate(self):
        x = self.dist.rvs()
        return self.transform(x)
    
    def update(self, a):
        variable = type(a) == list
        self.a = uniform(loc=a, scale=0) if not variable else uniform(loc=a[0], scale=(a[1]-a[0]))
        
    def transform(self, x):
        a = self.a.rvs()
        return self.t_0 * (1 - x) ** (1/(1-a))

        
