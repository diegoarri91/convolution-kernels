import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from .utils import get_arg_support, get_dt, searchsorted


class Kernel:

    def __init__(self, prior=None, prior_pars=None):
        self.prior = prior
        self.prior_pars = np.array(prior_pars)
        self.fix_values = False
        self.values = None

    def interpolate(self, t):
        pass

    # def get_KernelValues(self, t):
    #     kernel_values = self.interpolate(t)
    #     return KernelValues(values=kernel_values, support=self.support)

    def plot(self, t=None, ax=None, offset=0, invert_t=False, invert_values=False, exp_values=False, **kwargs):

        if t is None:
#             t = np.arange(self.support[0], self.support[1] + dt, dt)
            t = np.linspace(self.support[0], self.support[1], 200)

        if ax is None:
            fig, ax = plt.subplots()

        y = self.interpolate(t) + offset
        if invert_t:
            t = -t
        if invert_values:
            y = -y
        if exp_values:
            y = np.exp(y)
        ax.plot(t, y, **kwargs)

        return ax

    def set_values(self, dt):
        arg0 = int(self.support[0] / dt)
        argf = int(np.ceil(self.support[1] / dt))
        t_support = np.arange(arg0, argf + 1, 1) * dt
        self.values = self.interpolate(t_support)
        return self

    # def set_values(self, dt, ndim):
    #     arg0 = int(self.support[0] / dt)
    #     argf = int(np.ceil(self.support[1] / dt))
    #
    #     t_support = np.arange(arg0, argf + 1, 1) * dt
    #     t_shape = (len(t_support), ) + tuple([1] * (ndim-1))
    #     self.values = self.interpolate(t_support).reshape(t_shape)
    
    def convolve_continuous(self, t, x):
        """Implements the convolution of a time series with the kernel

        Args:
            t (array): time points
            x (array): time series to be convolved
            mode (str): 

        Returns:
            array: convolved time series
        """
        
        dt = get_dt(t)
        arg_support0, arg_supportf = get_arg_support(dt, self.support)

        if isinstance(self, KernelValues):
            kernel_values = self.values
        else:
            t_support = np.arange(arg_support0, arg_supportf, 1) * dt
            kernel_values = self.interpolate(t_support)
        
        shape = (kernel_values.shape[0], ) + tuple([1] * (x.ndim - 1))
        kernel_values = kernel_values.reshape(shape)

        convolution = np.zeros(x.shape)
        
        full_convolution = fftconvolve(kernel_values, x, mode='full', axes=0)

        if arg_support0 >= 0:
            convolution[arg_support0:, ...] = full_convolution[:len(t) - arg_support0, ...]
        elif arg_support0 < 0 and arg_supportf >= 0: # or to arg_support0 < 0 and len(t) - arg_support0 <= len(t) + arg_supportf - arg_support0:
            convolution = full_convolution[-arg_support0:len(t) - arg_support0, ...]
        else: # or arg0 < 0 and len(t) - arg0 > len(t) + arg_supportf - arg0:
            convolution[:len(t) + arg_supportf, ...] = full_convolution[-arg_supportf:, ...]
                
        convolution *= dt
        
        return convolution

    def correlate_continuous(self, t, I, mode='fft'):
        return self.convolve_continuous(t, I[::-1], mode=mode)[::-1]

    def fit(self, t, I, v, mask=None):

        if mask is None:
            mask = np.ones(I.shape, dtype=bool)

        X = self.convolve_basis_continuous(t, I)
        X = X[mask, :]
        v = v[mask]

        self.coefs = np.linalg.lstsq(X, v, rcond=None)[0]
        
    # def deconvolve_continuous(self, t, x, mask=None):
    #
    #     if mask is None:
    #         mask = np.ones(x.shape, dtype=bool)
    #
    #     X = self.convolve_basis_continuous(t, x)
    #     X = X[mask, :]
    #     v = v[mask]
    #
    #     self.coefs = np.linalg.lstsq(X, v, rcond=None)[0]

    def convolve_discrete(self, t, s, A=None, shape=None, renewal=False):
        
        # Given a 1d-array t and a tuple of 1d-arrays s=(tjs, shape) containing timings in the
        # first 1darray of the tuple returns the convolution of the kernels with the timings
        # the convolution of the kernel with the timings. conv.ndim = s.ndim and
        # conv.shape = (len(t), max of the array(accross each dimension))
        # A is used as convolution weights. A=(A) with len(A)=len(s[0]).
        # Assumes kernel is only defined on t >= 0
        
        if type(s) is not tuple:
            s = (s,)
            
        if A is None:
            A = (1. for ii in range(s[0].size)) # Instead of creating the whole list/array in memory x use a generator

        if shape is None:
            # max(s[dim]) determines the size of each dimension
            shape = tuple([max(s[dim]) + 1 for dim in range(1, len(s))])

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)

        convolution = np.zeros((len(t), ) + shape)

        for ii, (arg, A) in enumerate(zip(arg_s, A)):

            index = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))])
            if not(renewal):
                convolution[index] += A * self.interpolate(t[arg:] - t[arg])
            else:
                convolution[index] = A * self.interpolate(t[arg:] - t[arg])
                
        return convolution

class KernelValues(Kernel):

    def __init__(self, values=None, support=None):
        self.values = values
        self.support = np.array(support)
