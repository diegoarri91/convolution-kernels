import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from .utils import get_arg_support, get_dt, searchsorted
from .base import Kernel


class KernelFun(Kernel):

    def __init__(self, fun, basis_kwargs, shared_kwargs=None, support=None, coefs=None, prior=None, prior_pars=None):
        super().__init__(prior=prior, prior_pars=prior_pars)
        self.fun = fun
        self.basis_kwargs = {key:np.array(val)[None, :] for key, val in basis_kwargs.items()}
        self.shared_kwargs = shared_kwargs if shared_kwargs is not None else {}
        self.support = np.array(support)
        self.nbasis = list(self.basis_kwargs.values())[0].shape[1]
        self.coefs = np.array(coefs) if coefs is not None else np.ones(self.nbasis)

    def copy(self):
        kernel = KernelFun(self.fun, basis_kwargs=self.basis_kwargs.copy(),
                           shared_kwargs=self.shared_kwargs.copy(), support=self.support.copy(),
                           coefs=self.coefs.copy(), prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel

    def area(self, dt):
        return np.sum(self.interpolate(np.arange(self.support[0], self.support[1] + dt, dt))) * dt

    def interpolate(self, t):
        arg0, argf = searchsorted(t, self.support)
        values = np.zeros(len(t))
        values[arg0:argf] = np.sum(self.coefs[None, :] * self.fun(t[arg0:argf, None], **self.basis_kwargs, **self.shared_kwargs), 1)
        return values

    def interpolate_basis(self, t):
        arg0, argf = searchsorted(t, self.support)
        basis = np.zeros((len(t), self.nbasis))
        basis[arg0:argf] = self.fun(t[arg0:argf, None], **self.basis_kwargs, **self.shared_kwargs)
        return basis

    def convolve_basis_continuous(self, t, x):
        """# Given a 1d-array t and an nd-array x with x.shape=(len(t),...) returns X_te,
        # the convolution matrix of each rectangular function of the base with axis 0 of x for all other axis values
        # so that X_te.shape = (x.shape, nbasis)
        # Discrete convolution can be achieved by using an x with 1/dt on the correct timing values
        Assumes sorted t"""

        dt = get_dt(t)
        arg_support0, arg_supportf = get_arg_support(dt, self.support)

        basis_shape = tuple([arg_supportf - arg_support0] + [1 for ii in range(x.ndim - 1)] + [self.nbasis])
        t_support = np.arange(arg_support0, arg_supportf, 1) * dt
        basis = self.interpolate_basis(t_support).reshape(basis_shape)

        output = np.zeros(x.shape + (self.nbasis, ))
        full_convolution = fftconvolve(basis, x[..., None], axes=0)
        
        if arg_support0 >= 0:
            output[arg_support0:, ...] = full_convolution[:len(t) - arg_support0, ...]
        elif arg_support0 < 0 and arg_supportf >= 0:
            output = full_convolution[-arg_support0:len(t) - arg_support0, ...]
        else:
            output[:len(t) + arg_supportf, ...] = full_convolution[-arg_supportf:, ...]
        
        output = output * dt

        return output

    def convolve_basis_discrete(self, t, s, shape=None):

        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
        arg0, argf = searchsorted(t, self.support)

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )

        X = np.zeros(shape)

        kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}

#         basis = self.fun(t[:argf, None], **kwargs).reshape(basis_shape)
        
        for ii, arg in enumerate(arg_s):
            indices = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))] + [slice(0, self.nbasis)])
#             print(ii, self.nbasis, self.fun(t[arg:, None] - t[arg], **kwargs).shape)
            X[indices] += self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]), self.nbasis))

        return X
    
    @classmethod
    def gaussian(cls, tau, A, tm=0, support=None):
        coefs = np.array([A])
        support = support if support is not None else np.array([-5 * tau, 5 * tau]) + tm
        return cls(fun=lambda t, tau: np.exp(-((t - tm) / tau)**2), basis_kwargs=dict(tau=np.array([tau])),
                   support=support, coefs=coefs)

    @classmethod
    def gaussian_delta(cls, delta, tm=0, support=None):
        return cls.gaussian(np.sqrt(2) * delta, 1 / np.sqrt(2 * np.pi * delta ** 2), tm=tm, support=support)
    
    @classmethod
    def single_exponential(cls, tau, A=1, tm=0, support=None):
        support = support if support is not None else [tm, tm + 10 * tau]
        return cls(fun=lambda t, tau: np.exp(-(t - tm) / tau), basis_kwargs=dict(tau=np.array([tau])), support=support,
                   coefs=np.array([A]))
    
    @classmethod
    def exponential(cls, tau, coefs=None, support=None):
        coefs = coefs if coefs is not None else np.ones(len(tau))
        support = support if support is not None else [0, 10 * np.max(tau)]
        return cls(fun=lambda t, tau: np.exp(-t / tau), basis_kwargs=dict(tau=np.array(tau)), support=support,
                   coefs=coefs)
