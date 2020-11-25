import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from .base import Kernel
from .utils import get_arg_support, get_dt, searchsorted


class KernelBasisValues(Kernel):

    def __init__(self, basis_values, support, dt, coefs=None, prior=None, prior_pars=None):
        super().__init__(prior=prior, prior_pars=prior_pars)
        self.dt = dt
        self.basis_values = basis_values
        self.nbasis = basis_values.shape[1]
        self.coefs = np.array(coefs) if coefs is not None else np.ones(self.nbasis)
        self.support = np.array(support)

    def interpolate(self, t):

        assert np.isclose(self.dt, get_dt(t))

        t = np.atleast_1d(t)
        res = np.zeros(len(t))

        arg0, argf = get_arg_support(self.dt, self.support, t0=t[0])

        if arg0 >= 0:
            argf = min(argf, len(t))
            res[arg0:argf] = np.matmul(self.basis_values, self.coefs)[:argf - arg0]
        elif arg0 < 0 and argf > 0:
            n_times = self.basis_values.shape[0]
            res[:min(len(t), n_times + arg0)] = np.matmul(self.basis_values, self.coefs)[-arg0:min(len(t) - arg0, n_times)]
        
        return res

    def interpolate_basis(self, t):
        
        assert np.isclose(self.dt, get_dt(t))

        t = np.atleast_1d(t)

        arg0, argf = get_arg_support(self.dt, self.support, t0=t[0])

        if arg0 >= 0:
            argf = min(argf, len(t))
            return self.basis_values[:argf - arg0]
        elif arg0 < 0 and argf > 0:
            n_times = self.basis_values.shape[0]
            return self.basis_values[-arg0:min(len(t) - arg0, n_times)]

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
        
        for ii, arg in enumerate(arg_s):
            indices = tuple([slice(arg, min(arg + argf, len(t)))] + [s[dim][ii] for dim in range(1, len(s))] + [slice(0, self.nbasis)])
            X[indices] += self.basis_values[:min(arg + argf, len(t)) - arg, :]

        return X

    @classmethod
    def orthogonalized_raised_cosines(cls, dt, last_time_peak, n, b, a=1e0, coefs=None):

        range_locs = np.log(np.array([0, last_time_peak]) + b)
        delta = (range_locs[1] - range_locs[0]) / (n - 1)  # delta = 1 / (n - 1) * np.log(1 + last_peak / b)
        locs = np.linspace(range_locs[0], range_locs[1], n)

        last_time = np.exp(range_locs[1] + 2 * delta / a) - b
        t = np.arange(0, last_time, dt)
        support = [t[0], t[-1] + dt]

        raised_cosines = (1 + np.cos(np.maximum(-np.pi, np.minimum(
            a * (np.log(t[:, None] + b) - locs[None, :]) * np.pi / delta / 2, np.pi)))) / 2
        raised_cosines = raised_cosines / np.sqrt(np.sum(raised_cosines ** 2, 0))
        u, s, v = np.linalg.svd(raised_cosines)
        basis = u[:, :n]

        return cls(basis_values=basis, support=support, dt=dt, coefs=coefs)

    def copy(self):
        kernel = KernelBasisValues(self.basis_values.copy(), self.support.copy(), self.dt, coefs=self.coefs.copy(), 
                                   prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel
