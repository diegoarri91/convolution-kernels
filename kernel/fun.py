from math import pi

import torch
from torch import Tensor
from typing import Optional

from .base import Kernel
from .utils import index_evenstep, get_timestep


class KernelFun(Kernel):

    def __init__(self,
                 fun,
                 basis_kwargs,
                 support: Tensor = None,
                 weight: Optional[Tensor] = None
                 ):
        super(KernelFun, self).__init__(basis=None, support=support, weight=weight)
        self.fun = fun
        self.basis_kwargs = {key: val.unsqueeze(0) for key, val in basis_kwargs.items()} if basis_kwargs is not None else {}

    def interpolate(self, t: Tensor):
        dt = get_timestep(t)
        idx_start, idx_end = index_evenstep(dt, self.support, start=t[0])
        values = torch.zeros(len(t))
        basis_values = self.fun(t[idx_start:idx_end].unsqueeze(1), **self.basis_kwargs)
        values[idx_start:idx_end] = basis_values @ self.weight
        return values
    
    @classmethod
    def gaussian(cls, sigma: Tensor, weight: Optional[Tensor] = None, support: Tensor = None): # TODO. Define support type here too
        max_sigma = torch.max(sigma)
        support = support if support is not None else torch.tensor([-5 * max_sigma, 5 * max_sigma])
        pi_torch = torch.tensor([pi])
        gaussian_fun = lambda t, sigma: torch.exp(-(t / (2. * sigma))**2) / (torch.sqrt(2. * pi_torch) * sigma)
        return cls(gaussian_fun, basis_kwargs=dict(sigma=sigma), support=support, weight=weight)

#     @classmethod
#     def gaussian_delta(cls, delta, tm=0, support=None):
#         return cls.gaussian(np.sqrt(2) * delta, 1 / np.sqrt(2 * np.pi * delta ** 2), tm=tm, support=support)
    
#     @classmethod
#     def single_exponential(cls, tau, A=1, tm=0, support=None):
#         support = support if support is not None else [tm, tm + 10 * tau]
#         return cls(fun=lambda t, tau: np.exp(-(t - tm) / tau), basis_kwargs=dict(tau=np.array([tau])), support=support,
#                    coefs=np.array([A]))
    
#     @classmethod
#     def exponential(cls, tau, coefs=None, support=None):
#         coefs = coefs if coefs is not None else np.ones(len(tau))
#         support = support if support is not None else [0, 10 * np.max(tau)]
#         return cls(fun=lambda t, tau: np.exp(-t / tau), basis_kwargs=dict(tau=np.array(tau)), support=support,
#                    coefs=coefs)
