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
                 weight: Optional[Tensor] = None, 
                 requires_grad: bool = True
                 ):
        self.fun = fun
        self.nbasis = len(basis_kwargs)
        self.dtype = list(basis_kwargs.values())[0].dtype
        self.basis_kwargs = {} if basis_kwargs is None else {key: val.unsqueeze(0) for key, val in basis_kwargs.items()}
        super(KernelFun, self).__init__(basis=None, support=support, weight=weight, requires_grad=requires_grad)

    def clone(self):
        basis_kwargs = {key: val.clone() for key, val in self.basis_kwargs.items()}
        kernel = KernelFun(fun=self.fun, basis_kwargs=basis_kwargs, support=self.support,
                           weight=self.weight.detach().clone(), requires_grad=self.weight.requires_grad)
        return kernel

    def evaluate_basis(self, t: Tensor):
        dt = get_timestep(t)
        support_start, support_end = index_evenstep(dt, self.support, start=t[0])
        support_start = support_start if support_start >= 0 else 0
        support_end = support_end if support_end <= len(t) else 0
        basis_values = torch.zeros(len(t), self.nbasis)
        basis_values[support_start:support_end] = self.fun(t[support_start:support_end, None], **self.basis_kwargs)
        return basis_values

    @classmethod
    def exponential(cls, tau: Tensor, support: Optional[Tensor] = None, weight: Optional[Tensor] = None,
                    requires_grad: bool = True):
        support = support if support is not None else torch.tensor([0, 10 * torch.max(tau)])
        exp_fun = lambda t, tau: torch.exp(-t / tau)
        return cls(exp_fun, basis_kwargs=dict(tau=tau), support=support, weight=weight, requires_grad=requires_grad)

    @classmethod
    def gaussian(cls, sigma: Tensor, support: Optional[Tensor] = None, weight: Optional[Tensor] = None,
                 requires_grad: bool = True):
        max_sigma = torch.max(sigma)
        support = support if support is not None else torch.tensor([-5 * max_sigma, 5 * max_sigma])
        pi_torch = torch.tensor([pi])
        gaussian_fun = lambda t, sigma: torch.exp(-(t / (2. * sigma))**2) / (torch.sqrt(2. * pi_torch) * sigma)
        return cls(gaussian_fun, basis_kwargs=dict(sigma=sigma), support=support, weight=weight, 
                   requires_grad=requires_grad)

#     @classmethod
#     def gaussian_delta(cls, delta, tm=0, support=None):
#         return cls.gaussian(np.sqrt(2) * delta, 1 / np.sqrt(2 * np.pi * delta ** 2), tm=tm, support=support)
    
#     @classmethod
#     def single_exponential(cls, tau, A=1, tm=0, support=None):
#         support = support if support is not None else [tm, tm + 10 * tau]
#         return cls(fun=lambda t, tau: np.exp(-(t - tm) / tau), basis_kwargs=dict(tau=np.array([tau])), support=support,
#                    coefs=np.array([A]))
