from math import pi

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union

from .utils import index_evenstep, pad_dimensions, torch_convolve


class Kernel(nn.Module):
    r""" Implements the convolution of a signal with the kernel.

        Args:
            basis (Tensor): Kernel basis values of shape (L, K) where L is the length of the kernel and K is the number
            of basis functions or (L,) if there is a single basis function.
            support (Tensor, optional): 1d Tensor with two values that defines the start and end of the kernel support.
            Default is start=0 and end=L.
            weight (Tensor, optional): The optional predefined weights of shape (K,) for each basis element.
    """
    def __init__(self,
                 basis: Union[Tensor, None],
                 support: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None
                 ):
        super().__init__()
        self.basis = basis.reshape(len(basis), 1) if basis is not None and basis.ndim == 1 else basis
        self.support = support if support is not None else torch.tensor([0, basis.shape[0]])
        self.weight = weight if weight is not None else torch.randn(self.basis.shape[1])
        self.weight = nn.Parameter(self.weight)

        if basis is not None and basis.ndim > 2:
            raise ValueError('basis should be a Tensor of ndim <= 2')
        if self.support.shape != (2,):
            raise ValueError('support must be of shape (2,)')
        if self.basis is not None and self.weight.shape != (self.basis.shape[1],):
            raise ValueError('weight size should match dimension 2 of basis')

    def interpolate(self, t):
        pass

    def interpolate_basis(self, t):
        pass

    def forward(self,
                x: Tensor,
                dt: float = 1.,
                trim: bool = True,
                mode: str = 'fft'
                ) -> Tensor:

        size = x.shape[0]
        support_start, support_end = index_evenstep(dt, self.support, floor=[False, True])

        if self.basis is None:
            t_support = torch.arange(support_start, support_end, 1) * dt
            kernel_values = self.interpolate(t_support)
            print(kernel_values.shape)
        else:
            kernel_values = self.basis @ self.weight

        kernel_values = pad_dimensions(kernel_values, x.ndim - 1)
        convolution = torch_convolve(x, kernel_values, mode=mode) * dt

        if trim:
            if support_start >= 0:
                pad = torch.zeros((support_start,) + x.shape[1:])
                convolution = torch.cat((pad, convolution[:size - support_start, ...]), dim=0)
            elif support_start < 0 <= support_end:
                # or support_start < 0 and size - support_start <= size + support_end - support_start:
                convolution = convolution[-support_start:size - support_start, ...]
            else:  # or support_start < 0 and size - support_start > size + support_end - support_start:
                pad = torch.zeros((-support_end,) + x.shape[1:])
                convolution = torch.cat((convolution[-support_end:, ...], pad), dim=0)
        else:
            raise(NotImplementedError)

        return convolution

    # def convolve_basis_discrete(self, t, s, shape=None):
    #
    #     if type(s) is np.ndarray:
    #         s = (s,)
    #
    #     arg_s = searchsorted(t, s[0])
    #     arg_s = np.atleast_1d(arg_s)
    #     arg0, argf = searchsorted(t, self.support)
    #
    #     if shape is None:
    #         shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
    #     else:
    #         shape = shape + (self.nbasis,)
    #
    #     X = np.zeros(shape)
    #
    #     for ii, arg in enumerate(arg_s):
    #         indices = tuple([slice(arg, min(arg + argf, len(t)))] + [s[dim][ii] for dim in range(1, len(s))] + [
    #             slice(0, self.nbasis)])
    #         X[indices] += self.basis_values[:min(arg + argf, len(t)) - arg, :]
    #
    #     return X

    @classmethod
    def orthogonalized_raised_cosines(cls, dt, last_time_peak, n, b, a=1e0, weight=None):

        range_locs = torch.log(torch.tensor([0, last_time_peak]) + b)
        delta = (range_locs[1] - range_locs[0]) / (n - 1)
        locs = torch.linspace(range_locs[0], range_locs[1], n)

        last_time = torch.exp(range_locs[1] + 2 * delta / a) - b
        t = torch.arange(0, last_time, dt)
        support = torch.tensor([t[0], t[-1] + dt])

        pi_torch = torch.tensor([pi])
        raised_cosines = torch.minimum(a * (torch.log(t[:, None] + b) - locs[None, :]) * pi / delta / 2, pi_torch)
        raised_cosines = (1 + torch.cos(torch.maximum(-pi_torch, raised_cosines))) / 2
        raised_cosines = raised_cosines / torch.sqrt(torch.sum(raised_cosines ** 2, 0))
        u, s, v = torch.linalg.svd(raised_cosines)
        basis = u[:, :n]

        return cls(basis=basis, support=support, weight=weight)

    def clone(self):
        kernel = Kernel(basis=self.basis.clone(), support=self.support.clone(), weight=self.weight.detach().clone())
        return kernel
