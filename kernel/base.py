import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Module
from typing import Optional

from .utils import idx_evenstep, pad_dimensions, torch_convolve


class Kernel(Module):
    r""" Implements the convolution of a signal with the kernel.

        Args:
            basis (Tensor): Kernel basis values of shape (L, K) where L is the length of the kernel and K is the number
            of basis functions or (L,) if there is a single basis function.
            support (Tensor, optional): 1d Tensor with two values that defines the start and end of the kernel support.
            Default is start=0 and end=L.
            weight (Tensor, optional): The optional predefined weights of shape (K,) for each basis element.
    """
    def __init__(self,
                 basis: Tensor,
                 support: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None
                 ):
        super(Kernel, self).__init__()
        self.basis = basis.reshape(len(basis), 1) if basis is not None and basis.ndim == 1 else basis
        self.support = support if support is not None else torch.tensor([0, basis.shape[0]])
        self.weight = weight if weight is not None else torch.randn(self.basis.shape[1])
        self.weight = Parameter(self.weight)

        if basis is not None and basis.ndim > 2:
            raise ValueError('basis should be a Tensor of ndim <= 2')
        if support.shape != (2,):
            raise ValueError('support must be of shape (2,)')
        if basis is not None and weight.shape == (basis.shape[1],):
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
        support_start, support_end = idx_evenstep(dt, self.support, floor=[False, True])

        if self.basis is None:
            t_support = torch.arange(support_start, support_end, 1) * dt
            kernel_values = self.interpolate(t_support)
        else:
            kernel_values = self.basis @ self.weight

        kernel_values = pad_dimensions(kernel_values, x.ndim - 1)
        convolution = torch_convolve(x, kernel_values, dim=0, mode=mode) * dt

        if trim:
            if support_start >= 0:
                pad = torch.zeros((support_start,) + x.shape[1:])
                convolution = torch.cat((pad, convolution[:size - support_start, ...]), dim=0)
            elif support_start < 0 <= support_end:
                # or support_start < 0 and size - support_start <= size + support_end - support_start:
                convolution = convolution[-support_start:size - support_start, ...]
            else:  # or support_start < 0 and size - support_start > size + support_end - support_start:
                pad = torch.zeros((-support_end,) + x.shape[1:])
                convolution = torch.cat((convolution[:-support_end, ...], pad), dim=0)

        return convolution
