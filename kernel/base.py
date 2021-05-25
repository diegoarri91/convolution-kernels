# import matplotlib.pyplot as plt
import torch

from .utils import idx_evenstep, pad_dimensions, torch_convolve


class Kernel:

    def __init__(self, kernel_values=None):
        self.kernel_values = kernel_values

    def interpolate(self, t):
        pass
    
    def interpolate_basis(self, t):
        pass

    def kernel_values(self):


    def convolve_continuous(self, x, dt=1, trim=True, mode='fft'):
        """Implements the convolution of a time series with the kernel using fftconvolve.

        Args:
            t (array): time points
            x (array): time series to be convolved
            mode (str): 

        Returns:
            array: convolved time series
        """

        size = x.shape[0]
        idx_supporti, idx_supportf = idx_evenstep(dt, self.support, floor=[False, True])

        if self.kernel_values is None:
            t_support = torch.arange(idx_supporti, idx_supportf, 1) * dt
            kernel_values = self.interpolate(t_support)
        else:
            kernel_values = self.kernel_values

        kernel_values = pad_dimensions(kernel_values, x.ndim - 1)

        full_convolution = torch_convolve(x, kernel_values, dim=0, mode=mode)

        if trim:
            convolution = torch.zeros(x.shape)
            if idx_supporti >= 0:
                convolution[idx_supporti:, ...] = full_convolution[:size - idx_supporti, ...]
            elif idx_supporti < 0 and idx_supportf >= 0: # or idx_supporti < 0 and size - idx_supporti <= size + idx_supportf - idx_supporti:
                convolution = full_convolution[-idx_supporti:size - idx_supporti, ...]
            else: # or idx_supporti < 0 and size - idx_supporti > size + idx_supportf - idx_supporti:
                convolution[:size + idx_supportf, ...] = full_convolution[-idx_supportf:, ...]
                
        convolution *= dt
        
        return convolution
