import pytest
import torch

from kernel.utils import torch_fftconvolve

def test_convolve_shape():
    """ Tests that output of convolution is N1 + N2 - 1 and that convolution[:, i] involves convolution of x[:, i]"""
    N1, N2, M = 1000, 120, 2
    N = N1 + N2 - 1
    x = torch.randn(N1, 2)
    y = torch.exp(-torch.arange(N2) / 10)[:, None]
    conv = torch_fftconvolve(x, y)
    conv_inv = torch_fftconvolve(y, x)
    assert conv.shape[0] == N
    assert conv_inv.shape[0] == N
    for i in range(2):
        assert torch.all(torch.isclose(torch_fftconvolve(x[:, i], y[:, 0]), conv[:, i]))
    assert torch.all(torch.isclose(conv, conv_inv))


def test_torch_convolve_matches_scipy_direct_convolve():
    from scipy.signal import convolve as scipy_convolve
    N1, N2, M = 1000, 120, 2
    x = torch.randn(N1, 2).double()
    y = torch.exp(-torch.arange(N2) / 10)[:, None].double()
    scipy_conv = torch.from_numpy(scipy_convolve(x, y, mode='full', method='direct'))
    torch_conv = torch_fftconvolve(x, y)
    assert torch.all(torch.isclose(torch_conv, scipy_conv))

    
if __name__ == '__main__':
    pytest.main()
