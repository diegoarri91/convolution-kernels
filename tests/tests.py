import pytest
import torch

from ..kernel.utils import torch_fftconvolve

def test_convolve_shape():
    N, M = 1000, 2
    x = torch.randn(1000, 2)
    y = torch.exp(-torch.arange(120) / 10)[:, None]
    conv = torch_fftconvolve(x, y)
    conv_inv = torch_fftconvolve(y, x)
    for i in range(2):
        assert conv.shape[i] == N
        assert conv_inv.shape[i] == N
        assert torch.all(torch_fftconvolve(x[:, i], y[:, 0]) == conv[:, i])
    assert torch.all(conv == conv_inv)

    
if __name__ == '__main__':
    pytest.main()