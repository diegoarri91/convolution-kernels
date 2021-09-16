import pytest
from scipy.signal import convolve as scipy_convolve
import torch

from kernel.utils import torch_directconvolve, torch_fftconvolve

Tx, Ty, N, K = 1000, 120, 5, 3
Tout = Tx + Ty - 1

@pytest.fixture
def example_convolution_inputs_2d():
    x = torch.randn(Tx, N).double()
    y = torch.exp(-torch.arange(Ty) / 10)[:, None].double()
    return x, y

@pytest.fixture
def example_convolution_inputs_3d():
    tau = 10 + 90 * torch.rand(K)
    x = torch.randn(Tx, N, 1).double()
    y = torch.exp(-torch.arange(Ty)[:, None] / tau[None, :]).unsqueeze(1).double()
    return x, y

def test_convolve2d_shape(example_convolution_inputs_2d):
    x, y = example_convolution_inputs_2d
    torch_dirconv = torch_directconvolve(x, y)
    torch_fftconv = torch_fftconvolve(x, y)
    assert torch_dirconv.shape == (Tout, N)
    assert torch_fftconv.shape == (Tout, N)

def test_convolve3d_shape(example_convolution_inputs_3d):
    x, y = example_convolution_inputs_3d
    torch_dirconv = torch_directconvolve(x, y)
    torch_fftconv = torch_fftconvolve(x, y)
    assert torch_dirconv.shape == (Tout, N, K)
    assert torch_fftconv.shape == (Tout, N, K)

@pytest.mark.parametrize(
    "inputs",
    [
        "example_convolution_inputs_2d",
        'example_convolution_inputs_3d'
    ],
)
def test_torch_convolve_matches_scipy_direct_convolve(inputs, request):
    x, y = request.getfixturevalue(inputs)
    scipy_conv = torch.from_numpy(scipy_convolve(x, y, mode='full', method='direct'))
    torch_dirconv = torch_directconvolve(x, y)
    torch_fftconv = torch_fftconvolve(x, y)
    assert torch.all(torch.isclose(torch_dirconv, scipy_conv))
    assert torch.all(torch.isclose(torch_fftconv, scipy_conv))

# def test_delta_kernels():
#     pass
    
if __name__ == '__main__':
    pytest.main()
