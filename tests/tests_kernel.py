import pytest
import torch

from kernel.base import Kernel


# @pytest.fixture
def exponential_basis_kernel():
    tau = torch.tensor([10, 20])[None, :]
    t_range = torch.arange(0, 20, 1)[:, None]
    basis = torch.exp(-t_range / tau)
    ker = Kernel(basis=basis)
    return ker

t_out_of_range = [
    torch.arange(-20, -10),
    torch.arange(-10, -0),
    torch.arange(20, 30),
    torch.arange(30, 40),
]
@pytest.mark.parametrize("t_out", t_out_of_range)
def test_evaluate_basis_out_of_support(t_out):
    ker = exponential_basis_kernel()
    y = ker.evaluate_basis(t_out)
    assert torch.all(torch.isclose(y, torch.tensor([0.])))


def test_evaluate_basis_shifts():
    ker = exponential_basis_kernel()
    support_range = torch.arange(ker.support[0], ker.support[1], 1)
    basis_support = ker.evaluate_basis(support_range)
    shifted = ker.evaluate_basis(support_range + 10)
    assert torch.all(torch.isclose(shifted[:-10], basis_support[10:]))
    assert torch.all(torch.isclose(shifted[-10:], torch.tensor([0.])))
    shifted = ker.evaluate_basis(support_range - 10)
    assert torch.all(torch.isclose(shifted[10:], basis_support[:-10]))
    assert torch.all(torch.isclose(shifted[:10], torch.tensor([0.])))


def test_delta_kernels():
    ker = Kernel(basis=torch.tensor([1.]),
                 support=torch.tensor([0., 1.]),
                 weight=torch.ones(1))
    ker_right = Kernel(basis=torch.tensor([1.]),
                       support=torch.tensor([10., 11.]),
                       weight=torch.ones(1))
    ker_left = Kernel(basis=torch.tensor([1.]),
                      support=torch.tensor([-11., -10.]),
                      weight=torch.ones(1))
    x = torch.randn(200, 1)
    y = ker(x, mode='direct', trim=True)
    y_right = ker_right(x, mode='direct', trim=True)
    y_left = ker_left(x, mode='direct', trim=True)
    assert torch.all(torch.isclose(y, x))
    assert torch.all(torch.isclose(y_right[10:], x[:-10]))
    assert torch.all(torch.isclose(y_left[:-10], x[10:]))

if __name__ == '__main__':
    pytest.main()
