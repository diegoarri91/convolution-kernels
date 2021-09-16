import pytest
import torch

from kernel.base import Kernel

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
