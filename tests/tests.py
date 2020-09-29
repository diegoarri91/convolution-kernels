import numpy as np
import pytest

from kernel.fun import KernelFun
from kernel.utils import searchsorted


@pytest.fixture()
def exponential_kernel_white_noise():
    dt = 0.2
    td = 10

    support = np.array([0, 10 * td])
    ker = KernelFun.single_exponential(td, support=support)
    # t_int = np.arange(0, 10 * td, dt)
    # ker_vals = ker.interpolate(t_int)

    t = np.arange(0, 500, dt)

    x = np.random.randn(len(t), 1)
    y = ker.convolve_continuous(t, x)

    return ker, t, dt, x, y

@pytest.fixture()
def exponential_kernels_shifted():
    dt = 1
    td = 10

    ker = KernelFun.single_exponential(td, tm=0, support=[0, 100])
    ker_left = KernelFun.single_exponential(td, tm=-10, support=[-10, 90])
    ker_right = KernelFun.single_exponential(td, tm=10, support=[10, 110])

    t = np.arange(0, 500, dt)
    x = np.random.randn(len(t), 1)
    y = ker.convolve_continuous(t, x)
    y_left = ker_left.convolve_continuous(t, x)
    y_right = ker_right.convolve_continuous(t, x)

    return ker, ker_left, ker_right, t, dt, x, y, y_left, y_right

def test_searchsorted(exponential_kernel_white_noise):
    ker, t, dt, x, y_true = exponential_kernel_white_noise
    arg0, argf = searchsorted(t, ker.support)
    t = np.arange(arg0 - 10, argf + 10, 1) * dt
    y = ker.interpolate(t)
    assert np.isclose(y[arg0], 1) and np.isclose(y[argf], 0)

def test_convolve_continuous(exponential_kernel_white_noise):
    ker, t, dt, x, y_true = exponential_kernel_white_noise
    dt = t[1]
    ker_vals = ker.interpolate(np.arange(ker.support[0], ker.support[1], dt))
    y = np.array(
        [np.sum(ker_vals[:min(u + 1, len(ker_vals))][::-1] * \
                x[max(0, u + 1 - len(ker_vals)):u + 1, 0]) \
         for u in range(len(t))])
    y = y * dt
    assert np.all((y_true[:, 0] - y) < 5e-5)
    
def test_convolve_continuous_shifting(exponential_kernels_shifted):
    ker, ker_left, ker_right, t, dt, x, y, y_left, y_right = exponential_kernels_shifted
    assert np.allclose(y[:-10], y_right[10:])
    assert np.allclose(y[10:], y_left[:-10])
    
def test_convolve_continuous_basis(exponential_kernels_shifted):
    ker_center, ker_left, ker_right, t, dt, x, y, y_left, y_right = exponential_kernels_shifted

    t = np.arange(0, 200, 1)
    signal = np.random.randn(len(t))
    
    for ker in [ker_center, ker_left, ker_right]:
        convolved_signal = ker.convolve_continuous(t, signal)
        y_basis = ker.convolve_basis_continuous(t, signal)
        for ii in range(2):
            _ker = KernelFun(exponential, basis_kwargs=dict(tau=[taus[ii]]), shared_kwargs=ker.shared_kwargs, support=ker.support, coefs=[1])
            _y = _ker.convolve_continuous(t, signal)
            assert np.allclose(y_basis[:, ii], _y)
