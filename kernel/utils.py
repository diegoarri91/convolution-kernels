import torch
import torch.nn.functional as F
from torch import Tensor


def torch_directconvolve(x: Tensor, y: Tensor):
    r""" Implements direct convolution between x and y along dimension dim"""
    padding = y.shape[0] - 1
    input_2d = True if x.ndim == 2 and y.ndim == 2 else False
    if input_2d:
        x, y = x.unsqueeze(2), y.unsqueeze(2)
    x = x.movedim(0, 2)
    y = torch.flip(y, dims=[0]).transpose(0, 2)
    conv = F.conv1d(x, y, padding=padding).movedim(2, 0)
    if input_2d:
        conv = conv.squeeze(2)
    return conv


def torch_fftconvolve(x: Tensor, y: Tensor):
    r""" Implements fft convolution between x and y along dimension dim"""
    n = x.shape[0] + y.shape[0]
    parity = n % 2
    x_fft = torch.fft.rfft(x, n=n + parity, dim=0)
    y_fft = torch.fft.rfft(y, n=n + parity, dim=0)
    conv = torch.fft.irfft(x_fft * y_fft, dim=0)[:-(parity + 1)]
    return conv


def torch_convolve(x: Tensor, y: Tensor, mode: str = 'fft'):
    r""" Implements direct or fft convolution between x and y along dimension dim"""
    if mode == 'fft':
        conv = torch_fftconvolve(x, y)
    elif mode == 'direct':
        conv = torch_directconvolve(x, y)
    else:
        raise ValueError('Mode %s not implemented.' % mode)
    return conv


def index_evenstep(step, values, floor=True, start=0, rtol=1e-5, atol=1e-8):
    scaled = (values - start) / step
    rounded = torch.round(scaled).float()
    isequal = torch.isclose(scaled, rounded, rtol=rtol, atol=atol).int()
    if isinstance(floor, bool):
        if floor:
            side = torch.floor(scaled).int()
        else:
            side = torch.ceil(scaled).int()
        idx = (isequal * rounded + (1 - isequal) * side).int()
    else:
        floor = torch.tensor(floor).int()
        floored = torch.floor(scaled).int()
        ceiled = torch.ceil(scaled).int()
        idx = (isequal * rounded + (1 - isequal) * (floor * floored + (1 - floor) * ceiled)).int()
    return idx


def pad_dimensions(input, pad):
    shape = input.shape + tuple([1] * pad)
    input = input.reshape(shape)
    return input


def get_timestep(t: Tensor):
    assert torch.isclose(torch.diff(t), t[1] - t[0]).all(), 'time values are not evenly spaced'
    return t[1] - t[0]


# def searchsorted(t, s, side='left'):
#     '''
#     Uses np.searchsorted but handles numerical round error with care
#     such that returned index satisfies
#     t[i-1] < s <= t[i]
#     np.searchsorted(side='right') doesn't properly handle the equality sign
#     on the right side
#     '''
#     s = np.atleast_1d(s)
#     arg = np.searchsorted(t, s, side=side)

#     if len(t) > 1:
#         dt = get_dt(t)
#         s_ = (s - t[0]) / dt
#         round_s = np.round(s_, 0)
#         mask_round = np.isclose(s_, np.round(s_, 0)) & (round_s >= 0) & (round_s < len(t))
#         if side == 'left':
#             arg[mask_round] = np.array(round_s[mask_round], dtype=int)
#         elif side == 'right':
#             arg[mask_round] = np.array(round_s[mask_round], dtype=int) + 1
#     else:
#         s_ = s - t[0]
#         mask = np.isclose(s - t[0], 0.)# & (round_s >= 0) & (round_s < len(t))
#         arg[mask] = np.array(s_[mask], dtype=int)

#     if len(arg) == 1:
#         arg = arg[0]

#     return arg


# def shift_array(arr, shift, fill_value=False):
#     """
#     Shifts array on axis 0 filling the shifted values with fill_value
#     Positive shift is to the right, negative to the left
#     """

#     result = np.empty_like(arr)
#     if shift > 0:
#         result[:shift, ...] = fill_value
#         result[shift:, ...] = arr[:-shift, ...]
#     elif shift < 0:
#         result[shift:, ...] = fill_value
#         result[:shift, ...] = arr[-shift:, ...]
#     else:
#         result = arr
#     return result
