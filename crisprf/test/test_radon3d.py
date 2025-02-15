import torch
from scipy.io import loadmat

import numpy as np

from crisprf.model.radon3d import (
    cal_lipschitz,
    freq2time,
    init_radon3d_mat,
    radon3d_forward,
    radon3d_forward_adjoint,
    time2freq,
)
from crisprf.util.bridging import retrieve_single_xy
from crisprf.util.constants import FREQ_DTYPE, TIME_DTYPE, P, Q, T, dt, nfft

radon_matlab_trace = loadmat("log/radon_test.mat")

# matlab trace
radon_matlab = torch.tensor(radon_matlab_trace["LLL"])
assert radon_matlab.dtype == FREQ_DTYPE
lip_final = radon_matlab_trace["L"]

# randon init
x1 = torch.tensor(radon_matlab_trace["x1"])
assert x1.dtype == TIME_DTYPE
x1_freq = torch.tensor(radon_matlab_trace["x1_freq"])
assert x1_freq.dtype == FREQ_DTYPE

# steps
y2_freq = torch.tensor(radon_matlab_trace["y2_freq"])
assert y2_freq.dtype == FREQ_DTYPE
x2 = torch.tensor(radon_matlab_trace["x2"])
assert x2.dtype == TIME_DTYPE
x2_normed = torch.tensor(radon_matlab_trace["x2_normed"])
assert x2_normed.dtype == TIME_DTYPE
x2_freq = torch.tensor(radon_matlab_trace["x2_freq"])
assert x2_freq.dtype == FREQ_DTYPE


def test_init_radon3d():
    # test the initialization
    # of the 3d radon matrix
    radon_pytorch = init_radon3d_mat(**retrieve_single_xy(), dt=dt)
    assert torch.allclose(radon_pytorch, radon_matlab)


def test_lip0():
    # make sure x1 is rand [0, 1)
    assert torch.allclose(torch.zeros(1, dtype=TIME_DTYPE), x1.min(), atol=1e-6)
    assert torch.allclose(torch.ones(1, dtype=TIME_DTYPE), x1.max(), atol=1e-6)

    # freq domain x
    # ensure time2freq function is correct
    assert torch.allclose(time2freq(x1, nfft), x1_freq)


def test_lip1():
    # index 2-8193 for matlab, so 1-8192 for pytorch
    _y2_freq = radon3d_forward(x1_freq, radon_matlab, 1, nfft // 2)
    # y[0] == zeros
    assert torch.allclose(_y2_freq[0], torch.zeros_like(_y2_freq[0]))
    # y[nfft // 2] == zeros
    assert torch.allclose(_y2_freq[nfft // 2], torch.zeros_like(_y2_freq[0]))
    # other than those two, y is not zeros
    assert not torch.allclose(_y2_freq[nfft // 2 - 1], torch.zeros_like(_y2_freq[0]))
    assert not torch.allclose(_y2_freq[nfft // 2 + 1], torch.zeros_like(_y2_freq[0]))
    assert not torch.allclose(_y2_freq[-1], torch.zeros_like(_y2_freq[0]))

    # test for value correctness
    assert _y2_freq.dtype == FREQ_DTYPE

    assert y2_freq.shape == _y2_freq.shape
    assert torch.allclose(_y2_freq[1:], y2_freq[1:])


def test_lipschitz():
    lip_pytorch = cal_lipschitz(radon_matlab, T, 1, 8192)
    # some randomness in init random x, so be generous
    assert np.allclose(lip_pytorch, lip_final, rtol=1e-3)


if __name__ == "__main__":
    # test_lipschitz()
    pass
