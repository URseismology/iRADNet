import torch
from crisprf.util.constants import TIME_DTYPE, FREQ_DTYPE
from scipy.io import loadmat
from crisprf.util.bridging import retrieve_single_xy
from crisprf.util.constants import dt, T, nfft
from crisprf.model.radon3d import (
    cal_lipschitz,
    init_radon3d_mat,
    radon3d_forward,
    radon3d_forward_adjoint,
    time2freq,
    freq2time,
)


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
    radon_pytorch = init_radon3d_mat(**retrieve_single_xy(), dt=dt)
    # index 2-8193 for matlab, so 1-8192 for pytorch
    _y2_freq = radon3d_forward(x1_freq, radon_pytorch, 1, nfft // 2)
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
    # print(y2_freq - _y2_freq)
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(
    #     2, 2, sharex=True, sharey=True, figsize=(10, 10)
    # )
    # pct = (_y2_freq - y2_freq) / y2_freq
    # sns.heatmap(torch.real(pct), center=0, ax=ax00, vmin=-10, vmax=10)
    # sns.heatmap(torch.real(y2_freq), center=0, ax=ax01, vmin=-1e3, vmax=1e3)
    # sns.heatmap(torch.imag(pct), center=0, ax=ax10, vmin=-10, vmax=10)
    # sns.heatmap(torch.imag(y2_freq), center=0, ax=ax11, vmin=-1e3, vmax=1e3)
    # fig.savefig("log/y2_freq.png")

    assert torch.allclose(_y2_freq, y2_freq)


def test_lipschitz():
    lip_pytorch = cal_lipschitz(radon_matlab, T, 1, 8192)
    print(lip_pytorch, lip_final)


if __name__ == "__main__":
    test_lipschitz()
