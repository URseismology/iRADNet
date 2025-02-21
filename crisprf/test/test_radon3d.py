import torch
from scipy.io import loadmat

import numpy as np

import os.path as osp
import pytest

from crisprf.model.radon3d import (
    cal_lipschitz,
    freq2time,
    init_radon3d_mat,
    radon3d_forward,
    radon3d_forward_adjoint,
    time2freq,
)
from crisprf.util.bridging import RFDataShape
from crisprf.util.constants import FREQ_DTYPE, TIME_DTYPE

TRACE_PATH = "log/radon_test.mat"
TRACE_EXISTS = osp.exists(TRACE_PATH)
TRACE_REASON = "no matlab trace exists"


@pytest.mark.skipif(not TRACE_EXISTS, reason=TRACE_REASON)
class TestRadon3d:
    def setup_method(self):
        radon_matlab_trace = loadmat(TRACE_PATH)

        self.lip = radon_matlab_trace["L"].item()
        self.radon3d = torch.tensor(radon_matlab_trace["LLL"])

        self.x0 = torch.tensor(radon_matlab_trace["x0"])
        self.x0_freq = torch.tensor(radon_matlab_trace["x0_freq"])

        self.y1_freq = torch.tensor(radon_matlab_trace["y1_freq"])
        self.x1 = torch.tensor(radon_matlab_trace["x1"])

        self.qs = torch.tensor(radon_matlab_trace["qs"]).squeeze()
        self.rayP = torch.tensor(radon_matlab_trace["rayP"]).squeeze()
        self.shapes = RFDataShape(
            nt=radon_matlab_trace["nt"].item(),
            nfft=radon_matlab_trace["nfft"].item(),
            dt=radon_matlab_trace["dt"].item(),
            np=self.rayP.numel(),
            nq=self.qs.numel(),
        )

    def test_dtype(self):
        assert self.radon3d.dtype == FREQ_DTYPE
        # no need to test lip, just a scalar

        assert self.x0.dtype == TIME_DTYPE
        assert self.x0_freq.dtype == FREQ_DTYPE
        assert self.y1_freq.dtype == FREQ_DTYPE
        assert self.x1.dtype == TIME_DTYPE

    def test_init_radon3d(self):
        _radon3d = init_radon3d_mat(
            q=self.qs, rayP=self.rayP, shapes=self.shapes, device=self.qs.device
        )
        assert self.radon3d.shape == _radon3d.shape
        assert torch.allclose(self.radon3d, _radon3d)

    def test_fft(self):
        # test fft x0 to freq domain
        assert torch.allclose(self.x0_freq, time2freq(self.x0, self.shapes.nfft))

    def test_radon3d_forward(self):
        nffthalf = self.shapes.nfft // 2
        _y1_freq = radon3d_forward(self.x0_freq, self.radon3d, 1, nffthalf)
        assert _y1_freq.dtype == FREQ_DTYPE

        line_of_zeros = torch.zeros_like(_y1_freq[0])
        assert torch.allclose(line_of_zeros, _y1_freq[0])
        assert torch.allclose(line_of_zeros, _y1_freq[nffthalf])
        assert not torch.allclose(line_of_zeros, _y1_freq[nffthalf - 1])
        assert not torch.allclose(line_of_zeros, _y1_freq[nffthalf + 1])
        assert not torch.allclose(line_of_zeros, _y1_freq[-1])

        assert _y1_freq.shape == self.y1_freq.shape
        assert torch.allclose(_y1_freq, self.y1_freq)

    def test_radon3d_adjoint(self):
        _x1_freq = radon3d_forward_adjoint(
            self.y1_freq, self.radon3d, 1, self.shapes.nfft // 2
        )
        assert _x1_freq.dtype == FREQ_DTYPE

        _x1 = freq2time(_x1_freq, self.shapes.nt)
        assert _x1.shape == self.x1.shape
        assert torch.allclose(_x1, self.x1)

    def test_lipschitz(self):
        _lip = cal_lipschitz(self.radon3d, self.shapes.nt, 1, self.shapes.nfft // 2)
        # some randomness in init random x, so be generous
        assert np.allclose(_lip, self.lip, rtol=1e-3)


if __name__ == "__main__":
    t = TestRadon3d()
    t.setup_method()
    t.test_init_radon3d()
    t.test_fft()
    t.test_radon3d_forward()
