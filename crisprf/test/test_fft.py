import torch
import torch.fft as fft


def test_fft():
    n = 5000
    nfft = 16384
    x = torch.rand((n, 38))
    x_freq = fft.fft(x, n=nfft, dim=0)
    xp = torch.real(fft.ifft(x_freq, dim=0)[:n])
    assert torch.allclose(x, xp, atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    test_fft()
