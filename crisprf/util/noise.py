import torch
from scipy.signal import butter, sosfiltfilt

import torchaudio


def add_noise(y: torch.Tensor, dt: float, snr: float = 2.0, uniform: bool = True):
    # y should be shape (nt, np)
    # and y1d_norm norm over nt, giving (np,)
    y1d_norm = torch.linalg.vector_norm(y, ord=2, dim=0)
    assert type(y1d_norm) == torch.Tensor
    assert y1d_norm.shape == (2,), f"{y1d_norm.shape}"

    # init rand matrix, uniform or normal with mean 0; 0.5 is fine
    noise = torch.rand_like(y) - 0.5 if uniform else torch.randn_like(y)
    # scale to desired SNR, snr=signal/noise
    noise = noise * y1d_norm / torch.linalg.vector_norm(noise, ord=2, dim=0) / snr

    # filter with butterworth bandpass
    sos = butter(2, (0.1, 0.5), btype="band", output="sos", fs=1 / dt)
    # and we are doing on time domain so axis=0
    noise_filt = torch.tensor(sosfiltfilt(sos, noise, axis=0).copy())

    # fix for variance to preserve SNR before/after bandpass
    noise_filt *= torch.linalg.vector_norm(noise) / torch.linalg.vector_norm(noise_filt)

    return y + noise_filt


def butter_bandpass_filter_torch(y: torch.Tensor, dt: float):
    # Convert input to tensor and ensure correct shape
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    if y_tensor.dim() == 1:
        y_tensor = y_tensor.unsqueeze(0)  # Add batch dimension

    fs = 1.0 / dt  # Calculate sampling frequency

    # Forward filtering: highpass then lowpass (2nd-order Butterworth)
    y_filt = torchaudio.functional.highpass_biquad(y_tensor, fs, 0.1)
    y_filt = torchaudio.functional.lowpass_biquad(y_filt, fs, 0.5)

    # Reverse filtering for zero-phase
    y_rev = torch.flip(y_filt, dims=[-1])
    y_rev = torchaudio.functional.highpass_biquad(y_rev, fs, 0.1)
    y_rev = torchaudio.functional.lowpass_biquad(y_rev, fs, 0.5)
    y_filtered = torch.flip(y_rev, dims=[-1])

    # Remove batch dimension if input was 1D
    if y.ndim == 1:
        y_filtered = y_filtered.squeeze(0)

    return y_filtered


# Example usage:
# Assuming y is your input signal and dt is the sampling interval
# y_filt = butter_bandpass_filter_torch(y, dt)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    y = (
        torch.tensor(
            [
                [1, 2, 4, 8, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 2, 4, 8, 8, 4, 2, 1, 0, 0, 0, 0],
            ]
        )
        / 8
    ).T
    res = add_noise(y=y, dt=0.1, snr=2.0)
    sns.heatmap(torch.cat([y, res], dim=1), center=0)
    plt.savefig("tmp.png")
