import torch
from scipy.signal import butter, sosfiltfilt

import torchaudio


def gen_noise(
    y: torch.Tensor,
    dt: float,
    snr: float = 2.0,
    uniform: bool = True,
    lowcut: float = 0.1,
    highcut: float = 0.5,
) -> torch.Tensor:
    # y should be shape (nt, np)
    # and y1d_norm norm over nt, giving (np,)
    y1d_norm = torch.linalg.vector_norm(y, ord=2, dim=0)

    # init rand matrix, uniform or normal with mean 0; 0.5 is fine
    noise = torch.rand_like(y) - 0.5 if uniform else torch.randn_like(y)
    # scale to desired SNR, snr=signal/noise
    noise = noise * y1d_norm / torch.linalg.vector_norm(noise, ord=2, dim=0) / snr

    # filter with butterworth bandpass
    sos = butter(2, (lowcut, highcut), btype="band", output="sos", fs=1 / dt)
    # apply on time domain so axis=0
    noise_filt = torch.tensor(
        sosfiltfilt(sos, noise.cpu(), axis=0).copy(), device=y.device
    )

    # fix for variance to preserve SNR before/after bandpass
    noise_filt = (
        noise_filt
        * torch.linalg.vector_norm(noise, ord=2, dim=0)
        / torch.linalg.vector_norm(noise_filt, ord=2, dim=0)
    )

    return noise_filt


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
