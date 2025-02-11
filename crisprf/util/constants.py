import torch

TIME_DTYPE = torch.float64
FREQ_DTYPE = torch.complex128


def nextpow2(x: int):
    if x == 0:
        return 0
    return 2 ** (x - 1).bit_length()


T = 5000
P = 38
Q = 200
dt = 0.02
nfft = 2 * nextpow2(T)


if __name__ == "__main__":
    print(
        nextpow2(0),
        nextpow2(1),
        nextpow2(2),
        nextpow2(3),
        nextpow2(4),
        nextpow2(10),
        nextpow2(1000),
        nextpow2(1024),
        nextpow2(5000),
    )
    print(nfft)
