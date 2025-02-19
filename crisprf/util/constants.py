import torch

TIME_DTYPE = torch.float64
FREQ_DTYPE = torch.complex128

AUTO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
