import torch
from torch.utils.data import Dataset

from glob import glob

from .bridging import RFData, retrieve_single_xy


class SRTDataset(Dataset):
    def __init__(
        self,
        re_path: str = "data/*.mat",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.paths = glob(re_path)
        self.device = device

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> RFData:
        return retrieve_single_xy(self.paths[idx], device=self.device)
