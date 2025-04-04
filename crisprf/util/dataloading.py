import torch
from torch.utils.data import Dataset

import json
from glob import glob

from .bridging import RFData, RFDataShape, retrieve_single_xy
from .constants import AUTO_DEVICE
from .noise import gen_noise


class SRTDataset(Dataset):

    def __init__(
        self,
        re_path: str = "data/*.mat",
        snr: float | None = None,
        device: torch.device = AUTO_DEVICE,
    ):
        super().__init__()
        if re_path.endswith('.json'):
            with open(re_path) as f:
                # metadata is a {size: [list of station names]} dict
                metadata = json.load(f)["(1201, 500)"]
                self.paths = [f"data/gt/*/{station_name}_*.mat" for station_name in metadata]
        else:
            self.paths = glob(re_path)
        self.device = device
        self.snr = snr
        self.shapes = RFDataShape.from_sample(
            **retrieve_single_xy(*glob(self.paths[0]), device="cpu")
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> RFData:
        sample = retrieve_single_xy(*glob(self.paths[idx]), device=self.device)
        if self.snr is None:
            sample["y_noise"] = sample["y"]
        else:
            noise = gen_noise(sample["y"], dT=self.shapes.dT, snr=self.snr)
            sample["y_noise"] = sample["y"] + noise
        return sample
