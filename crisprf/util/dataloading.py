import torch
from torch.utils.data import Dataset
from glob import glob
from crisprf.util.bridging import retrieve_single_xy, RFData


class SRTDataset(Dataset):
    def __init__(
        self,
        re_path: str = "data/Ps_RF_*.mat",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.paths = glob(re_path)
        self.device = device

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> RFData:
        return retrieve_single_xy(self.paths[idx], device=self.device)


if __name__ == "__main__":
    from pprint import pprint
    import seaborn as sns

    dataset = SRTDataset()
    sample = dataset[0]
    pprint(sample)
