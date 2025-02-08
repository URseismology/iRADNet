from crisprf.util.constants import T, P, Q
from crisprf.util.dataloading import SRTDataset
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def test_srt_dataset():
    # test if dataset is loaded correctly
    expected_shape = {
        "rayP": (P,),
        "t": (T,),
        "x": (T, Q),
        "y": (T, P),
        "q": (Q,),
    }

    dataset = SRTDataset()
    for sample in dataset:
        for k, v in expected_shape.items():
            assert sample[k].shape == v


def test_srt_dataset_cuda():
    # test if data is loaded to GPU
    if torch.cuda.is_available():
        dataset = SRTDataset(device=torch.device("cuda"))
        sample = dataset[0]
        assert sample["x"].get_device() >= 0


if __name__ == "__main__":
    test_srt_dataset()
