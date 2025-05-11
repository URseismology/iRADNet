import torch

import pytest

from crisprf.util import RFDataShape, SRTDataset


def test_srt_dataset():
    dataset = SRTDataset("data/*.mat")
    for sample in dataset:
        RFDataShape.verify(sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_srt_dataset_cuda():
    # test if data can be loaded to GPU
    dataset = SRTDataset(device=torch.device("cuda"))
    sample = dataset[0]
    assert sample["x"].get_device() >= 0


def test_srt_dataset_json():
    dataset = SRTDataset("data/gt/meta.json")
    for sample in dataset:
        RFDataShape.verify(sample)


if __name__ == "__main__":
    # test_srt_dataset()
    dataset = SRTDataset("data/gt/meta.json")
