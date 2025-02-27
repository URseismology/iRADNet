import argparse

from crisprf.util import (
    SRTDataset,
    plot_sample,
)


def show_problem(snr: float | None = None):
    dataset = SRTDataset(snr=snr, device="cpu")
    sample = dataset[0]

    plot_sample(**sample, save_path=f"fig/problem/snr={snr}.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    show_problem(args.snr)
