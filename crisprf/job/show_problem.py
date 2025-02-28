import argparse

from crisprf.util import SRTDataset, gen_noise, plot_sample


def show_problem(snrs: list[float], out_path: str):
    dataset = SRTDataset(snr=None, device="cpu")
    sample = dataset[0]
    sample["y_noise_snr=inf"] = sample["y"]
    for _snr in snrs:
        noise = gen_noise(sample["y"], dT=dataset.shapes.dT, snr=_snr)
        sample[f"y_noise_snr={_snr:.1f}"] = sample["y"] + noise

    plot_sample(
        key_excl=["y", "y_noise", "y_hat"],
        **sample,
        save_path=out_path,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--snrs", type=float, nargs="+", default=[])
    parser.add_argument("-o", "--out-path", type=str, default="fig/problem.png")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    show_problem(args.snrs, args.out_path)
