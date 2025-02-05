from crisprf.util.dataloading import SRTDataset
import seaborn as sns
import matplotlib.pyplot as plt


def test_srt_dataset():
    expected_shape = {
        "tshift": tuple(),
        "rayP": (38,),
        "t": (5000,),
        "x": (200, 5000),
        "y": (38, 5000),
        "q": (200,),
    }

    dataset = SRTDataset()
    for sample in dataset:
        for k, v in sample.items():
            assert v.shape == expected_shape[k]


def plot_sample():
    sample = SRTDataset()[0]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, dpi=100)
    sns.heatmap(
        sample["y"],
        ax=ax0,
        yticklabels=list(map(lambda x: f"{x:.3f}", sample["rayP"].numpy())),
        linewidths=0,
        center=0,
        cmap="coolwarm",
        # [
        #     f"{t:.3f}" if i % 5 == 0 or i == len(sample["rayP"]) - 1 else ""
        #     for i, t in enumerate(sample["rayP"].numpy())
        # ],
    )
    sns.heatmap(
        sample["x"],
        ax=ax1,
        xticklabels=list(map(lambda x: f"{x:.0f}", sample["t"].numpy())),
        yticklabels=list(map(lambda x: f"{x:.0f}", sample["q"].numpy())),
        linewidths=0,
        center=0,
        cmap="coolwarm",
    )
    ax0.locator_params(axis="y", nbins=10)
    ax0.set_ylabel("Ray parameter (s/km)")
    ax0.set_title("Seismic traces")
    ax1.locator_params(axis="both", nbins=10)
    ax1.set_ylabel("q (s/km)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Radon transform")
    plt.tight_layout()
    plt.savefig("fig/teasor.pdf", pad_inches=0)


if __name__ == "__main__":
    test_srt_dataset()
