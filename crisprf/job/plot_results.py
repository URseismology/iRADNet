import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
from glob import glob

# trasnlation to model name in the paper, e.g. FISTA -> SRT-FISTA
PATH_TO_ARCH_TRANSLATION = {
    "FISTA": "SRT-FISTA",
    "SRT_LISTA": "iRADNet",
    "SRT_LISTA_CP": "iRADNet",
    "SRT_AdaLISTA": "iRADNet",
    "SRT_AdaLFISTA": "iRADNet",
}
PATH_TO_VARIANT_TRANSLATION = {
    "FISTA": None,
    "SRT_LISTA": "vanilla LISTA",
    "SRT_LISTA_CP": "LISTA-CP",
    "SRT_AdaLISTA": "AdaLISTA",
    "SRT_AdaLFISTA": "AdaLFISTA",
}


def log_to_df(log_path: str, snr: float = None):
    df = pd.read_json(log_path, lines=True)
    if snr is None:
        df = df[df["snr"].isna()]
    else:
        df = df[df["snr"] == snr]

    # get last record of x(0)-x(10) 11 values
    df = df.tail(11)
    # get time interval of x(n)-x(0)
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    # we only need x(1)-x(10) time elapsed
    df = df.tail(10)

    # convert to percentage
    df["sparsity"] = 100 - (df["density"] * 100)
    # convert to ms
    df["timestamp"] = df["timestamp"] // 1_000_000

    # add filename and model information
    filename = os.path.basename(log_path).split(".")[0]
    df["filename"] = filename
    arch_name = PATH_TO_ARCH_TRANSLATION.get(filename, filename)
    variant_name = PATH_TO_VARIANT_TRANSLATION.get(filename, None)
    full_name = f"{arch_name} ({variant_name})" if variant_name else arch_name
    df["Arch"] = arch_name
    df["Variant"] = variant_name
    df["Model"] = full_name
    return df


def plot_from_log(
    log_path: str,
    ax: plt.Axes,
    ax2: plt.Axes,
    snr: float = None,
):
    df = pd.concat([log_to_df(p, snr) for p in sorted(glob(log_path))], ignore_index=True)

    ax = sns.lineplot(
        data=df,
        x="timestamp",
        y="NMSE",
        # label=full_name,
        marker="o",
        ax=ax,
        palette="cubehelix",
        hue="Model",
        legend=snr is None,
    )
    # dashed line
    ax2 = sns.lineplot(
        data=df,
        x="timestamp",
        y="sparsity",
        linestyle=":",
        marker="^",
        ax=ax2,
        palette="cubehelix",
        hue="Model",
        legend=False,
    )


def plot_each_snr():
    for snr in [1, 2, 5, 10, None]:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.grid()

        ax.set_xlabel("Time (ms)")
        ax.set_ylim(0.6, 1.1)
        ax.set_ylabel("NMSE")

        ax2 = ax.twinx()
        ax2.set_ylim(100, 0)
        ax2.set_ylabel("Sparsity (%)")
        ax.set_yticks(
            [0.6, 0.7, 0.8, 0.9, 1, 1.1],
        )
        ax2.set_yticks(
            [100, 80, 60, 40, 20, 0],
        )

        plot_from_log("log/eval/*.jsonl", ax, ax2, snr)
        plt.tight_layout()
        fig.savefig(f"fig/results_snr={snr}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    plot_each_snr()
