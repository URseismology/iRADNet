import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
from glob import iglob


def plot_from_log(
    log_path: str,
    ax: plt.Axes,
    ax2: plt.Axes,
    snr: float = None,
):
    df = pd.read_json(log_path, lines=True)
    if snr is not None:
        df = df[df["snr"] == snr]
    else:
        df = df[df["snr"].isna()]
    
    # get last record of x(0)-x(10) 11 values
    df = df.tail(11)
    # get time interval of x(n)-x(0), and convert ns to ms
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = df["timestamp"] // 1_000_000
    # we only need x(1)-x(10) time elapsed, because x(0) is all-zero, not useful
    df = df.tail(10)

    # convert to percentage
    df["density"] = df["density"] * 100

    # traslate to model name in the paper
    filename = os.path.basename(log_path).split(".")[0]
    PATH_TO_MODEL_TRANSLATION = {
        "FISTA": "SRT-FISTA",
        "SRT_LISTA": "iRADNet",
        "SRT_LISTA_CP": "iRADNet",
        "SRT_AdaLISTA": "iRADNet",
        "SRT_AdaLFISTA": "iRADNet",
    }
    PATH_TO_SIZE_TRANSLATION = {
        "FISTA": 2,
        "SRT_LISTA": 2,
        "SRT_LISTA_CP": 1,
        "SRT_AdaLISTA": 1,
        "SRT_AdaLFISTA": 1,
    }
    PATH_TO_VARIANT_TRANSLATION = {
        "FISTA": None,
        "SRT_LISTA": "vanilla LISTA",
        "SRT_LISTA_CP": "LISTA-CP",
        "SRT_AdaLISTA": "AdaLISTA",
        "SRT_AdaLFISTA": "AdaLFISTA",
    }

    model_name = PATH_TO_MODEL_TRANSLATION.get(filename, filename)
    variant_name = PATH_TO_VARIANT_TRANSLATION.get(filename, None)
    full_name = f"{model_name} ({variant_name})" if variant_name else model_name
    # print(
    #     model_name,
    #     snr,
    #     f'{df.tail(1)["NMSE"].item():.4f}',
    #     f'{df.tail(1)["density"].item():.2f}',
    # )

    ax = sns.lineplot(
        data=df,
        x="timestamp",
        y="NMSE",
        label=full_name,
        marker="o",
        linewidth=PATH_TO_SIZE_TRANSLATION[filename],
        ax=ax,
    )
    # dashed line
    ax2 = sns.lineplot(
        data=df,
        x="timestamp",
        y="density",
        linestyle=":",
        marker="^",
        linewidth=PATH_TO_SIZE_TRANSLATION[filename],
        ax=ax2,
    )


def plot_each_snr():
    for snr in [1, 2, 5, 10, None]:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.set_ylim(0.6, 1.1)
        ax2.set_ylim(0, 100)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("NMSE")
        ax2.set_ylabel("Density (%)")
        ax.grid()
        ax.set_yticks(
            [0.6, 0.7, 0.8, 0.9, 1, 1.1],
        )
        ax2.set_yticks(
            [0, 20, 40, 60, 80, 100],
        )

        for log_path in iglob("log/*.jsonl"):
            if log_path.startswith("log/train_"):
                continue
            plot_from_log(log_path, ax, ax2, snr)
        plt.tight_layout()
        fig.savefig(f"fig/results_snr={snr}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    plot_each_snr()
