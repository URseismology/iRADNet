from glob import iglob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import ast
import numpy as np


os.chdir("/home/wmeng/crisprf")


def plot_from_log(
    log_path: str,
    ax: plt.Axes,
    ax2: plt.Axes,
    snr: float = None,
):
    model_name = os.path.basename(log_path).split(".")[0]
    df = pd.read_csv(log_path)
    df["settings"] = df["settings"].apply(ast.literal_eval)
    df = df.join(pd.json_normalize(df["settings"])).drop(columns=["settings"])
    if snr is not None:
        df = df[df["snr"] == snr]
    else:
        df = df[df["snr"].isna()]
    df = df.tail(11)
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df = df.tail(10)
    df["timestamp"] = df["timestamp"] // 1_000_000
    df["timestamp"] = df["timestamp"]
    df["nonzeros"] = df["nonzeros"] / 2450 / 200 * 100
    print(
        model_name,
        snr,
        f'{df.tail(1)["nmse"].item():.4f}',
        f'{df.tail(1)["nonzeros"].item():.2f}',
    )
    # get the first 10 rows

    ax = sns.lineplot(
        data=df,
        x="timestamp",
        y="nmse",
        label=model_name,
        marker="o",
        ax=ax,
    )
    # dashed line
    ax2 = sns.lineplot(
        data=df,
        x="timestamp",
        y="nonzeros",
        linestyle=":",
        marker="^",
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
        ax2.set_ylabel("Non-Zeros (%)")
        ax.grid()
        ax.set_yticks(
            [0.6, 0.7, 0.8, 0.9, 1, 1.1],
        )
        ax2.set_yticks(
            [0, 20, 40, 60, 80, 100],
        )

        for log_path in iglob("log/*.csv"):
            if log_path.startswith("log/train_"):
                continue
            plot_from_log(log_path, ax, ax2, snr)
        plt.tight_layout()
        fig.savefig(f"fig/results_snr={snr}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    plot_each_snr()
