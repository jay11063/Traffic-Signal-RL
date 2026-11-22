import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main(
    results_dir="results",
    output_name="learning_curve_three_seeds.png",
    history_dir=None,
):
    results_dir = Path(results_dir)
    conditions = ("normal", "disrupted")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    sources = [(results_dir, "", "-")]
    if history_dir is not None:
        sources = [
            (results_dir, "Current", "-"),
            (Path(history_dir), "History", "--"),
        ]

    for directory, variant, line_style in sources:
        for seed in (0, 1, 2):
            path = directory / f"learning_curve_seed_{seed}.csv"
            series = {condition: [] for condition in conditions}

            with path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    episode = int(row["episode"])
                    wait = float(row["mean_wait_s"])
                    series[row["condition"]].append((episode, wait))

            label = f"{variant} · seed {seed}" if variant else f"Agent seed {seed}"
            for ax, condition in zip(axes, conditions):
                episodes, waits = zip(*sorted(series[condition]))
                ax.plot(
                    episodes, waits,
                    color=f"C{seed}",
                    linestyle=line_style,
                    marker="o",
                    markersize=3,
                    label=label,
                )

    for ax, condition in zip(axes, conditions):
        ax.set_title(condition.capitalize())
        ax.set_xlabel("Training episodes completed")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean validation wait (s)")
    axes[1].legend()
    title = (
        "Current-observation and history-augmented Q-learning"
        if history_dir is not None
        else "Q-learning validation curves across agent seeds"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(results_dir / output_name, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()