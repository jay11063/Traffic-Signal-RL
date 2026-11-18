import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main(
    results_dir="results",
    output_name="learning_curve_three_seeds.png",
):
    results_dir = Path(results_dir)
    conditions = ("normal", "disrupted")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for agent_seed in (0, 1, 2):
        csv_path = results_dir / f"learning_curve_seed_{agent_seed}.csv"
        series = {condition: [] for condition in conditions}

        with csv_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                episode = int(row["episode"])
                wait = float(row["mean_wait_s"])
                series[row["condition"]].append((episode, wait))

        for ax, condition in zip(axes, conditions):
            episodes, waits = zip(*sorted(series[condition]))
            ax.plot(
                episodes,
                waits,
                marker="o",
                markersize=3,
                label=f"Agent seed {agent_seed}",
            )

    for ax, condition in zip(axes, conditions):
        ax.set_title(condition.capitalize())
        ax.set_xlabel("Training episodes completed")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean validation wait (s)")
    axes[1].legend()
    fig.suptitle("Q-learning validation curves across agent seeds")
    fig.tight_layout()
    fig.savefig(results_dir / output_name, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()