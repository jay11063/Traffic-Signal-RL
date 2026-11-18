import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main(
    csv_path="results/learning_curve_seed_0.csv",
    output_path="results/learning_curve_seed_0.png",
):
    series = {"normal": [], "disrupted": []}

    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            episode = int(row["episode"])
            mean_wait = float(row["mean_wait_s"])
            series[row["condition"]].append((episode, mean_wait))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for condition, points in series.items():
        episodes, waits = zip(*sorted(points))
        ax.plot(episodes, waits, marker="o", markersize=4,
                label=condition.capitalize())

    ax.set_xlabel("Training episodes completed")
    ax.set_ylabel("Mean validation wait (s)")
    ax.set_title("Q-learning validation curve (agent seed 0)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()