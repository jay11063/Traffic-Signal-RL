import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .config import TEST_SCENARIO_SEEDS


METRICS = (
    "mean_wait_completed",
    "p95_wait_completed",
    "max_queue_N",
    "max_queue_S",
    "max_queue_E",
    "max_queue_W",
)

CONTROLLERS = (
    ("FixedTime", "FixedTime (45, 10)", ("",)),
    ("RuleBased", "RuleBased (5)", ("",)),
    ("current", "Current Q-learning", ("0", "1", "2")),
    ("history", "History Q-learning", ("0", "1", "2")),
)


def main():
    path = Path("results/experiments/test/episodes.csv")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    groups = defaultdict(list)
    for row in rows:
        key = (row["controller"], row["agent_seed"], row["condition"])
        groups[key].append(row)

    expected_seeds = set(TEST_SCENARIO_SEEDS)
    expected_keys = {
        (controller, seed, condition)
        for controller, _, seeds in CONTROLLERS
        for seed in seeds
        for condition in ("normal", "disrupted")
    }
    if set(groups) != expected_keys:
        raise ValueError("Unexpected or missing controller, agent seed, or condition.")

    for condition in ("normal", "disrupted"):
        print(f"\n#### {condition.capitalize()}\n")
        print("| Controller | Agent seed | Mean wait (s) | P95 wait (s) "
              "| Max N | Max S | Max E | Max W |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

        for controller, label, seeds in CONTROLLERS:
            seed_values = []

            for seed in seeds:
                episodes = groups[controller, seed, condition]
                actual_seeds = {int(row["scenario_seed"]) for row in episodes}
                if len(episodes) != len(expected_seeds) or actual_seeds != expected_seeds:
                    raise ValueError(f"Incomplete or duplicate runs: {controller}, {seed}, {condition}")

                values = [
                    mean(float(row[metric]) for row in episodes)
                    for metric in METRICS
                ]
                seed_values.append(values)

                cells = " | ".join(f"{value:.2f}" for value in values)
                print(f"| {label} | {seed or '—'} | {cells} |")

            if len(seeds) > 1:
                cells = " | ".join(
                    f"{mean(column):.2f} ± {stdev(column):.2f}"
                    for column in zip(*seed_values)
                )
                print(f"| **{label}** | **Mean ± SD** | {cells} |")

    unfinished_runs = sum(int(row["remaining"]) > 0 for row in rows)
    remaining_total = sum(int(row["remaining"]) for row in rows)
    print(
        f"\nWaiting statistics cover completed vehicles. Across {len(rows)} runs, "
        f"{unfinished_runs} runs had unfinished vehicles "
        f"({remaining_total} vehicles remaining in total)."
    )


if __name__ == "__main__":
    main()