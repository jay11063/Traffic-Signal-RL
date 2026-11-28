import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from .config import DIRECTION, TEST_SCENARIO_SEEDS, config_for_k
from .controllers import run_episode
from .env import IntersectionEnv
from .run_experiments import build_controllers
from .train import build_scenario_pairs


LABELS = {
    "FixedTime": "FixedTime",
    "RuleBased": "RuleBased",
    "current": "Current Q-learning",
    "history": "History Q-learning",
}
CONDITIONS = ("normal", "disrupted")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_counts(k):
    cfg = config_for_k(k)
    scenarios = build_scenario_pairs(cfg, TEST_SCENARIO_SEEDS)
    rows = []

    # Keep the evaluator and scenario order consistent with run_experiments.py.
    for controller_name, agent_seed, controller in build_controllers(cfg):
        for condition in CONDITIONS:
            for scenario in scenarios[condition]:
                result = run_episode(IntersectionEnv(cfg), controller, scenario)
                logs = result["logs"]

                row = {
                    "k": k,
                    "controller": controller_name,
                    "agent_seed": agent_seed,
                    "scenario_seed": scenario.seed,
                    "condition": condition,
                    "post_steps": len(logs),
                }
                for index, direction in enumerate(DIRECTION):
                    row[f"hits_{direction}"] = sum(
                        step["queue_real"][index] >= k for step in logs
                    )
                rows.append(row)

    return rows


def summarize(rows):
    # First pool the 20 scenario runs within each policy and condition.
    groups = defaultdict(list)
    for row in rows:
        key = (row["controller"], row["agent_seed"], row["condition"])
        groups[key].append(row)

    seed_rates = defaultdict(list)
    for (controller, _, condition), episodes in groups.items():
        steps = sum(row["post_steps"] for row in episodes)
        rates = {
            direction: 100 * sum(row[f"hits_{direction}"] for row in episodes) / steps
            for direction in DIRECTION
        }
        rates["overall"] = mean(rates.values())
        seed_rates[controller, condition].append(rates)

    # Each trained policy receives equal weight in its controller's summary.
    summary = []
    for controller in LABELS:
        for condition in CONDITIONS:
            rates = seed_rates[controller, condition]
            row = {
                "k": rows[0]["k"],
                "controller": controller,
                "condition": condition,
                "policy_count": len(rates),
            }
            for key in (*DIRECTION, "overall"):
                row[f"pct_{key}"] = mean(rate[key] for rate in rates)
            summary.append(row)

    return summary


def plot_summary(summary, k, path):
    overall = {
        (row["controller"], row["condition"]): row["pct_overall"]
        for row in summary
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.36

    for condition, offset in (("normal", -width / 2), ("disrupted", width / 2)):
        positions = [i + offset for i in range(len(LABELS))]
        heights = [overall[name, condition] for name in LABELS]
        bars = ax.bar(positions, heights, width, label=condition.capitalize())
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)

    ax.set_xticks(range(len(LABELS)), list(LABELS.values()))
    ax.set_ylabel("Saturated post-step approach observations (%)")
    ax.set_title(f"Held-out detector saturation at K = {k}")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(k):
    output_dir = Path("results/experiments") / f"k{k}" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_counts(k)
    summary = summarize(rows)
    write_csv(output_dir / "saturation_episodes.csv", rows)
    write_csv(output_dir / "saturation_summary.csv", summary)
    plot_summary(summary, k, output_dir / "saturation.png")

    print(f"K={k}: saved {len(rows)} episode rows and {len(summary)} summary rows.")
    for row in summary:
        print(f"{row['controller']:10} {row['condition']:10} "
              f"{row['pct_overall']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, choices=(3, 7, 14), default=7)
    main(parser.parse_args().k)