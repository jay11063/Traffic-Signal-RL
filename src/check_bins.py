from collections import Counter, defaultdict

import numpy as np
import csv
from pathlib import Path

from .config import Config, DIRECTION, N, S, E, W, VALIDATION_SCENARIO_SEEDS
from .controllers import FixedTime, RuleBased
from .env import IntersectionEnv
from .scenario import make_scenario, sample_disruption


def bin_index(value, edges):
    low, high = edges
    if value <= low:
        return 0
    if value <= high:
        return 1
    return 2


def collect_observations(cfg, controller, scenario):
    env = IntersectionEnv(cfg)
    obs = env.reset(scenario)
    controller.reset()

    done = False
    while not done:
        yield obs
        action = controller.act(obs)
        obs, _, done, _ = env.step(action)


def main():
    cfg = Config()
    controllers = (FixedTime((45, 10)), RuleBased(threshold=5))
    candidates = {
        "queue": [(1, 4), (2, 5), (2, 6)],
        "departures": [(0, 2), (0, 4), (0, 6)],
        "green": [(9, 19), (9, 24), (9, 29)],
    }
    histograms = defaultdict(Counter)

    for seed in VALIDATION_SCENARIO_SEEDS:
        normal = make_scenario(seed, cfg, base=0.15, peak=0.65)
        disrupted = sample_disruption(normal, np.random.default_rng([seed, 1]))

        for condition, scenario in (("normal", normal), ("disrupted", disrupted)):
            for controller in controllers:
                group = (condition, controller.name)

                for obs in collect_observations(cfg, controller, scenario):
                    for direction, value in zip(DIRECTION, obs["queue_observation"]):
                        histograms[group, "queue", direction][value] += 1

                    departures = obs["throughput"]
                    histograms[group, "departures", "NS"][departures[N] + departures[S]] += 1
                    histograms[group, "departures", "EW"][departures[E] + departures[W]] += 1

                    phase = ("NS", "EW")[obs["phase"]]
                    histograms[group, "green", phase][obs["phase_time"]] += 1

    output_path = Path("validation") / "bin_counts.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "condition", "controller", "feature", "label",
            "low_edge", "high_edge",
            "bin_0_count", "bin_1_count", "bin_2_count", "total",
        ])

        for (group, feature, label), histogram in histograms.items():
            condition, controller = group
            print(f"\n{condition} | {controller} | {feature} {label}")

            for edges in candidates[feature]:
                counts = [0, 0, 0]
                for value, frequency in histogram.items():
                    counts[bin_index(value, edges)] += frequency

                low, high = edges
                writer.writerow([
                    condition, controller, feature, label,
                    low, high, counts[0], counts[1], counts[2], sum(counts),
                ])
                print(f"  {edges} -> {counts}")

    print(f"\nSaved bin counts to {output_path}")


if __name__ == "__main__":
    main()