import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .config import Config, DIRECTION, TEST_SCENARIO_SEEDS, VALIDATION_SCENARIO_SEEDS
from .controllers import FixedTime, RuleBased, run_episode
from .env import IntersectionEnv
from .q_learning import HistoryQLearning, QLearning, load_q_table
from .train import build_scenario_pairs


def build_controllers(cfg):
    yield "FixedTime", None, FixedTime((45, 10))
    yield "RuleBased", None, RuleBased(threshold=5)

    for variant in ("current", "history"):
        for training_seed in (0, 1, 2):
            if variant == "history":
                agent = HistoryQLearning(cfg, epsilon=0.0, seed=0)
                folder = Path("results/history")
                state_length = 10
            else:
                agent = QLearning(epsilon=0.0, seed=0)
                folder = Path("results")
                state_length = 8

            path = folder / f"q_table_seed_{training_seed}.json"
            agent.q_table = load_q_table(path)
            if not agent.q_table or any(len(s) != state_length for s in agent.q_table):
                raise ValueError(f"Unexpected Q-table format in {path}")

            yield variant, training_seed, agent


def main(split):
    cfg = Config()
    seeds = VALIDATION_SCENARIO_SEEDS if split == "validation" else TEST_SCENARIO_SEEDS
    scenarios = build_scenario_pairs(cfg, seeds)
    rows = []

    for name, training_seed, controller in build_controllers(cfg):
        for condition in ("normal", "disrupted"):
            for scenario in scenarios[condition]:
                result = run_episode(IntersectionEnv(cfg), controller, scenario)
                metrics = result["metrics"]

                row = {
                    "controller": name,
                    "agent_seed": training_seed,
                    "scenario_seed": scenario.seed,
                    "condition": condition,
                    "mean_wait_completed": metrics["mean_wait_completed"],
                    "p95_wait_completed": metrics["p95_wait_completed"],
                    "total_arrived": metrics["total_arrived"],
                    "total_passed": metrics["total_passed"],
                    "remaining": metrics["remaining"],
                    "unfinished_ratio": metrics["unfinished_ratio"],
                }
                for direction, count in zip(DIRECTION, metrics["max_queue"]):
                    row[f"max_queue_{direction}"] = count

                assert row["total_arrived"] == row["total_passed"] + row["remaining"]
                rows.append(row)

    output_dir = Path("results/experiments") / split
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "episodes.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, allow_nan=False)

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} episode results to {output_dir}")
    print(f"Runs with unfinished vehicles: {sum(r['remaining'] > 0 for r in rows)}")

    # Print one line per policy so the validation output can be checked against the README.
    for name, training_seed in dict.fromkeys((r["controller"], r["agent_seed"]) for r in rows):
        values = []
        for condition in ("normal", "disrupted"):
            waits = [
                r["mean_wait_completed"] for r in rows
                if r["controller"] == name and r["agent_seed"] == training_seed
                and r["condition"] == condition
            ]
            # An episode with no departures has no completed mean.
            value = "n/a" if None in waits else f"{mean(waits):.2f}"
            values.append(value)
        print(f"{name:10} seed={str(training_seed):4}  normal={values[0]}  disrupted={values[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("validation", "test"))
    main(parser.parse_args().split)