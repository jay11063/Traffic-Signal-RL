import numpy as np
import csv
from pathlib import Path
from statistics import mean
import argparse

from .config import config_for_k, TRAIN_SCENARIO_SEEDS, VALIDATION_SCENARIO_SEEDS
from .controllers import run_episode
from .env import IntersectionEnv
from .q_learning import QLearning, HistoryQLearning, save_q_table, train_episode
from .scenario import make_scenario, sample_disruption

EPSILON_START = 0.20
EPSILON_END = 0.02


def build_scenario_pairs(cfg, seeds):
    scenarios = {"normal": [], "disrupted": []}

    for seed in seeds:
        normal = make_scenario(seed, cfg, base=0.15, peak=0.65)
        disrupted = sample_disruption(
            normal, np.random.default_rng([seed, 1])
        )
        assert normal.arrivals is disrupted.arrivals, f"Arrival mismatch at seed {seed}"
        scenarios["normal"].append(normal)
        scenarios["disrupted"].append(disrupted)

    return scenarios


def build_validation_scenarios(cfg):
    return build_scenario_pairs(cfg, VALIDATION_SCENARIO_SEEDS)


def evaluate_checkpoint(cfg, agent, scenarios, episode):
    if isinstance(agent, HistoryQLearning):
        evaluator = HistoryQLearning(cfg, epsilon=0.0, seed=0)
    else:
        evaluator = QLearning(epsilon=0.0, seed=0, queue_edges=cfg.queue_edges())

    evaluator.q_table = agent.q_table

    rows = []
    for condition, condition_scenarios in scenarios.items():
        waits = []
        for scenario in condition_scenarios:
            result = run_episode(IntersectionEnv(cfg), evaluator, scenario)
            waits.append(result["metrics"]["mean_wait_completed"])

        rows.append((episode, condition, mean(waits)))

    return rows


def main(
    episode_limit=None,
    validation_every=100,
    curve_path="results/learning_curve_seed_0.csv",
    agent_seed=0,
    learning_rate=0.10,
    use_history=False,
    k=7,
):
    cfg = config_for_k(k)
    env = IntersectionEnv(cfg)
    agent_options = dict(
        learning_rate=learning_rate,
        gamma=0.95,
        epsilon=EPSILON_START,
        seed=agent_seed,
    )
    if use_history:
        agent = HistoryQLearning(cfg, **agent_options)
    else:
        agent = QLearning(queue_edges=cfg.queue_edges(), **agent_options)

    all_seeds = TRAIN_SCENARIO_SEEDS
    seeds = all_seeds if episode_limit is None else all_seeds[:episode_limit]

    validation_scenarios = build_validation_scenarios(cfg)
    curve_rows = evaluate_checkpoint(cfg, agent, validation_scenarios, episode=0)

    for index, seed in enumerate(seeds):
        # Episode index 0 starts at 0.20. Index 1999 ends at 0.02.
        progress = index / (len(all_seeds) - 1)
        agent.epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * progress

        scenario = make_scenario(seed, cfg, base=0.15, peak=0.65)
        condition = "normal"

        if index % 2 == 1:
            disruption_rng = np.random.default_rng([seed, 1])
            scenario = sample_disruption(scenario, disruption_rng)
            condition = "disrupted"

        metrics = train_episode(env, agent, scenario)
        episode = index + 1
        if episode % validation_every == 0 or episode == len(seeds):
            curve_rows.extend(evaluate_checkpoint(cfg, agent, validation_scenarios, episode))

        if episode_limit is not None or index == 0 or (index + 1) % 100 == 0:
            print(
                f"episode={index + 1} seed={seed} condition={condition} "
                f"epsilon={agent.epsilon:.5f} "
                f"states={len(agent.q_table)} "
                f"mean_wait={metrics['mean_wait_completed']:.2f}"
            )

    curve_path = Path(curve_path)
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    with curve_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "condition", "mean_wait_s"])
        writer.writerows(curve_rows)

    print(f"Saved validation curve to {curve_path}")
    return agent


def policy_folder(k, use_history):
    if k == 7:
        return Path("results/history") if use_history else Path("results")

    variant = "history" if use_history else "current"
    return Path("results") / f"k{k}" / variant


def run_training_seeds(learning_rate=0.10, output_dir=None, use_history=False, k=7):
    output_dir = Path(output_dir) if output_dir is not None else policy_folder(k, use_history)

    for agent_seed in (0, 1, 2):
        print(f"\nTraining agent seed {agent_seed}, learning rate {learning_rate}")

        agent = main(
            agent_seed=agent_seed,
            learning_rate=learning_rate,
            curve_path=output_dir / f"learning_curve_seed_{agent_seed}.csv",
            use_history=use_history,
            k=k,
        )

        table_path = output_dir / f"q_table_seed_{agent_seed}.json"
        save_q_table(agent.q_table, table_path)
        print(f"Saved {len(agent.q_table)} states to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, choices=(3, 7, 14), default=7)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()

    run_training_seeds(k=args.k, use_history=args.history)