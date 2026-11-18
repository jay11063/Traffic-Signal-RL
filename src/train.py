import numpy as np

from .config import Config, TRAIN_SCENARIO_SEEDS
from .env import IntersectionEnv
from .q_learning import QLearning, save_q_table, train_episode
from .scenario import make_scenario, sample_disruption

EPSILON_START = 0.20
EPSILON_END = 0.02


def main(episode_limit=None):
    cfg = Config()
    env = IntersectionEnv(cfg)
    agent = QLearning(
        learning_rate=0.1,
        gamma=0.95,
        epsilon=EPSILON_START,
        seed=0,
    )

    all_seeds = TRAIN_SCENARIO_SEEDS
    seeds = all_seeds if episode_limit is None else all_seeds[:episode_limit]

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

        if episode_limit is not None or index == 0 or (index + 1) % 100 == 0:
            print(
                f"episode={index + 1} seed={seed} condition={condition} "
                f"epsilon={agent.epsilon:.5f} "
                f"states={len(agent.q_table)} "
                f"mean_wait={metrics['mean_wait_completed']:.2f}"
            )

    return agent


if __name__ == "__main__":
    agent = main()
    path = "results/q_table_seed_0.json"
    save_q_table(agent.q_table, path)
    print(f"Saved {len(agent.q_table)} states to {path}")