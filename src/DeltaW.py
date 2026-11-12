import numpy as np

from src.config import Config, VALIDATION_SCENARIO_SEEDS
from src.scenario import make_scenario, sample_disruption
from src.env import IntersectionEnv
from src.controllers import FixedTime, RuleBased, run_episode

cfg = Config()

controllers = {
    "FixedTime (45, 10)": FixedTime((45, 10)),
    "RuleBased (threshold=5)": RuleBased(threshold=5),
}

# Pool waiting increments separately for each condition and controller.
conditions = ("normal", "disrupted")
delta_w = {(condition, name): [] for condition in conditions for name in controllers}

for seed in VALIDATION_SCENARIO_SEEDS:
    normal = make_scenario(
        seed=seed,
        cfg=cfg,
        base=0.15,
        peak=0.65,
    )

    disruption_rng = np.random.default_rng([seed, 1])
    disrupted = sample_disruption(normal, disruption_rng)

    for condition, scenario in zip(conditions, (normal, disrupted)):
        for name, controller in controllers.items():
            result = run_episode(IntersectionEnv(cfg), controller, scenario)
            for log in result["logs"]:
                delta_w[condition, name].append(log["wait_increment"])

# Calculate statistics across all steps from all ten seeds.
for (condition, name), values in delta_w.items():
    mean = np.mean(values)
    p95 = np.percentile(values, 95)

    print(f"\n{condition.capitalize()} | {name}")
    print(f"Number of steps: {len(values)}")
    print(f"Total waiting: {sum(values)}")
    print(f"Mean: {mean}")
    print(f"p95: {p95}")
    print(f"Mean / p95: {mean:.1f} / {p95:.1f}")
