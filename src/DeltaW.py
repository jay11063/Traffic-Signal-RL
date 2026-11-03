import numpy as np

from src.config import Config
from src.scenario import make_scenario
from src.env import IntersectionEnv
from src.controllers import FixedTime, RuleBased, run_episode

cfg = Config()

controllers = {
    "FixedTime (45, 10)": FixedTime((45, 10)),
    "RuleBased (threshold=5)": RuleBased(threshold=5),
}

# Store each controller's waiting increments separately.
delta_w = {name: [] for name in controllers}

for seed in range(100, 110):
    # Both controllers receive the same arrivals.
    scenario = make_scenario(
        seed=seed,
        cfg=cfg,
        base=0.15,
        peak=0.65,
    )

    for name, controller in controllers.items():
        result = run_episode(
            IntersectionEnv(cfg),
            controller,
            scenario,
        )

        for log in result["logs"]:
            delta_w[name].append(log["wait_increment"])

# Calculate statistics across all steps from all ten seeds.
for name, values in delta_w.items():
    mean = np.mean(values)
    p95 = np.percentile(values, 95)

    print(f"\n{name}")
    print(f"Number of steps: {len(values)}")
    print(f"Total waiting: {sum(values)}")
    print(f"Mean: {mean}")
    print(f"p95: {p95}")
    print(f"Mean / p95: {mean:.1f} / {p95:.1f}")