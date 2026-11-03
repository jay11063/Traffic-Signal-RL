from .config import Config, N, S
from .scenario import make_scenario, with_disruption
from .env import IntersectionEnv
from .controllers import FixedTime, RuleBased, run_episode

cfg = Config()

normal = make_scenario(
    seed=100,
    cfg=cfg,
    peak_dirs=(N, S),
    base=0.15,
    peak=0.65
)

disrupted = with_disruption(
    scenario=normal,
    direction=N,
    start=600,
    duration=300,
    alpha=0.5
)

env = IntersectionEnv(cfg)

fixed_result = run_episode(
    env,
    FixedTime(green_times=(45, 10)),  # Normal-traffic validation seeds 100-109.
    normal
)

rule_result = run_episode(
    env,
    RuleBased(threshold=5),  # Best of thresholds 1-5 on validation seeds 100-109.
    normal
)

print(fixed_result["metrics"])
print(rule_result["metrics"])
