from .config import Config, N, S
from .scenario import make_scenario, with_disruption
from .env import IntersectionEnv
from .controllers import FixedTime, RuleBased, run_episode

cfg = Config()

normal = make_scenario(
    seed=42,
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
    FixedTime(green_times=(30, 30)),
    normal
)

rule_result = run_episode(
    env,
    RuleBased(threshold=2),
    normal
)

print(fixed_result["metrics"])
print(rule_result["metrics"])