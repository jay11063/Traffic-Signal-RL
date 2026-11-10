
import numpy as np
from .config import N, S, E, W
from dataclasses import dataclass

@dataclass
class Scenario:
    seed: int
    arrivals: list[list[int]] # arrivals[time(sec)][dir]
    disruptions: list # (direction, start, end, alpha)

    def  alpha_at(self, t):
        alpha = [1.0 for _ in range(4)]

        for direction, start, end, factor in self.disruptions:
            if start <= t < end:
                alpha[direction] = min(alpha[direction], factor)

        return alpha


def rush_profile(t, cfg, peak_dirs, base, peak):
    if t<0 or t>=cfg.inflow_duration:
        return [0.0 for _ in range(4)]

    progress = t / cfg.inflow_duration

    if progress < (1/3): 
        level = progress * 3
    elif progress < (2/3):
        level = 1.0
    else:
        level = (1-progress)*3

    rates = [base for _ in range(4)]

    for direction in peak_dirs:
        rates[direction] = base + (peak - base) * level

    return rates


def make_scenario(seed, cfg, base, peak, peak_dirs=(N,S)):
    rng = np.random.default_rng(seed)
    arrivals = []

    for t in range(cfg.inflow_duration):
        rates = rush_profile(t,cfg, peak_dirs, base, peak)
        counts = rng.poisson(rates)
        arrivals.append([int(count) for count in counts])

    return Scenario(seed=seed, arrivals=arrivals, disruptions=[])

def with_disruption(scenario:Scenario, direction:int, start:int, duration:int, alpha:float) -> Scenario:
    new_disruptions = scenario.disruptions.copy()
    new_disruptions.append((direction, start, start+duration, alpha))

    return Scenario(seed=scenario.seed, arrivals=scenario.arrivals, disruptions=new_disruptions)


def sample_disruption(scenario: Scenario, rng: np.random.Generator) -> Scenario:
    """return a scenario with one additional sampled disruption."""
    direction = int(rng.choice((N,S,E,W)))
    start = int(rng.integers(450,901))
    duration = int(rng.integers(150,451))
    alpha = float(rng.uniform(0.4, 0.6))

    return with_disruption(scenario, direction, start, duration, alpha)
