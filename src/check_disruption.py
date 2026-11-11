from collections import Counter

import numpy as np

from src.config import Config, DIRECTION
from src.scenario import make_scenario, sample_disruption

cfg  = Config()
seed = 42
normal = make_scenario(seed, cfg, base=0.15, peak=0.65)

# Save the arrival values before sampling a disruption.
arrivals_before = [row.copy() for row in normal.arrivals]

disruption_rng = np.random.default_rng([seed, 1])
disrupted = sample_disruption(normal, disruption_rng)
repeated = sample_disruption(normal, np.random.default_rng([seed, 1]))

# Same seed and a fresh generator give the same disruption.
assert len(disrupted.disruptions) == 1
assert disrupted.disruptions == repeated.disruptions

# Check both object identity and the original arrival values.
assert disrupted.arrivals is normal.arrivals
assert disrupted.arrivals == arrivals_before
assert normal.disruptions == []

# Advance the disruption generator, then regenerate arrivals.
disruption_rng.random(100)
regenerated = make_scenario(seed, cfg, base=0.15, peak=0.65)
assert regenerated.arrivals == arrivals_before

# Check parameter ranges and count sampled directions.
direction_counts = Counter()
sample_seeds = range(10000, 10800)

for sample_seed in sample_seeds:
    rng = np.random.default_rng([sample_seed, 1])
    sampled = sample_disruption(normal, rng)
    assert len(sampled.disruptions) == 1

    direction, start, end, alpha = sampled.disruptions[0]
    duration = end - start

    assert 0 <= direction < len(DIRECTION)
    assert 450 <= start <= 900
    assert 150 <= duration <= 450
    assert 0.4 <= alpha <= 0.6

    direction_counts[direction] += 1

assert normal.disruptions == []
assert normal.arrivals == arrivals_before

for direction, name in enumerate(DIRECTION):
    count = direction_counts[direction]
    print(f"{name}: {count}/{len(sample_seeds)} ({count / len(sample_seeds):.1%})")

print("All assertions passed.")
