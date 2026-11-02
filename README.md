# Traffic-Signal-RL

A reinforcement learning agent for traffic signal control at a single four-way intersection, where the detector can only observe a limited stretch of road near the stop line and one approach may temporarily lose capacity.

COMP 3106 (Introduction to Artificial Intelligence), Carleton University, Fall 2026. Individual final project.

## Status
The queue simulator, reproducible scenarios, fixed-time baseline and rule-based
baseline are implemented. Learning controllers and validation-based baseline tuning
are future work. This is a numerical simulator, with no graphical interface yet.

## Run

Tested with Python 3.10.11 and NumPy 1.25.2 (pinned in `requirements.txt`).
Run these commands from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m src --seed 42
python -m unittest discover -s tests -v
```

Save the four paired example runs, including configuration, complete arrival tables,
disruption events, metrics, decision logs and runtime versions:

```powershell
python -m src --seed 42 --output results/demo_seed42.json
```

The demo compares FixedTime and RuleBased on identical arrivals, with and without
a north-approach capacity reduction to 0.5 during seconds [600, 900).
Use `python -m src --help` for disruption options. These are demonstration results,
not evidence of general performance; the default fixed-time split is not tuned.

## Model and interface

- Direction order: N, S, E, W. Two phases: NS and EW, starting with NS green.
- One FIFO queue per approach; turns, travel time and finite storage are omitted.
- Each second: arrivals, discharge, then one second of waiting for remaining cars.
- All-red clearance lasts exactly `clearance_time`; green timers exclude it.
- Fractional service accumulates only while an approach is green and nonempty.
  Unused service is discarded when its queue empties or the signal switches.
- Maximum green is enforced every second, including between controller decisions.
- Observation has exactly `queue_observation`, `throughput`, `phase`, `phase_time`,
  and `act_mask`. True queues and disruption factors are evaluator-only information.
- K = floor(detection_range_m / vehicle_occupancy_m) per approach, as a simplified
  detector cap. It does not model visibility separately for each physical lane.
- Mean/p95/maximum completed waits exclude unfinished cars. Always report the
  unfinished ratio and total accumulated waiting alongside those statistics.

The environment returns `(observation, reward, done, info)`; it is not a Gymnasium
subclass. `run_episode()` returns a dictionary containing metrics and logs.

## Files and explanation

| File | Purpose |
| --- | --- |
| `src/config.py` | Immutable settings and shared constants |
| `src/scenario.py` | Poisson arrivals and capacity disruption events |
| `src/env.py` | Queues, signals, observations, reward and metrics |
| `src/controllers.py` | Common controller interface, two baselines, episode loop |
| `src/__main__.py` | Reproducible command-line example and JSON export |
| `tests/test_env.py` | 26 deterministic tests using standard-library unittest |
| [docs/simulator_guide_ko.md](docs/simulator_guide_ko.md) | Detailed Korean explanation of every function |
