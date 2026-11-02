# Traffic-Signal-RL

Traffic signal control at a single four-way intersection with limited queue
observations and temporary reductions in traffic capacity.

**COMP 3106 - Introduction to Artificial Intelligence**

Carleton University, Fall 2026

Individual project by Jaeyoon Lee

## Current status

Implemented:

- A one-second FIFO queue simulator with two signal phases.
- Reproducible Poisson arrival scenarios and temporary capacity disruptions.
- Fixed-time and rule-based controllers.
- Waiting-time, queue-length, throughput, and unfinished-vehicle metrics.

Reinforcement learning controllers, observation-history features, baseline tuning,
and graphical visualization are planned work. The current implementation is a
numerical simulator.

## Setup

The project has been tested with Python 3.10.11 and NumPy 1.25.2.
From the repository root, install the dependency:

```powershell
python -m pip install -r requirements.txt
```

## Run a comparison

From the repository root, start Python with `python`, then run the following
example. It compares both implemented controllers on normal and disrupted traffic,
using identical arrival records for all four runs.

```python
from src.config import Config, N, S
from src.scenario import make_scenario, with_disruption
from src.env import IntersectionEnv
from src.controllers import FixedTime, RuleBased, run_episode

cfg = Config()

normal = make_scenario(
    seed=42,
    cfg=cfg,
    base=0.15,
    peak=0.65,
    peak_dirs=(N, S),
)
disrupted = with_disruption(
    normal, direction=N, start=600, duration=300, alpha=0.5
)

env = IntersectionEnv(cfg)

for condition, scenario in (("normal", normal), ("disrupted", disrupted)):
    for controller in (FixedTime((30, 30)), RuleBased(threshold=2)):
        result = run_episode(env, controller, scenario)
        print(condition, controller.name)
        print(result["metrics"])
```

`run_episode()` resets the environment and controller for each run. It returns:

- `result["metrics"]`: the final episode statistics.
- `result["logs"]`: one evaluator log per decision interval, including the
  requested action and reward.

The fixed green durations and rule threshold above are example settings, not
validation-tuned parameters. A single seed does not establish which controller
performs better in general.

The existing `src/test.py` is an earlier demonstration script that compares both
controllers on normal traffic only. Its top-level imports are not yet consistent
with the package-relative imports in the simulator modules. Use the package API
example above until those imports are updated. There is currently no
`src/__main__.py` entry point or command-line JSON export.

## Check simulation results

Use small, manually specified arrival scenarios to inspect the simulation logs
and compare results with calculations by hand. In particular, check that:

- Total arrivals equal total departures plus the vehicles still queued.
- Vehicles do not depart on red or during clearance.
- Minimum and maximum green constraints are respected.
- Waiting times match known arrival and departure times.
- Controller observations contain only the intended information and queues are
  clipped to the detector cap.
- Repeating a scenario with the same controller produces the same results.

These are checks to perform during development, not a claim that an automated
test suite is currently implemented. Separately, controller performance will be
evaluated across multiple seeds and traffic conditions using the metrics below.

## Simulation model

The direction order is **N, S, E, W**. The signal has two phases:

- `PHASE_NS = 0`: north and south may discharge.
- `PHASE_EW = 1`: east and west may discharge.

Each approach has one aggregate FIFO queue representing its lanes. The queue
stores each vehicle's arrival time. Every episode starts with empty queues and
NS green.

Each one-second tick:

1. Applies any forced switch required by the maximum green time.
2. Adds the scheduled arrivals.
3. Records the maximum queue lengths before discharge.
4. Discharges vehicles from green approaches, unless clearance is active.
5. Adds one second of waiting for every remaining vehicle.

A vehicle arriving and departing in the same tick has zero waiting time.
Fractional discharge capacity accumulates while an approach is green and has
waiting vehicles. Unused capacity is discarded when the queue empties or the
phase switches.

During clearance, all approaches stop and the green timer does not advance.
The phase value identifies the upcoming green phase. Maximum green is enforced
every second, including between controller decisions.

### Current configuration

Edit the defaults in `src/config.py` to change the environment settings.

| Setting | Default | Meaning |
| --- | --- | --- |
| Simulation resolution | 1 s | One internal tick |
| `decision_interval` | 5 s | Time between controller decisions |
| `min_green` | 10 s | Minimum green before a requested switch |
| `max_green` | 60 s | Maximum green before a forced switch |
| `clearance_time` | 3 s | All-red interval after switching |
| `inflow_duration` | 1,800 s | Duration of new vehicle arrivals |
| `drain_cap` | 900 s | Maximum additional time to clear remaining vehicles |
| `lanes_per_approach` | 2 | Lanes represented by each approach queue |
| `saturation_flow_per_lane` | 1.0 vehicle/s | Service capacity per lane during green |
| `vehicle_occupancy_m` | 7 m | Effective length per stopped vehicle |
| `detection_range_m` | 50 m | Nominal detector coverage |
| `reward_scale` | 200 | Divisor used to scale the waiting penalty |
| `history_length` | 6 | Reserved for a future history-based controller |

The current derived values are:

- `cfg.max_observable()`: floor(50 / 7) = **7 vehicles per approach**.
- `cfg.nominal_capacity()`: 2 x 1.0 = **2 vehicles/s per green approach**.

These are methods and are called with parentheses. The detector cap is an
aggregate observation abstraction; it does not model separate visibility for
each physical lane.

### Arrivals and disruptions

`make_scenario(seed, cfg, base, peak, peak_dirs)` generates the entire arrival
table before an episode. Each row contains the four approach arrival counts for
one second. Selected peak directions follow a trapezoidal arrival-rate profile:
increase during the first third, remain at the peak during the middle third,
and decrease during the final third. Other directions remain at the base rate.

`with_disruption()` creates a new scenario with the same arrival table and an
additional `(direction, start, end, alpha)` event. Treat the shared arrival table
as read-only.

Actual green capacity is `nominal_capacity() * alpha`. With the current defaults,
`alpha=0.5` gives 1 vehicle/s on the affected approach. Disruption intervals include
the start and exclude the end. Overlapping events on the same approach use the
lowest active alpha. The controller is not told the disruption schedule or alpha.

## Controller interface

A controller implements `act(obs) -> action` and may override `reset()` to clear
episode-specific memory.

- `ACTION_HOLD = 0`: keep the current phase.
- `ACTION_SWITCH = 1`: request a switch to the other phase.

The environment enforces minimum green, maximum green, and clearance constraints.
Controllers receive only the following observation:

| Key | Meaning |
| --- | --- |
| `queue_observation` | Four queue counts, each clipped to K |
| `throughput` | Departures by direction during the previous step |
| `phase` | NS or EW phase |
| `phase_time` | Elapsed green time, excluding clearance |
| `act_mask` | Boolean list: [HOLD allowed, SWITCH allowed] |

`reset(scenario)` returns the initial observation.
`step(action)` returns `(observation, reward, done, info)`.

Reward is:

```python
reward = -wait_increment / cfg.reward_scale
```

`wait_increment` is measured in vehicle-seconds over that step.
True queue lengths, disruption factors, and cumulative statistics are returned
in evaluator-only `info`; `run_episode()` does not pass them to the controller.
The alpha in `info` describes its reported time boundary, not an average over
the preceding step.

### Implemented controllers

**FixedTime** uses separate NS/EW target green durations. It requests a switch
at the first permitted decision when the target is reached. Actual green
durations can exceed the target because decisions occur at fixed intervals;
the environment still enforces maximum green every second.

**RuleBased** compares the observed queue sums of the two axes and switches
when the opposing axis is larger by at least the threshold. If all four counts
are clipped to K, both axis sums are 2K and the rule cannot distinguish the true
queue imbalance.

## Evaluation metrics

An episode ends when arrivals have finished and all queues are empty, or when
`inflow_duration + drain_cap` is reached. Call `metrics()` after the episode ends.

| Key | Meaning |
| --- | --- |
| `mean_wait_completed` | Mean wait of vehicles that have departed, in seconds |
| `p95_wait_completed` | 95th percentile of completed waits, in seconds |
| `max_wait_completed` | Longest completed wait, in seconds |
| `total_waiting` | Accumulated waiting across all vehicles, in vehicle-seconds |
| `max_queue` | Maximum queue length for each direction: N, S, E, W |
| `total_arrived` | Total number of arrivals |
| `total_passed` | Total number of departures |
| `remaining` | Number of vehicles still queued at termination |
| `unfinished_ratio` | Remaining vehicles divided by total arrivals; zero if no arrivals |
| `unfinished_wait_so_far` | Accumulated wait of the remaining vehicles |

Completed waiting statistics are `None` if no vehicle has departed. If the drain
cap leaves unfinished vehicles, report their fraction alongside completed-only
waiting statistics. The four maximum queue lengths may occur at different times.

## Project files

| File | Purpose |
| --- | --- |
| `src/config.py` | Environment settings, direction/phase constants, and action constants |
| `src/scenario.py` | Pre-generated arrivals and capacity disruption scenarios |
| `src/env.py` | Queue dynamics, signal constraints, observations, rewards, and metrics |
| `src/controllers.py` | Controller interface, two baselines, and episode runner |
| `src/test.py` | Earlier demonstration script; see the import note above |
| `requirements.txt` | Pinned NumPy dependency |

## Scope and limitations

The model does not simulate turns, pedestrians, travel time, lane changes,
finite road storage, or camera image processing. Its parameters are experimental
settings, not a calibrated real-world intersection. Learning and history-based
control remain future work. Evaluation will use separate training, validation,
and test scenarios with shared arrival records for fair comparisons.
