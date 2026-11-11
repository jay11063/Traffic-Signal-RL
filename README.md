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

Normal-traffic fixed-time and rule-threshold tuning have been completed on
validation seeds 100-109, along with normal/disruption saturation measurements.
Reinforcement learning controllers, observation-history features,
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
    for controller in (FixedTime((45, 10)), RuleBased(threshold=5)):
        result = run_episode(env, controller, scenario)
        print(condition, controller.name)
        print(result["metrics"])
```

`run_episode()` resets the environment and controller for each run. It returns:

- `result["metrics"]`: the final episode statistics.
- `result["logs"]`: one evaluator log per decision interval, including the
  requested action and reward.

Both baseline parameters above were selected on normal-traffic validation seeds
100-109 and are frozen for the normal/disruption comparison. A single
development seed does not establish which controller performs better in general.

The existing `src/test.py` compares both controllers on normal traffic only. Run
it from the repository root with `python -m src.test`; its relative imports need
package execution rather than `python src/test.py`. There is currently no
`src/__main__.py` entry point or command-line JSON export.

### Scenario seeds

| Split | Scenario seeds (inclusive) | Count | Purpose |
| --- | --- | ---: | --- |
| Training | 1000–2999 | 2,000 | Train the Q-learning agents |
| Validation | 100–109 | 10 | Tune baselines and select model settings |
| Test | 3000–3019 | 20 | Evaluate all controllers after settings are fixed |

Vehicle arrivals are generated with `np.random.default_rng(seed)`.
Disruption parameters are sampled with `np.random.default_rng([seed,1])`,
where `1` is fixed tag for the disruption generator. The two generators have seperate states, so sampling a disruption does not advance the arrival generator. Each normal/disrupted pair shares the same arrival list.
Treat this list as read-only.

Each test seed produces a normal scenario and a paired disrupted scenario
with the same vehicle arrivals. All controllers use the same 20 scenario
pairs. Test scenarios are reserved for final evaluation.

These seeds control scenario generation. Q-learning exploration uses separate
random seeds.

### Baseline selection

The results and frozen settings below apply only to the reference demand profile
(base 0.15, NS peak 0.65), representing a recurring weekday rush hour. This rate
curve stays fixed across episodes; Poisson arrival realizations differ. Before
Q-learning training, the proposal plans independent uniform sampling of disruption
direction N/S/E/W, integer start time 450-900 s, integer duration 150-450 s, and
remaining-capacity multiplier alpha 0.4-0.6. Both normal-tuned baselines stay fixed for normal/disrupted
tests. Existing JSON disruption results use N, alpha 0.5, and [600, 900) s.
Each condition will have at least 20 shared held-out scenarios. Normal scenarios
differ only in arrivals; disrupted scenarios additionally vary disruption
parameters. Paired conditions share arrivals. Results are limited to the known,
recurring demand pattern; other demand profiles are not tested.

The search used normal scenarios with `base=0.15`, `peak=0.65`, NS peak demand,
and validation seeds 100-109. It evaluated all 28 combinations of NS targets
30-60 s and EW targets 10-25 s, both in 5 s increments. The objective was the
unweighted mean of the ten episode mean completed waits. Every candidate run
finished with zero remaining vehicles, so completed-only censoring did not affect
selection. The selected targets were **(45, 10) s**, with validation mean wait
**13.75 s**. These are the best targets in this grid, not a global optimum.
The (40, 10) alternative has mean wait 13.90 s and mean episode p95 41.24 s,
versus 13.75 s and 47.17 s for (45, 10). We retain (45, 10) because mean wait is
the selection objective; p95 is a secondary reported metric. This is not a tie,
and no p95 tie-break was applied.

Rule thresholds 1-5 used the same ten normal validation scenarios and objective:

| Threshold | Mean episode wait (s) | Mean episode p95 wait (s) |
| --- | ---: | ---: |
| 1 | 16.20 | 40.30 |
| 2 | 15.63 | 39.60 |
| 3 | 13.85 | 34.36 |
| 4 | 13.44 | 33.04 |
| **5 (selected)** | **12.84** | **31.20** |

All 50 runs drained completely. Exact objective ties would use the lower
threshold; no tie occurred. Threshold 5 is best within the requested range,
not a claimed optimum over all possible thresholds. Both baseline settings are
frozen for normal/disruption comparisons; no incident-specific tuning is used.

With 5 s decisions and 3 s clearance, the first NS green lasts 45 s; subsequent
NS/EW greens last 47/12 s, giving a steady 65 s cycle. Targets are not exact
executed green durations. Freeze this normal-tuned pair for the planned paired
normal/disruption evaluation, without retuning for the disruption schedule.

Seed 42 is a previously inspected development example, not a held-out test:

| Controller | Mean wait (s) | p95 wait (s) | Maximum queues (N, S, E, W) |
| --- | ---: | ---: | --- |
| Untuned fixed (30, 30) | 145.45 | 368.3 | 182, 152, 14, 13 |
| Tuned fixed (45, 10) | 14.54 | 60.0 | 17, 17, 24, 16 |
| Rule, threshold 2 | 11.98 | 25.0 | 25, 19, 10, 8 |
| Tuned rule, threshold 5 | 10.94 | 25.0 | 21, 19, 9, 10 |

All four passed all 2,235 arrivals. The tuned rule reduces mean wait by about
24.8% relative to the tuned fixed controller on this development example. Its
N/S maximum queues are still larger. On the ten validation scenarios, tuned
fixed-time averages 13.75 s and the tuned rule 12.84 s. Final conclusions require
held-out multi-seed comparisons. The earlier threshold-2 result is retained for
comparison. Full grid metrics, reward summaries, and saturation counts are
recorded in `output/validation/baseline_validation.json`.

### Observation saturation

Using fixed targets (45, 10), K=7, and paired validation seeds 100-109:

| Condition | N | S | E | W | All step-direction pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 25.64% | 23.53% | 24.38% | 29.10% | 25.66% |
| N half capacity, [600, 900) s | 56.30% | 23.53% | 24.38% | 29.10% | 33.33% |

Saturation means the returned post-step observation equals K (true queue >= K,
including equality). Counts are pooled across seeds, with 3,650 returned steps
per condition: 3,650 samples per direction and 14,600 step-direction pairs.
The denominator includes clearance, drain, and the terminal step; reset is
excluded. A shortened final step still counts as one observation. This measures
the whole episode, not just the incident window. All runs drained. Because the
fixed schedule is unchanged, the other directions have identical observations
in paired normal/disrupted runs. Saturation supports the partial-observation
premise but does not establish that history improves control.

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

Development checks on November 2, 2026 passed vehicle conservation and equality
between accumulated step waiting and completed plus unfinished vehicle waits,
including nominal approach capacities 1.0, 0.5, and 0.25 vehicle/s with a temporary
half-capacity disruption. Checks also passed for exact observation keys, clipping
above K, repeated-run reproducibility, and a hand-calculated fractional-service
example. One-second checks verified minimum/maximum green, no red departures,
and no discharge during clearance. These were ad hoc checks, not a maintained
automated test suite. Controller performance evaluation remains separate.

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
| `saturation_flow_per_lane` | 0.5 vehicle/s | Service capacity per lane during green |
| `vehicle_occupancy_m` | 7 m | Effective length per stopped vehicle |
| `detection_range_m` | 50 m | Nominal detector coverage |
| `reward_scale` | 200 | Divisor used to scale the waiting penalty |
| `history_length` | 6 | Reserved for a future history-based controller |

The current derived values are:

- `cfg.max_observable()`: floor(50 / 7) = **7 vehicles per approach**.
- `cfg.nominal_capacity()`: 2 x 0.5 = **1 vehicle/s per green approach**.

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
`alpha=0.5` gives 0.5 vehicle/s on the affected approach. Disruption intervals include
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
Across normal validation seeds 100-109, pooled step increments had mean/p95
86.57/174.55 for tuned fixed-time and 80.10/179.00 for the tuned rule (threshold 5).
Dividing by 200 gives mean reward about -0.43 and -0.40, respectively, so the existing scale is
retained provisionally. Disruption runs still need scale validation before
learning. Both learning variants must use the same frozen scale. A common
divisor changes reward magnitude, not the ratio of two policies' penalties.
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
| `src/test.py` | Normal-traffic demonstration; run with `python -m src.test` |
| `requirements.txt` | Pinned NumPy dependency |

## Scope and limitations

The model does not simulate turns, pedestrians, travel time, lane changes,
finite road storage, or camera image processing. Its parameters are experimental
settings, not a calibrated real-world intersection. Learning and history-based
control remain future work. Evaluation will use separate training, validation,
and test scenarios with shared arrival records for fair comparisons.

Queues have unlimited storage, so long queues do not cause upstream spillback
or prevent new arrivals. Queue counts aggregate two lanes: 182 vehicles at 7 m
per vehicle would occupy about 637 m if evenly divided across lanes, not 1.3 km
of road. The simulator does not model lane allocation or physical queue length.
The 0.5 vehicle/s/lane assumption corresponds to the 1,800 vehicles/hour/lane
approximation in the [FHWA Traffic Signal Timing Manual, Chapter 3](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter3.htm).
