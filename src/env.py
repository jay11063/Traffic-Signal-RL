"""
Direction = (N,S,E,W)
traffic:= (0: N,S), (1: E,W)

reset(scenario) -> observation
step(action) -> observation, reward, done, info

observation := {
    queue_observation: [min(len(q),K) x direction],
    throughput: (N,S,E,W),
    phase: 0/1 (NS or EW),
    phase_time: int,
    act_mask: action_mask()->[ACTION_HOLD, ACTION_SWITCH]
}

info := {
    queue_real: [len(q) x direction]
    alpha: [float x direction],
    wait_increment: sum(waiting_per_tick),
    total_arrived: int (vehicles),
    total_passed: int (vehicles),
    total_waiting: int (vehicles),
    max_queue: [int x direction]
    t: time
}

act(observation) -> action
action_mask() {
    mask = [True, True]

    if phase_time < min_green:      mask[ACTION_SWITCH] = False
    if clearing > 0:                mask[ACTION_SWITCH] = False
    if phase_time >= max_green:     mask[ACTION_HOLD]   = False
}


"""

from collections import deque
import math
import numpy as np

from .config import *
from .scenario import Scenario

class IntersectionEnv:
    """Controller-independent simulation core with no pygame dependency."""

    def __init__(self, cfg: Config):
        """Store settings; initialize episode state in reset()."""
        self.cfg = cfg

    def reset(self, scenario: Scenario) -> dict:
        """Reset episode state and create FIFO deques containing vehicle arrival times."""
        self.scenario = scenario
        self.t = 0
        self.queues = [deque() for _ in range(4)]
        self.phase = PHASE_NS
        self.phase_time = 0
        self.clearing = 0
        self.service_credit = [0.0, 0.0, 0.0, 0.0]
        self.throughput = [0,0,0,0]
        self.total_arrived = 0
        self.total_passed = 0
        self.total_waiting = 0
        self.wait_increment = 0
        self.max_queue = [0,0,0,0]
        self.wait_times = []
        return self._observe()

    def step(self, action):
        """Advance up to decision_interval seconds and return observation, reward, done, info.

        Apply the requested action once, then stop early if the episode finishes.
        The reward is the negative waiting increment divided by reward_scale.
        """
        # 1. Apply the requested action.
        # 2. Run up to decision_interval one-second ticks.
        # 3. reward = -wait_increment / reward_scale
        self.throughput = [0,0,0,0]
        self.wait_increment = 0
        self._apply_action(action)

        for _ in range(self.cfg.decision_interval):
            if self._is_done():
                break

            self._tick()

        reward = -self.wait_increment / self.cfg.reward_scale

        observation = self._observe()
        done = self._is_done()
        info = self._info()

        return observation, reward, done, info


    def _tick(self) -> None:
        """Advance one second: add arrivals, discharge vehicles, and accumulate waiting."""
        # Enforce maximum green even between controller decisions.
        self._apply_action(ACTION_HOLD)

        # 1. Add the arrivals scheduled for the current second.
        if self.t < self.cfg.inflow_duration:
            arrivals_now = self.scenario.arrivals[self.t]

            for direction, count in enumerate(arrivals_now):
                self.queues[direction].extend([self.t] * count)
                self.total_arrived += count

        # Record the peak queue length immediately after arrivals.
        for direction in range(4):
            self.max_queue[direction] = max(self.max_queue[direction], len(self.queues[direction]))

        # 2. Discharge vehicles.
        if self.clearing > 0:
            self.clearing -= 1
            self.service_credit = [0.0, 0.0, 0.0, 0.0]

        else:
            alpha = self.scenario.alpha_at(self.t)
            green_directions = DIRS_IN_PHASE[self.phase]

            for direction in range(4):
                queue = self.queues[direction]

                if direction not in green_directions or not queue:
                    self.service_credit[direction] = 0.0
                    continue

                capacity = (self.cfg.nominal_capacity() * alpha[direction])

                self.service_credit[direction] += capacity

                available = math.floor(self.service_credit[direction] + 1e-9)

                passed = min(len(queue), available)

                for _ in range(passed):
                    arrival_time = queue.popleft()

                    waiting_time = self.t - arrival_time
                    self.wait_times.append(waiting_time)

                self.service_credit[direction] = max(0.0, self.service_credit[direction] - passed)

                if not queue:
                    self.service_credit[direction] = 0.0

                self.throughput[direction] += passed
                self.total_passed += passed

            self.phase_time += 1

        # 3. Add one second of waiting for every remaining vehicle.
        waiting_now = sum(len(queue) for queue in self.queues)

        self.wait_increment += waiting_now
        self.total_waiting += waiting_now

        self.t += 1


    def _apply_action(self, action: int) -> None:
        """Apply a permitted switch or force one at maximum green; start clearance."""
        if self.clearing > 0:
            return

        force_switch = self.phase_time >= self.cfg.max_green

        requested_switch = (action == ACTION_SWITCH and self.phase_time >= self.cfg.min_green)

        if force_switch or requested_switch:
            self.phase = 1 - self.phase
            self.phase_time = 0
            self.clearing = self.cfg.clearance_time

            self.service_credit = [0.0, 0.0, 0.0, 0.0]

    def action_mask(self):
        """Return [HOLD allowed, SWITCH allowed] under the signal timing constraints."""
        mask = [True, True]

        if self.phase_time < self.cfg.min_green:
            mask[ACTION_SWITCH] = False

        if self.clearing > 0:
            mask[ACTION_SWITCH] = False

        if self.phase_time >= self.cfg.max_green:
            mask[ACTION_HOLD] = False

        return mask

    def _is_done(self) -> bool:
        """Check whether traffic has drained after inflow ends or the drain cap is reached."""
        inflow_finished = self.t >= self.cfg.inflow_duration

        queues_empty = all(len(queue) == 0 for queue in self.queues)

        drain_limit_reached = (self.t >= self.cfg.inflow_duration + self.cfg.drain_cap)

        return (inflow_finished and queues_empty) or drain_limit_reached

    def _observe(self) -> dict:
        """Build the restricted controller observation without true queues or alpha."""
        k = self.cfg.max_observable()

        return {
            "queue_observation": [min(len(queue), k) for queue in self.queues],
            "throughput": self.throughput.copy(),
            "phase": self.phase,
            "phase_time": self.phase_time,
            "act_mask": self.action_mask()
        }

    def _info(self) -> dict:
        """Return evaluator-only statistics; alpha is evaluated at the current time boundary."""
        return {
            "queue_real": [len(queue) for queue in self.queues],
            "alpha": self.scenario.alpha_at(self.t),
            "wait_increment": self.wait_increment,
            "total_arrived": self.total_arrived,
            "total_passed": self.total_passed,
            "total_waiting": self.total_waiting,
            "max_queue": self.max_queue.copy(),
            "t": self.t
        }

    def metrics(self) -> dict:
        """Summarize completed waits, queue peaks, throughput, and unfinished traffic.

        Call after the episode ends. Completed-wait statistics exclude queued vehicles
        and are None when no vehicle has departed. Report unfinished traffic separately.
        """
        remaining = sum(len(queue) for queue in self.queues)

        unfinished_waits = [
            self.t - arrival_time
            for queue in self.queues
            for arrival_time in queue
        ]

        if self.wait_times:
            mean_wait = float(np.mean(self.wait_times))
            p95_wait = float(np.percentile(self.wait_times, 95))
            max_wait = max(self.wait_times)
        else:
            mean_wait = None
            p95_wait = None
            max_wait = None

        if self.total_arrived > 0:
            unfinished_ratio = remaining / self.total_arrived
        else:
            unfinished_ratio = 0.0

        return {
            "mean_wait_completed": mean_wait,
            "p95_wait_completed": p95_wait,
            "max_wait_completed": max_wait,
            "total_waiting": self.total_waiting,
            "max_queue": self.max_queue.copy(),
            "total_arrived": self.total_arrived,
            "total_passed": self.total_passed,
            "remaining": remaining,
            "unfinished_ratio": unfinished_ratio,
            "unfinished_wait_so_far": sum(unfinished_waits)
        }
