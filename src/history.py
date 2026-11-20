from collections import deque

from .config import DIRS_IN_PHASE, PHASE_NS, PHASE_EW
from .env import IntersectionEnv


DECREASING = 0
MAINTAINING = 1
INCREASING = 2
UNAVAILABLE = 3

TREND_THRESHOLD = 2  # vehicles per full 5-second interval


def departure_trend(history, axis):
    counts = [record[axis] for record in history if record[axis] is not None]

    if len(counts) < 2:
        return UNAVAILABLE

    change = counts[-1] - counts[0]
    if change >= TREND_THRESHOLD:
        return INCREASING
    if change <= -TREND_THRESHOLD:
        return DECREASING
    return MAINTAINING


class HistoryIntersectionEnv(IntersectionEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.history = deque(maxlen=cfg.history_length)

    def reset(self, scenario):
        obs = super().reset(scenario)
        self.history.clear()
        self.previous_obs = obs
        return {**obs, "departure_trends": (UNAVAILABLE, UNAVAILABLE)}

    def step(self, action):
        obs, reward, done, info = super().step(action)
        previous = self.previous_obs
        phase = obs["phase"]

        full_green = (phase == previous["phase"] and 
                      obs["phase_time"] - previous["phase_time"] == self.cfg.decision_interval)

        record = [None, None]  # NS, EW
        if full_green:
            record[phase] = sum(obs["throughput"][direction] 
                                for direction in DIRS_IN_PHASE[phase])

        self.history.append(tuple(record))
        self.previous_obs = obs

        trends = (departure_trend(self.history, PHASE_NS), 
                  departure_trend(self.history, PHASE_EW))
        return {**obs, "departure_trends": trends}, reward, done, info