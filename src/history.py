from .config import DIRS_IN_PHASE


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


def interval_departures(previous_obs, obs, decision_interval):
    """Return (NS count, EW count) for one completed decision interval."""
    phase = obs["phase"]
    same_phase = phase == previous_obs["phase"]
    elapsed_green = obs["phase_time"] - previous_obs["phase_time"]

    record = [None, None]
    if same_phase and elapsed_green == decision_interval:
        record[phase] = sum(
            obs["throughput"][direction]
            for direction in DIRS_IN_PHASE[phase]
        )

    return tuple(record)