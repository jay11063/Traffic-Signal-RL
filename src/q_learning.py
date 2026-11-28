import numpy as np
import json
from pathlib import Path

from .config import N, S, E, W, PHASE_NS, PHASE_EW
from .controllers import Controller
from collections import deque
from .history import departure_trend, interval_departures

def bin_index(value, edges):
    low, high = edges
    if value <= low:
        return 0
    if value <= high:
        return 1
    return 2


def encode_state(obs, queue_edges=(1, 4)):
    """
    return: (N queue, S queue, E queue, W queue, NS departures, EW departures, green time, phase)
    """
    queues = tuple(
        bin_index(count, queue_edges)
        for count in obs["queue_observation"]
    )

    departures = obs["throughput"]
    ns_departures = bin_index(departures[N] + departures[S], (0,4))
    ew_departures = bin_index(departures[E] + departures[W], (0,4))

    green_time = bin_index(obs["phase_time"], (9,19))

    state = (*queues, ns_departures, ew_departures, green_time, obs["phase"])

    if "departure_trends" in obs:
        return (*state, *obs["departure_trends"])

    return state


class QLearning(Controller):
    name="q_learning"

    def __init__(self, learning_rate=0.1, gamma=0.95, epsilon=0.2, seed=0, queue_edges=(1, 4)):
        self.queue_edges = queue_edges
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = {}

    def act(self, obs):
        legal_actions = [action for action, allowed in enumerate(obs["act_mask"]) if allowed]

        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        state = encode_state(obs, self.queue_edges)
        q_value = self.q_table.get(state, (0.0,0.0))

        best_value = max(q_value[action] for action in legal_actions)
        best_action = [action for action in legal_actions 
                       if q_value[action]==best_value]

        return int(self.rng.choice(best_action))

    def update(self, obs, action, reward, next_obs, done):
        state = encode_state(obs, self.queue_edges)
        q_value = self.q_table.setdefault(state, [0.0, 0.0])

        target = reward
        if not done:
            next_state = encode_state(next_obs, self.queue_edges)
            next_values = self.q_table.get(next_state, (0.0,0.0))
            best_next_value = max(
                next_values[next_action] 
                for next_action, allowed in enumerate(next_obs["act_mask"]) 
                if allowed
            )
            target += self.gamma*best_next_value

        q_value[action] += self.learning_rate * (target - q_value[action])


class HistoryQLearning(QLearning):
    def __init__(self, cfg, **kwargs):
        super().__init__(queue_edges=cfg.queue_edges(), **kwargs)
        self.decision_interval = cfg.decision_interval
        self.history = deque(maxlen=cfg.history_length)
        self.previous_obs = None

    def reset(self):
        self.history.clear()
        self.previous_obs = None

    def observe(self, obs):
        if self.previous_obs is not None:
            record = interval_departures(
                self.previous_obs, obs, self.decision_interval
            )
            self.history.append(record)

        self.previous_obs = obs
        trends = (
            departure_trend(self.history, PHASE_NS),
            departure_trend(self.history, PHASE_EW),
        )
        return {**obs, "departure_trends": trends}


def train_episode(env, agent, scenario):
    agent.reset()
    obs = agent.observe(env.reset(scenario))
    done = False

    while not done:
        action = agent.act(obs)
        raw_next_obs, reward, done, _ = env.step(action)
        next_obs = agent.observe(raw_next_obs)
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs

    return env.metrics()


def save_q_table(q_table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [[list(state), values] for state, values in q_table.items()]
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, allow_nan=False)


def load_q_table(path):
    with Path(path).open("r", encoding="utf-8") as file:
        rows = json.load(file)

    return {tuple(state): values for state, values in rows}
