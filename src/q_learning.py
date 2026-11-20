import numpy as np
import json
from pathlib import Path

from .config import N, S, E, W
from .controllers import Controller

def bin_index(value, edges):
    low, high = edges
    if value <= low:
        return 0
    if value <= high:
        return 1
    return 2


def encode_state(obs):
    """
    return: (N queue, S queue, E queue, W queue, NS departures, EW departures, green time, phase)
    """
    queues = tuple(bin_index(count, (1,4)) for count in obs["queue_observation"])

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

    def __init__(self, learning_rate=0.1, gamma=0.95, epsilon=0.2, seed=0):
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

        state = encode_state(obs)
        q_value = self.q_table.get(state, (0.0,0.0))

        best_value = max(q_value[action] for action in legal_actions)
        best_action = [action for action in legal_actions 
                       if q_value[action]==best_value]

        return int(self.rng.choice(best_action))

    def update(self, obs, action, reward, next_obs, done):
        state = encode_state(obs)
        q_value = self.q_table.setdefault(state, [0.0, 0.0])

        target = reward
        if not done:
            next_state = encode_state(next_obs)
            next_values = self.q_table.get(next_state, (0.0,0.0))
            best_next_value = max(
                next_values[next_action] 
                for next_action, allowed in enumerate(next_obs["act_mask"]) 
                if allowed
            )
            target += self.gamma*best_next_value

        q_value[action] += self.learning_rate * (target - q_value[action])

def train_episode(env, agent, scenario):
    obs = env.reset(scenario)
    agent.reset()
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
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
