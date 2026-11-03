from .config import ACTION_HOLD, ACTION_SWITCH, DIRS_IN_PHASE


class Controller:
    name = "base"
    def reset(self): pass
    def act(self, obs):
        raise NotImplementedError


class FixedTime(Controller):
    name = "fixed_time"

    def __init__(self, green_times=(45, 10)):
        self.green_times = green_times

    def act(self, obs) -> int:
        mask = obs["act_mask"]

        if not mask[ACTION_HOLD]:
            return ACTION_SWITCH

        if not mask[ACTION_SWITCH]:
            return ACTION_HOLD

        phase = obs["phase"]
        target_time = self.green_times[phase]

        if obs["phase_time"] >= target_time:
            return ACTION_SWITCH

        return ACTION_HOLD


class RuleBased(Controller):
    name = "rule_based"

    def __init__(self, threshold=2):
        self.threshold = threshold

    def act(self, obs) -> int:
        mask = obs["act_mask"]

        if not mask[ACTION_HOLD]:
            return ACTION_SWITCH

        if not mask[ACTION_SWITCH]:
            return ACTION_HOLD

        phase = obs["phase"]
        queues = obs["queue_observation"]

        current_queue = sum(queues[direction] for direction in DIRS_IN_PHASE[phase])

        other_queue = sum(queues[direction] for direction in DIRS_IN_PHASE[1 - phase])

        difference = other_queue - current_queue

        if difference > 0 and difference >= self.threshold:
            return ACTION_SWITCH

        return ACTION_HOLD


def run_episode(env, controller, scenario) -> dict:
    obs = env.reset(scenario)
    controller.reset()

    logs = []
    done = False

    while not done:
        action = controller.act(obs)

        obs, reward, done, info = env.step(action)

        logs.append({
            **info,
            "action": action,
            "reward": reward
        })

    return {
        "metrics": env.metrics(),
        "logs": logs
    }

