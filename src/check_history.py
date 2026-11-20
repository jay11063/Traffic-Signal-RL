from .config import Config, PHASE_NS, PHASE_EW
from .history import DECREASING, MAINTAINING, INCREASING, UNAVAILABLE
from .q_learning import HistoryQLearning


def obs(phase, time, departures):
    return {
        "phase": phase,
        "phase_time": time,
        "throughput": departures,  # N, S, E, W
    }


def main():
    cfg = Config()
    agent = HistoryQLearning(cfg)
    ns, ew = PHASE_NS, PHASE_EW

    def check(phase, time, departures, expected):
        result = agent.observe(obs(phase, time, departures))
        assert result["departure_trends"] == expected, result

    # 초기 관측은 완료된 구간이 아니다.
    check(ns, 0, (0, 0, 0, 0), (UNAVAILABLE, UNAVAILABLE))
    assert len(agent.history) == 0

    # NS 유효 기록 3 → 4 → 5. 첫 값과의 차이는 차례로 +1, +2.
    check(ns, 5,  (2, 1, 0, 0), (UNAVAILABLE, UNAVAILABLE))
    check(ns, 10, (2, 2, 0, 0), (MAINTAINING, UNAVAILABLE))
    check(ns, 15, (3, 2, 0, 0), (INCREASING, UNAVAILABLE))

    # 전환 후 EW 녹색은 2초뿐. 통과 차량이 있어도 양쪽 모두 제외한다.
    check(ew, 2, (0, 0, 1, 1), (INCREASING, UNAVAILABLE))
    assert agent.history[-1] == (None, None)

    # EW 유효 기록 2 → 0. 녹색 중 0대는 유효하고 차이 −2는 decreasing.
    check(ew, 7,  (0, 0, 1, 1), (INCREASING, UNAVAILABLE))
    check(ew, 12, (0, 0, 0, 0), (INCREASING, DECREASING))
    assert agent.history[-1] == (None, 0)

    # 7번째 기록에서 NS의 첫 값 3이 밀려난다. 남은 4 → 5는 maintaining.
    check(ew, 17, (0, 0, 0, 0), (MAINTAINING, DECREASING))
    assert len(agent.history) == cfg.history_length == 6

    # 8번째 기록에서 NS에는 5 하나만 남아 unavailable이 된다.
    check(ew, 22, (0, 0, 0, 0), (UNAVAILABLE, DECREASING))

    # controller.reset() 직후 지난 에피소드의 기록은 사라진다.
    agent.reset()
    assert len(agent.history) == 0
    check(ns, 0, (0, 0, 0, 0), (UNAVAILABLE, UNAVAILABLE))
    assert len(agent.history) == 0

    print("History checks passed.")


if __name__ == "__main__":
    main()