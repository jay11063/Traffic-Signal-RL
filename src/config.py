from dataclasses import dataclass

DIRECTION = ("N","S","E","W")
N,S,E,W = range(4)
PHASE_NS, PHASE_EW = 0,1

DIRS_IN_PHASE = {PHASE_NS: (N,S), PHASE_EW: (E,W)}
ACTION_HOLD, ACTION_SWITCH = 0,1

@dataclass(frozen=True)
class Config:
    decision_interval: int = 5
    min_green: int = 10
    max_green: int = 60
    clearance_time : int= 3
    inflow_duration: int = 1800
    drain_cap: int = 900

    lanes_per_approach: int = 2
    saturation_flow_per_lane:float = 0.5
    vehicle_occupancy_m:float = 7.0
    detection_range_m: int = 50

    history_length: int = 6
    reward_scale:float = 200.0

    def max_observable(self):
        return int(self.detection_range_m // self.vehicle_occupancy_m)

    def nominal_capacity(self):
        return self.lanes_per_approach * self.saturation_flow_per_lane
