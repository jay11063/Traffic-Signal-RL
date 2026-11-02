from dataclasses import dataclass

DIRECTION = ("N","S","E","W")
N,S,E,W = range(4)
PHASE_NS, PHASE_EW = 0,1

DIRS_IN_PHASE = {PHASE_NS: (N,S), PHASE_EW: (E,W)}
ACTION_HOLD, ACTION_SWITCH = 0,1

@dataclass(frozen=True)
class Config:
    decision_interval = 5
    min_green = 10
    max_green = 60
    clearance_time = 3
    inflow_duration = 1800
    drain_cap = 900

    lanes_per_approach = 2
    saturation_flow_per_lane = 1.0
    vehicle_occupancy_m = 7.0
    detection_range_m = 50

    history_length = 6
    reward_scale = 200.0

    def max_observable(self):
        return self.detection_range_m // self.vehicle_occupancy_m

    def nominal_capacity(self):
        return self.lanes_per_approach * self.saturation_flow_per_lane
