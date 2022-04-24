from .road import Road
from .road_link import RoadLink
from .phase import Phase
from typing import List
from utilities.utils import list_with_unique_element


class Intersection:
    def __init__(self, inter_idx, inter_id, inter_dict, env):
        self.inter_idx = inter_idx
        self.inter_id = inter_id
        self.inter_dict = inter_dict
        self.env = env
        self.eng = env.eng
        self.current_phase = 0
        self.current_phase_time = 0
        self.yellow_phase = -1
        self._yellow_time = 3

        self.n_road = []  # type: List[Road]
        self.n_in_road = []  # type: List[Road]
        self.n_out_road = []  # type: List[Road]
        self.n_lane_id = []  # type: List[str]
        self.n_in_lane_id = []  # type: List[str]
        self.n_out_lane_id = []  # type: List[str]

        # scan all road
        for road_id in self.inter_dict['roads']:
            road = self.env.id2road[road_id]
            self.n_road.append(road)

        # scan all roadlink
        self.n_roadlink = []  # type: List[RoadLink]
        self.n_num_lanelink = []  # type: List[int]
        for roadlink_dict in self.inter_dict['roadLinks']:
            roadlink = RoadLink(roadlink_dict, self)
            self.n_roadlink.append(roadlink)
            self.n_num_lanelink.append(len(roadlink.n_lanelink_id))

        # scan all in_road and out_road by roadlink
        for roadlink in self.n_roadlink:
            self.n_in_road.append(self.env.id2road[roadlink.startroad_id])
            self.n_out_road.append(self.env.id2road[roadlink.endroad_id])
        self.n_in_road = list_with_unique_element(self.n_in_road)
        self.n_out_road = list_with_unique_element(self.n_out_road)

        # fill in lane_id
        for road in self.n_road:
            self.n_lane_id.extend(road.n_lane_id)
        for in_road in self.n_in_road:
            self.n_in_lane_id.extend(in_road.n_lane_id)
        for out_road in self.n_out_road:
            self.n_out_lane_id.extend(out_road.n_lane_id)

        # scan all phase
        self.n_phase = []  # type: List[Phase]
        for phase_idx, phase_dict in enumerate(self.inter_dict['trafficLight']['lightphases']):
            if len(phase_dict['availableRoadLinks']) > 0:
                self.n_phase.append(Phase(phase_idx, phase_dict, self))

        self.n_neighbor_idx = [self.inter_idx]  # this will be determined in TSCEnv once all intersections are scanned
        self.phase_2_passable_lane_idx = self._get_phase_2_passable_lane_idx()
        self.phase_2_passable_lanelink_idx = self._get_phase_2_passable_lanelink_idx()

    def _get_phase_2_passable_lane_idx(self):
        phase_2_passable_lane_idx = []
        for phase_idx in range(len(self.n_phase)):
            n_lane = [0 for _ in range(len(self.n_in_lane_id))]
            for pass_lane_id in self.n_phase[phase_idx].n_available_startlane_id:
                lane_idx = self.n_in_lane_id.index(pass_lane_id)
                n_lane[lane_idx] = 1
            phase_2_passable_lane_idx.append(n_lane)
        return phase_2_passable_lane_idx

    def _get_phase_2_passable_lanelink_idx(self):
        phase_2_passable_lanelink_idx = []
        for phase_idx in range(len(self.n_phase)):
            n_lanelink = []
            for roadlink_idx, roadlink in enumerate(self.n_roadlink):
                if roadlink_idx in self.n_phase[phase_idx].n_available_roadlink_idx:
                    n_lanelink.extend([1 for _ in range(self.n_num_lanelink[roadlink_idx])])
                else:
                    n_lanelink.extend([0 for _ in range(self.n_num_lanelink[roadlink_idx])])
            phase_2_passable_lanelink_idx.append(n_lanelink)
        return phase_2_passable_lanelink_idx

    def step(self, action, interval):
        if self.current_phase == self.yellow_phase:
            if self.current_phase_time < self._yellow_time:
                self.current_phase_time += interval
            else:
                self.eng.set_tl_phase(self.inter_id, self.n_phase[action].phase_idx)
                self.current_phase = action
                self.current_phase_time = interval
        elif action == self.current_phase:
            self.current_phase_time += interval
        else:
            self.current_phase = self.yellow_phase
            self.current_phase_time = interval

    def reset(self):
        self.current_phase = 0
        self.current_phase_time = 0
        self.eng.set_tl_phase(self.inter_id, self.n_phase[self.current_phase].phase_idx)

    def __str__(self):
        return self.inter_id
