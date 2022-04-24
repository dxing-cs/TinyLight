from math import atan2, pi, sqrt
from typing import List


class Road:
    def __init__(self, road_id, road_dict):
        self.road_id = road_id
        self.road_dict = road_dict

        self.road_direction = self._get_road_direction()
        self.n_lane_id = []  # type: List[str]
        self.length = self._get_road_length()
        for lane_idx in range(len(self.road_dict['lanes'])):
            self.n_lane_id.append('{}_{}'.format(road_dict['id'], lane_idx))

    def _get_road_direction(self):
        # 0.00 -> east, 1/2 * pi -> north, pi -> west, 3/2 * pi -> south
        delta_x = self.road_dict['points'][1]['x'] - self.road_dict['points'][0]['x']
        delta_y = self.road_dict['points'][1]['y'] - self.road_dict['points'][0]['y']
        direction = atan2(delta_y, delta_x)
        return direction if direction >= 0 else (direction + 2 * pi)

    def _get_road_length(self):
        delta_x = self.road_dict['points'][1]['x'] - self.road_dict['points'][0]['x']
        delta_y = self.road_dict['points'][1]['y'] - self.road_dict['points'][0]['y']
        return sqrt(delta_x ** 2 + delta_y ** 2)

    def __str__(self):
        return str({'road_id': self.road_id,
                    'num_lanes': len(self.n_lane_id)})
