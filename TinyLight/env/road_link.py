from typing import List, Tuple
from utilities.utils import list_with_unique_element


class RoadLink:
    def __init__(self, roadlink_dict, intersection):
        self.intersection = intersection

        self.startroad_id = roadlink_dict['startRoad']
        self.endroad_id = roadlink_dict['endRoad']

        # scan each lane link
        self.n_lanelink_id = []  # type: List[Tuple[str, str]]
        self.n_startlane_id = []  # type: List[str]
        for lanelink_dict in roadlink_dict['laneLinks']:
            startlane_id = '{}_{}'.format(self.startroad_id, lanelink_dict['startLaneIndex'])
            endlane_id = '{}_{}'.format(self.endroad_id, lanelink_dict['endLaneIndex'])
            self.n_lanelink_id.append((startlane_id, endlane_id))
            self.n_startlane_id.append(startlane_id)

        self.n_startlane_id = list_with_unique_element(self.n_startlane_id)

    def __str__(self):
        return str({'startroad_id': self.startroad_id,
                    'endroad_id': self.endroad_id,
                    'n_lanelink_id': self.n_lanelink_id})
