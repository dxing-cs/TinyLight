from typing import List, Tuple
from utilities.utils import list_with_unique_element


class Phase:
    def __init__(self, phase_idx, phase_dict, intersection):
        self.phase_idx = phase_idx
        self.intersection = intersection

        self.n_available_roadlink_idx = phase_dict['availableRoadLinks']  # type: List[int]
        self.n_available_lanelink_id = []  # type: List[Tuple[str, str]]
        self.n_available_startlane_id = []  # type: List[str]

        for available_roadlink_idx in self.n_available_roadlink_idx:
            roadlink = self.intersection.n_roadlink[available_roadlink_idx]
            self.n_available_lanelink_id.extend(roadlink.n_lanelink_id)
            self.n_available_startlane_id.extend(roadlink.n_startlane_id)
        self.n_available_startlane_id = list_with_unique_element(self.n_available_startlane_id)

    def __str__(self):
        return str({
            'phase_idx': self.phase_idx,
            'available_roadlink': self.n_available_roadlink_idx
        })
