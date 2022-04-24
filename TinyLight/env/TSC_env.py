import cityflow
import json
import os
import torch
from .road import Road
from .intersection import Intersection
import numpy as np
import gym
from math import sqrt, floor
import platform
from utilities.utils import get_dumpable_config
if platform.system() == 'Darwin':
    from gym.envs.classic_control import rendering


class TSCEnv:
    def __init__(self, config):
        self.config = config
        self.eng = cityflow.Engine(json.dumps(get_dumpable_config(self.config)), thread_num=self.config['engine_thread'])
        self.roadnet_dic = self._get_roadnet()
        self.interval = self.config['interval']

        # info function dict
        self._info_functions = {
            'lane_2_num_vehicle': self._lane_2_num_vehicle,
            'lane_2_num_waiting_vehicle': self._lane_2_num_waiting_vehicle,
            'lane_2_sum_waiting_time': self._lane_2_sum_waiting_time,
            'lane_2_delay': self._lane_2_delay,
            'lane_2_num_vehicle_seg_by_k': self._lane_2_num_vehicle_seg_by_k,

            'inlane_2_num_vehicle': self._inlane_2_num_vehicle,
            'inlane_2_num_waiting_vehicle': self._inlane_2_num_waiting_vehicle,
            'inlane_2_sum_waiting_time': self._inlane_2_sum_waiting_time,
            'inlane_2_delay': self._inlane_2_delay,
            'inlane_2_pressure': self._inlane_2_pressure,
            'inlane_2_num_vehicle_seg_by_k': self._inlane_2_num_vehicle_seg_by_k,

            'outlane_2_num_vehicle': self._outlane_2_num_vehicle,
            'outlane_2_num_waiting_vehicle': self._outlane_2_num_waiting_vehicle,
            'outlane_2_sum_waiting_time': self._outlane_2_sum_waiting_time,
            'outlane_2_delay': self._outlane_2_delay,
            'outlane_2_num_vehicle_seg_by_k': self._outlane_2_num_vehicle_seg_by_k,

            'phase_2_num_vehicle': self._phase_2_num_vehicle,
            'phase_2_num_waiting_vehicle': self._phase_2_num_waiting_vehicle,
            'phase_2_sum_waiting_time': self._phase_2_sum_waiting_time,
            'phase_2_delay': self._phase_2_delay,
            'phase_2_pressure': self._phase_2_pressure,

            'inroad_2_num_vehicle': self._inroad_2_num_vehicle,
            'inroad_2_num_waiting_vehicle': self._inroad_2_num_waiting_vehicle,
            'inroad_2_sum_waiting_time': self._inroad_2_sum_waiting_time,
            'inroad_2_delay': self._inroad_2_delay,

            'inter_2_num_vehicle': self._inter_2_num_vehicle,
            'inter_2_num_waiting_vehicle': self._inter_2_num_waiting_vehicle,
            'inter_2_sum_waiting_time': self._inter_2_sum_waiting_time,
            'inter_2_delay': self._inter_2_delay,
            'inter_2_pressure': self._inter_2_pressure,
            'inter_2_vehicle_position_image': self._inter_2_vehicle_position_image,
            'inter_2_current_phase': self._inter_2_current_phase,
            'inter_2_next_phase': self._inter_2_next_phase,
            'inter_2_phase_has_changed': self._inter_2_phase_has_changed,
            'inter_2_num_passed_vehicle_since_last_action': self._inter_2_num_passed_vehicle_since_last_action,
            'inter_2_sum_travel_time_since_last_action': self._inter_2_sum_travel_time_since_last_action,

            'lanelink_2_pressure': self._lanelink_2_pressure,
            'lanelink_2_num_vehicle': self._lanelink_2_num_vehicle,

            # evaluation metric
            'world_2_average_travel_time': self._world_2_average_travel_time,
            'world_2_average_queue_length': self._world_2_average_queue_length,
            'world_2_average_throughput': self._world_2_average_throughput,
            'world_2_average_delay': self._world_2_average_delay,
        }
        self._vehicle_waiting_time = {}
        self._vehicle_trajectory = {}
        self._vehicle_trajectory_last_update_time = -1
        self._cache_queue_length = [0, 0, 0]  # number of sample, sum of sample, sum of sample^2
        self._cache_average_delay = [0, 0, 0]  # number of sample, sum of sample, sum of sample^2
        self._cache_throughput = [0, 0, 0, 0]  # number of sample, sum of sample, sum of sample^2, last throughput

        # parsing roads
        self.id2road = {}
        for road_dict in self.roadnet_dic['roads']:
            road_id = road_dict['id']
            self.id2road[road_id] = Road(road_id, road_dict)

        # parsing intersections
        self.id2intersection = {}
        self.n_intersection = []
        for inter_idx, inter_dict in enumerate(
                filter(lambda inter_: (not inter_['virtual']) and (len(inter_['trafficLight']['lightphases']) > 1),
                       self.roadnet_dic['intersections'])
        ):
            inter_id = inter_dict['id']
            inter = Intersection(inter_idx, inter_id, inter_dict, self)
            self.id2intersection[inter_id] = inter
            self.n_intersection.append(inter)
        # find out the neighbors of each intersection
        for inter_idx, inter_i in enumerate(self.n_intersection):
            for inter_jdx, inter_j in enumerate(self.n_intersection):
                if inter_idx == inter_jdx:
                    continue
                if not set(inter_i.n_out_road).isdisjoint(set(inter_j.n_in_road)):
                    inter_i.n_neighbor_idx.append(inter_j.inter_idx)
        self.n = len(self.n_intersection)

        # relevant parameters
        self.n_action_space = [gym.spaces.Discrete(len(inter.n_phase)) for inter in self.n_intersection]
        self.n_obs_shape = self._get_n_obs_shape()
        self.current_action = None
        self.last_action = None
        self._set_seed()
        self._viewer = None

        # reserved for cases where an agent wants to visit other agent's info
        self.n_agent = None
        self.n_obs = None

    def reset(self):
        self._vehicle_waiting_time = {}
        self._vehicle_trajectory = {}
        self._vehicle_trajectory_last_update_time = -1
        self._cache_queue_length = [0, 0, 0]
        self._cache_average_delay = [0, 0, 0]
        self._cache_throughput = [0, 0, 0, 0]
        self.current_action = None
        self.last_action = None

        self.eng.reset(self.config['seed'])
        for inter in self.n_intersection:
            inter.reset()
        self.n_obs = self._get_n_obs()
        return self.n_obs

    def _set_seed(self):
        self.eng.set_random_seed(self.config['seed'])
        for action_space in self.n_action_space:
            action_space.seed(self.config['seed'])
            action_space.np_random.seed(self.config['seed'])

    def _update_average_queue_length(self):
        for inter in self.n_intersection:
            lane2waiting_vehicle = self._lane_2_num_waiting_vehicle(inter)
            for waiting_vehicle in lane2waiting_vehicle:
                self._cache_queue_length[0] += 1
                self._cache_queue_length[1] += waiting_vehicle
                self._cache_queue_length[2] += waiting_vehicle * waiting_vehicle

    def _update_average_throughput(self):
        current_throughput = self.eng.get_finished_vehicle_cnt()
        throughput_this_minute = current_throughput - self._cache_throughput[3]
        self._cache_throughput[0] += 1
        self._cache_throughput[1] += throughput_this_minute
        self._cache_throughput[2] += throughput_this_minute * throughput_this_minute
        self._cache_throughput[3] = current_throughput

    def _update_average_delay(self):
        for inter in self.n_intersection:
            inlane2delay = self._inlane_2_delay(inter)
            for delay in inlane2delay:
                self._cache_average_delay[0] += 1
                self._cache_average_delay[1] += delay
                self._cache_average_delay[2] += delay * delay

    def step(self, n_action):
        self.last_action, self.current_action = self.current_action, n_action
        for _ in range(self.config['action_interval']):
            for inter, action in zip(self.n_intersection, n_action):
                inter.step(action, self.interval)
            self.eng.next_step()

            if self.config['render']:
                self.render(inter_idx=0)

        n_next_obs = self._get_n_obs()
        n_reward = self._get_n_reward()
        n_done = self._get_n_done()
        info = self._get_info()

        self.n_obs = n_next_obs

        if self.config['current_episode_step_idx'] % 60 == 0 and self.config['current_episode_step_idx'] > 0:
            metric_feature_list = self.config[self.config['cur_agent']]['metric_feature_list']
            if 'world_2_average_queue_length' in metric_feature_list:
                self._update_average_queue_length()
            if 'world_2_average_delay' in metric_feature_list:
                self._update_average_delay()
            if 'world_2_average_throughput' in metric_feature_list:
                self._update_average_throughput()

        return n_next_obs, n_reward, n_done, info

    def render(self, inter_idx):
        if self._viewer is None:
            self._viewer = rendering.SimpleImageViewer()
        inter_img = self._inter_2_vehicle_position_image(self.n_intersection[inter_idx], grid_height=1, grid_width=1)
        inter_img = np.uint8(np.repeat(inter_img.transpose([1, 2, 0]), 3, axis=2) / (np.max(inter_img) + 1e-6) * 255.0)
        self._viewer.imshow(inter_img)
        return self._viewer.isopen

    def _get_n_obs_shape(self):
        n_shape = []
        n_obs = self._get_n_obs()
        for inter_obs in n_obs:
            shape = []
            for feature in inter_obs:
                shape.append(feature.shape[1:])  # dim 0 is batch
            n_shape.append(shape)
        return n_shape

    def _get_n_obs(self):
        n_obs = []
        for inter in self.n_intersection:
            obs = []
            for observation_feature in self.config[self.config['cur_agent']]['observation_feature_list']:
                feature = self._info_functions[observation_feature](inter)
                obs.append(torch.from_numpy(feature).float().unsqueeze(0).to(self.config['device']))
            n_obs.append(obs)
        return n_obs

    def _get_n_reward(self):
        n_reward = []
        for inter in self.n_intersection:
            reward = np.array(0.)
            for reward_feature, reward_weight in zip(
                    self.config[self.config['cur_agent']]['reward_feature_list'],
                    self.config[self.config['cur_agent']]['reward_feature_weight']
            ):
                feature = self._info_functions[reward_feature](inter)
                reward = reward + feature * reward_weight
            n_reward.append(torch.from_numpy(reward).float().view(1, 1).to(self.config['device']))
        return n_reward

    def _get_n_done(self):
        return [False for _ in range(self.n)]

    def _get_info(self):
        info = {}
        for metric_feature in self.config[self.config['cur_agent']]['metric_feature_list']:
            info[metric_feature] = self._info_functions[metric_feature]()
        return info

    def _get_roadnet(self):
        roadnet_file_path = os.path.join(self.config['dir'], self.config['roadnetFile'])
        with open(roadnet_file_path) as f:
            roadnet_dic = json.load(f)
        return roadnet_dic

    ##############################################
    # Functions below are used for info collection
    ##############################################

    def _get_vehicle_waiting_time(self):
        n_vehicle_id = self.eng.get_vehicles()
        vehicle2speed = self.eng.get_vehicle_speed()
        for vehicle_id in n_vehicle_id:
            if vehicle_id not in self._vehicle_waiting_time.keys():  # vehicle appears for the first time
                self._vehicle_waiting_time[vehicle_id] = 0
            elif vehicle2speed[vehicle_id] < 0.1:  # vehicle is waiting
                self._vehicle_waiting_time[vehicle_id] += 1
            else:  # vehicle is moving
                self._vehicle_waiting_time[vehicle_id] = 0
        return self._vehicle_waiting_time

    def _get_vehicle_trajectory(self):
        cur_time = self.eng.get_current_time()
        if cur_time <= self._vehicle_trajectory_last_update_time:
            return self._vehicle_trajectory

        self._vehicle_trajectory_last_update_time = cur_time
        vehicle_2_lane = self._get_vehicle_2_lane()
        n_vehicle_id = self.eng.get_vehicles(include_waiting=False)
        for vehicle_id in n_vehicle_id:
            if vehicle_id not in self._vehicle_trajectory.keys():  # vehicle appears for the first time
                self._vehicle_trajectory[vehicle_id] = [{
                    "lane_id": vehicle_2_lane[vehicle_id],
                    "enter_time": int(cur_time),
                    "time_on_lane": 0
                }]
            else:
                if vehicle_id not in vehicle_2_lane.keys():
                    continue
                if vehicle_2_lane[vehicle_id] == self._vehicle_trajectory[vehicle_id][-1]["lane_id"]:  # on last lane
                    self._vehicle_trajectory[vehicle_id][-1]["time_on_lane"] += 1
                else:  # on a new lane
                    self._vehicle_trajectory[vehicle_id].append({
                        "lane_id": vehicle_2_lane[vehicle_id],
                        "enter_time": int(cur_time),
                        "time_on_lane": 0
                    })
        return self._vehicle_trajectory

    def _get_vehicle_2_lane(self):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_lane = {}
        for lane_id in lane_2_n_vehicle_id.keys():
            for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                vehicle_2_lane[vehicle_id] = lane_id
        return vehicle_2_lane

    def _concat_by_unit(self, lane_stat_dict, inter: Intersection, unit, inroad_only, outroad_only):
        assert unit in ['intersection', 'road', 'lane']

        result = []
        roadset = inter.n_road
        if inroad_only:
            roadset = inter.n_in_road
        elif outroad_only:
            roadset = inter.n_out_road
        # for road in (inter.n_in_road if inroad_only else inter.n_road):
        for road in roadset:
            result_by_road = []
            for lane_id in road.n_lane_id:
                result_by_road.append(lane_stat_dict[lane_id])
            if unit == 'lane':
                result_by_road = np.array(result_by_road)
            elif unit in ['road', 'intersection']:
                result_by_road = np.mean(result_by_road)
            result.append(result_by_road)

        if unit == 'lane':
            result = np.concatenate(result)
        elif unit == 'road':
            result = np.array(result)
        elif unit == 'intersection':
            result = np.array([np.mean(result)])
        return result

    def _lane_2_num_vehicle(self, inter):
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, inter, unit='lane', inroad_only=False, outroad_only=False)

    def _lane_2_num_waiting_vehicle(self, inter):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, inter, unit='lane', inroad_only=False, outroad_only=False)

    def _outlane_2_num_vehicle(self, inter: Intersection):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        n_outlane_2_num_vehicle = []
        for road in inter.n_out_road:
            for lane_id in road.n_lane_id:
                n_outlane_2_num_vehicle.append(lane_2_num_waiting_vehicle[lane_id])
        n_outlane_2_num_vehicle = np.array(n_outlane_2_num_vehicle)
        return n_outlane_2_num_vehicle

    def _inlane_2_num_vehicle(self, inter):
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, inter, unit='lane', inroad_only=True, outroad_only=False)

    def _inlane_2_num_waiting_vehicle(self, inter):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, inter, unit='lane', inroad_only=True, outroad_only=False)

    def _outlane_2_num_waiting_vehicle(self, inter):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, inter, unit='lane', inroad_only=False, outroad_only=True)

    def _inlane_2_sum_waiting_time(self, inter: Intersection):
        # the sum of waiting times of vehicles on the lane since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_lane_waiting_time = []
        for in_road in inter.n_in_road:
            for lane_id in in_road.n_lane_id:
                lane_waiting_time = 0.
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    lane_waiting_time += vehicle_2_waiting_time[vehicle_id]
                n_lane_waiting_time.append(lane_waiting_time)
        n_lane_waiting_time = np.array(n_lane_waiting_time)
        return n_lane_waiting_time

    def _outlane_2_sum_waiting_time(self, inter: Intersection):
        # the sum of waiting times of vehicles on the lane since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_lane_waiting_time = []
        for out_road in inter.n_out_road:
            for lane_id in out_road.n_lane_id:
                lane_waiting_time = 0.
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    lane_waiting_time += vehicle_2_waiting_time[vehicle_id]
                n_lane_waiting_time.append(lane_waiting_time)
        n_lane_waiting_time = np.array(n_lane_waiting_time)
        return n_lane_waiting_time

    def _lane_2_sum_waiting_time(self, inter: Intersection):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_lane_waiting_time = []
        for road in inter.n_road:
            for lane_id in road.n_lane_id:
                lane_waiting_time = 0.
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    lane_waiting_time += vehicle_2_waiting_time[vehicle_id]
                n_lane_waiting_time.append(lane_waiting_time)
        n_lane_waiting_time = np.array(n_lane_waiting_time)
        return n_lane_waiting_time

    def _inlane_2_delay(self, inter):
        # delay of each lane = 1. - lane_avg_speed / speed_limit
        # by default the speed limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_lane_delay = []

        for in_road in inter.n_in_road:
            for lane_id in in_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_speed_sum = 0
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
                if len(n_vehicle_id) == 0:
                    lane_avg_speed = speed_limit
                else:
                    lane_avg_speed = vehicle_speed_sum * 1.0 / len(n_vehicle_id)
                n_lane_delay.append(1. - lane_avg_speed / speed_limit)

        n_lane_delay = np.array(n_lane_delay)
        return n_lane_delay

    def _lane_2_delay(self, inter):
        # delay of each lane = 1. - lane_avg_speed / speed_limit
        # by default the speed limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_lane_delay = []

        for road in inter.n_road:
            for lane_id in road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_speed_sum = 0
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
                if len(n_vehicle_id) == 0:
                    lane_avg_speed = speed_limit
                else:
                    lane_avg_speed = vehicle_speed_sum * 1.0 / len(n_vehicle_id)
                n_lane_delay.append(1. - lane_avg_speed / speed_limit)

        n_lane_delay = np.array(n_lane_delay)
        return n_lane_delay

    def _outlane_2_delay(self, inter):
        # delay of each lane = 1. - lane_avg_speed / speed_limit
        # by default the speed limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_lane_delay = []

        for out_road in inter.n_out_road:
            for lane_id in out_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_speed_sum = 0
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
                if len(n_vehicle_id) == 0:
                    lane_avg_speed = speed_limit
                else:
                    lane_avg_speed = vehicle_speed_sum * 1.0 / len(n_vehicle_id)
                n_lane_delay.append(1. - lane_avg_speed / speed_limit)

        n_lane_delay = np.array(n_lane_delay)
        return n_lane_delay

    def _inlane_2_pressure(self, inter: Intersection):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()

        n_lane_pressure = []
        for in_road in inter.n_in_road:
            for lane_id in in_road.n_lane_id:
                lane_pressure = 0.0
                for road_link in inter.n_roadlink:
                    if road_link.startroad_id != in_road.road_id:
                        continue
                    for lane_link in road_link.n_lanelink_id:
                        if lane_link[0] == lane_id:
                            lane_pressure += lane_2_vehicle_count[lane_link[0]]
                            lane_pressure -= lane_2_vehicle_count[lane_link[1]]
                n_lane_pressure.append(lane_pressure)
        return np.array(n_lane_pressure)

    def _lanelink_2_pressure(self, inter: Intersection):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()
        n_lanelink_pressure = []
        for roadlink in inter.n_roadlink:
            for lanelink in roadlink.n_lanelink_id:
                lanelink_pressure = lane_2_vehicle_count[lanelink[0]] - lane_2_vehicle_count[lanelink[1]]
                n_lanelink_pressure.append(lanelink_pressure)
        return np.array(n_lanelink_pressure)

    def _lanelink_2_num_vehicle(self, inter: Intersection):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()
        n_lanelink_num_vehicle = []
        for roadlink in inter.n_roadlink:
            for lanelink in roadlink.n_lanelink_id:
                inlane_id, outlane_id = lanelink[0], lanelink[1]
                lanelink_num_vehicle = lane_2_vehicle_count[inlane_id] + lane_2_vehicle_count[outlane_id]
                n_lanelink_num_vehicle.append(lanelink_num_vehicle)
        return np.array(n_lanelink_num_vehicle)

    def _represent_feature_from_inlane_to_phase(self, inlane_feature, inter: Intersection):
        phase_2_passable_lane = inter.phase_2_passable_lane_idx
        lane_2_applicable_phase = np.transpose(phase_2_passable_lane)
        phase_feature = np.matmul(inlane_feature, lane_2_applicable_phase)
        return phase_feature

    def _phase_2_num_vehicle(self, inter: Intersection):
        inlane_2_num_vehicle = self._inlane_2_num_vehicle(inter)
        return self._represent_feature_from_inlane_to_phase(inlane_2_num_vehicle, inter)

    def _phase_2_num_waiting_vehicle(self, inter: Intersection):
        inlane_2_num_waiting_vehicle = self._inlane_2_num_waiting_vehicle(inter)
        return self._represent_feature_from_inlane_to_phase(inlane_2_num_waiting_vehicle, inter)

    def _phase_2_sum_waiting_time(self, inter: Intersection):
        inlane_2_sum_waiting_time = self._inlane_2_sum_waiting_time(inter)
        return self._represent_feature_from_inlane_to_phase(inlane_2_sum_waiting_time, inter)

    def _phase_2_delay(self, inter: Intersection):
        inlane_2_delay = self._inlane_2_delay(inter)
        return self._represent_feature_from_inlane_to_phase(inlane_2_delay, inter)

    def _phase_2_pressure(self, inter: Intersection):
        inlane_2_pressure = self._inlane_2_pressure(inter)
        return self._represent_feature_from_inlane_to_phase(inlane_2_pressure, inter)

    def _lane_2_num_vehicle_seg_by_k(self, inter: Intersection):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_distance = self.eng.get_vehicle_distance()
        k = self.config[self.config['cur_agent']]['K']

        n_lane_2_num_vehicle_seg_by_k = []
        for road in inter.n_road:
            for lane_id in road.n_lane_id:
                lane_2_num_vehicle_seg_by_k = [0.0 for _ in range(k)]
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                for vehicle_id in n_vehicle_id:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    idx = floor(vehicle_distance / (road.length / k))
                    idx = max(min(idx, k - 1), 0)
                    lane_2_num_vehicle_seg_by_k[idx] += 1.0
                n_lane_2_num_vehicle_seg_by_k.extend(lane_2_num_vehicle_seg_by_k)
        return np.array(n_lane_2_num_vehicle_seg_by_k)

    def _inlane_2_num_vehicle_seg_by_k(self, inter: Intersection):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_distance = self.eng.get_vehicle_distance()
        k = self.config[self.config['cur_agent']]['K']

        n_inlane_2_num_vehicle_seg_by_k = []
        for road in inter.n_in_road:
            for lane_id in road.n_lane_id:
                inlane_2_num_vehicle_seg_by_k = [0.0 for _ in range(k)]
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                for vehicle_id in n_vehicle_id:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    idx = floor(vehicle_distance / (road.length / k))
                    idx = max(min(idx, k - 1), 0)
                    inlane_2_num_vehicle_seg_by_k[idx] += 1.0
                n_inlane_2_num_vehicle_seg_by_k.extend(inlane_2_num_vehicle_seg_by_k)
        return np.array(n_inlane_2_num_vehicle_seg_by_k)

    def _outlane_2_num_vehicle_seg_by_k(self, inter: Intersection):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_distance = self.eng.get_vehicle_distance()
        k = self.config[self.config['cur_agent']]['K']

        n_outlane_2_num_vehicle_seg_by_k = []
        for road in inter.n_out_road:
            for lane_id in road.n_lane_id:
                outlane_2_num_vehicle_seg_by_k = [0.0 for _ in range(k)]
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                for vehicle_id in n_vehicle_id:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    idx = floor(vehicle_distance / (road.length / k))
                    idx = max(min(idx, k - 1), 0)
                    outlane_2_num_vehicle_seg_by_k[idx] += 1.0
                n_outlane_2_num_vehicle_seg_by_k.extend(outlane_2_num_vehicle_seg_by_k)
        return np.array(n_outlane_2_num_vehicle_seg_by_k)

    def _inroad_2_num_vehicle(self, inter):
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, inter, unit='road', inroad_only=True, outroad_only=False)

    def _inroad_2_num_waiting_vehicle(self, inter):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, inter, unit='road', inroad_only=True, outroad_only=False)

    def _inroad_2_sum_waiting_time(self, inter):
        # the sum of waiting time of vehicles on the road since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_road_waiting_time = []
        for in_road in inter.n_in_road:
            road_waiting_time = 0.
            for lane_id in in_road.n_lane_id:
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    road_waiting_time += vehicle_2_waiting_time[vehicle_id]
            n_road_waiting_time.append(road_waiting_time)
        n_road_waiting_time = np.array(n_road_waiting_time)
        return n_road_waiting_time

    def _inroad_2_delay(self, inter):
        # delay of each road = 1. - road_avg_speed / speed_limit
        # by default the speed_limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_road_delay = []

        for in_road in inter.n_in_road:
            vehicle_speed_sum = 0.
            vehicle_num = 0
            for lane_id in in_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_num += len(n_vehicle_id)
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
            if vehicle_num == 0:
                road_avg_speed = speed_limit
            else:
                road_avg_speed = vehicle_speed_sum * 1.0 / vehicle_num
            n_road_delay.append(1. - road_avg_speed / speed_limit)

        n_road_delay = np.array(n_road_delay)
        return n_road_delay

    def _inter_2_num_vehicle(self, inter):
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, inter, unit='intersection', inroad_only=True, outroad_only=False)

    def _inter_2_num_waiting_vehicle(self, inter):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, inter, unit='intersection', inroad_only=True, outroad_only=False)

    def _inter_2_sum_waiting_time(self, inter):
        # the sum of waiting times of vehicles on the intersection since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        inter_waiting_time = 0.
        for in_road in inter.n_in_road:
            for lane_id in in_road.n_lane_id:
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    inter_waiting_time += vehicle_2_waiting_time[vehicle_id]
        inter_waiting_time = np.array([inter_waiting_time])
        return inter_waiting_time

    def _inter_2_delay(self, inter):
        # delay of inter = 1. - inter_avg_speed / speed_limit
        # by default the speed_limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()

        vehicle_speed_sum = 0.
        vehicle_num = 0

        for in_road in inter.n_in_road:
            for lane_id in in_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_num += len(n_vehicle_id)
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]

        if vehicle_num == 0:
            inter_avg_speed = speed_limit  # follow the implementation of GeneraLight
        else:
            inter_avg_speed = vehicle_speed_sum * 1.0 / vehicle_num
        inter_delay = np.array([1. - inter_avg_speed / speed_limit])
        return inter_delay

    def _inter_2_pressure(self, inter):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()
        pressure = 0.
        for in_road in inter.n_in_road:
            for lane_id in in_road.n_lane_id:
                pressure += lane_2_vehicle_count[lane_id]
        for out_road in inter.n_out_road:
            for lane_id in out_road.n_lane_id:
                pressure -= lane_2_vehicle_count[lane_id]
        return np.array([pressure])

    def _inter_2_vehicle_position_image(self, inter: Intersection, grid_height=4, grid_width=4):
        # USE WITH CAUTION: this implementation follows IntelliLight and is only applicable for squared intersection
        area_height, area_width = 600, 600
        map_of_car = np.zeros((1, area_height // grid_height, area_width // grid_width))

        inter_x, inter_y = inter.inter_dict["point"]["x"], inter.inter_dict["point"]["y"]
        vehicle_2_distance = self.eng.get_vehicle_distance()
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()

        for road in inter.n_road:
            for lane_id in road.n_lane_id:
                start_x, start_y, norm_x, norm_y = self._get_lane_start_location(road.road_dict, int(lane_id.split('_')[-1]))
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    vehicle_x, vehicle_y = start_x + vehicle_distance * norm_x, start_y + vehicle_distance * norm_y
                    transform_x, transform_y = int((vehicle_x - inter_x + 300) / grid_width), int((vehicle_y - inter_y + 300) / grid_height)
                    transform_x = max(min(transform_x, int(area_width / grid_width) - 1), 0)
                    transform_y = max(min(transform_y, int(area_height / grid_height) - 1), 0)

                    flip_y = max(min(area_height // grid_height - transform_y, int(area_height / grid_height) - 1), 0)
                    map_of_car[0, flip_y, transform_x] += 1
        return map_of_car

    def _get_lane_start_location(self, road_dict, lane_idx):
        road_start_point = (road_dict["points"][0]["x"], road_dict["points"][0]["y"])
        road_end_point = (road_dict["points"][1]["x"], road_dict["points"][1]["y"])
        delta_x, delta_y = road_end_point[0] - road_start_point[0], road_end_point[1] - road_start_point[1]
        norm = sqrt(delta_x ** 2 + delta_y ** 2)
        delta_x, delta_y = delta_x / norm, delta_y / norm

        lane_width = 0.
        for lane_jdx in range(lane_idx):
            lane_width += road_dict["lanes"][lane_jdx]["width"]
        lane_width += road_dict["lanes"][lane_idx]["width"] / 2
        bias_x, bias_y = delta_y * lane_width, -1. * delta_x * lane_width  # 90 degree clockwise rotation
        return road_start_point[0] + bias_x, road_start_point[1] + bias_y, delta_x, delta_y

    def _inter_2_current_phase(self, inter: Intersection):
        phase_one_hot = np.zeros(len(inter.n_phase))
        phase_one_hot[inter.current_phase] = 1.0
        return phase_one_hot

    def _inter_2_next_phase(self, inter: Intersection):
        phase_one_hot = np.zeros(len(inter.n_phase))
        phase_one_hot[(inter.current_phase + 1) % len(inter.n_phase)] = 1.0
        return phase_one_hot

    def _inter_2_phase_has_changed(self, inter: Intersection):
        if self.last_action is None or self.current_action is None:
            return np.zeros(1)
        inter_idx = inter.inter_idx
        if self.last_action[inter_idx] == self.current_action[inter_idx]:
            return np.zeros(1)
        else:
            return np.ones(1)

    def _get_n_vehicle_id_passed_since_last_action(self, inter: Intersection):
        vehicle_2_trajectory = self._get_vehicle_trajectory()
        n_passed_vehicle_id = []

        for vehicle_id, trajectory in vehicle_2_trajectory.items():
            if len(trajectory) < 2:
                continue
            if trajectory[-2]["lane_id"] in inter.n_in_lane_id \
                    and trajectory[-1]["lane_id"] in inter.n_out_lane_id \
                    and trajectory[-1]["time_on_lane"] < self.interval:
                n_passed_vehicle_id.append(vehicle_id)
        return n_passed_vehicle_id

    def _inter_2_num_passed_vehicle_since_last_action(self, inter: Intersection):
        n_passed_vehicle_id = self._get_n_vehicle_id_passed_since_last_action(inter)
        return np.array(len(n_passed_vehicle_id))

    def _inter_2_sum_travel_time_since_last_action(self, inter: Intersection):
        n_passed_vehicle_id = self._get_n_vehicle_id_passed_since_last_action(inter)
        sum_travel_time = 0

        for passed_vehicle_id in n_passed_vehicle_id:
            sum_travel_time += self._vehicle_trajectory[passed_vehicle_id][-2]["time_on_lane"]
        return np.array(sum_travel_time)

    def _world_2_average_travel_time(self):
        return self.eng.get_average_travel_time(), self.eng.get_std_travel_time()

    def _world_2_average_queue_length(self):
        sample_number = self._cache_queue_length[0]
        mean = (self._cache_queue_length[1] / sample_number) if sample_number > 0 else 0
        std = sqrt(self._cache_queue_length[2] / sample_number - mean * mean) if sample_number > 1 else 0
        return mean, std

    def _world_2_average_throughput(self):
        sample_number = self._cache_throughput[0]
        mean = (self._cache_throughput[1] / sample_number) if sample_number > 0 else 0
        std = sqrt(self._cache_throughput[2] / sample_number - mean * mean) if sample_number > 1 else 0
        return mean, std

    def _world_2_average_delay(self):
        sample_number = self._cache_average_delay[0]
        mean = (self._cache_average_delay[1] / sample_number) if sample_number > 0 else 0
        std = sqrt(self._cache_average_delay[2] / sample_number - mean * mean) if sample_number > 1 else 0
        return mean, std
