import numpy as np
from typing import Dict, Text, List, Tuple, Optional, Callable
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "initial_ego_speed": 25,
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "vehicle_speeds": np.linspace(20, 30, 3),  # The available target
                                                       # speeds for MDPVehicles.
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with
                                       # a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the
                                       # right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at
                                       # full speed, linearly mapped to zero for
                                       # lower speeds according to
                                       # config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change
                                       # action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=self.config["initial_ego_speed"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed)
            vehicle.target_speeds = self.config["vehicle_speeds"]
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            num_front = others // 2
            num_rear = others - num_front
            # Add vehicles in front of the most forward vehicle.
            for _ in range(num_front):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.target_speeds = self.config["vehicle_speeds"]
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
            # Add vehicles behind the most rearward vehicle.
            for _ in range(num_rear):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"],
                    add_backwards=True)
                vehicle.target_speeds = self.config["vehicle_speeds"]
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) *
                     reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class HighwayEnvLidar(HighwayEnv):
    """
    A variant of HighwayEnv with Lidar Observations:
        - a new reward function based on lidar observations
        - flexibility for the lidar sampling rate to be changed
        - faster simulation frequency
    """

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.lidar_buffer = np.ones((16, 4)) * np.inf  # stored lidar points:
        #    0: radial distance
        #    1: position_x
        #    2: position_y
        #    3: relative velocity
        self.lidar_count = 0
        self.obs_step = 0

        self.indexes_to_update = []

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "AdaptiveLidarObservation"
            },
            "vehicle_type_distribution": {
                "sedan": 0.4,
                "truck": 0.5,
                "semi": 0.1,
            },
            # Simulation related configurations.
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "duration": 30 * 1,  # [max steps per episode]
            "base_lidar_frequency": 0.5,  # [Hz], <= policy_frequency
            # Scenario related configurations.
            "vehicles_count": 14,
            "lanes_count": 2,
            "initial_ego_speed": 30,  # [m/s]
            "road_speed_limit": 33,  # [m/s]
            "vehicle_speeds": np.linspace(10, 35, 5),  # [m/s]
            "vehicles_density": 2.0,
            "adaptive_observations": True,
            "constant_base_lidar": False,  # uses base lidar frequency without
                                           # adaptive sampling.
            # Observations related configurations.
            "reaction_distance": 15,    # [m] distance at which higher sampling
                                        # rates are used for lidar indices.
            "reaction_velocity": 7.5,   # [m/s] velocity at which higher
                                        # rates are used for lidar indices.
            # Rewards related configuraions.
            "obstacle_distance_reward": -10,    # reward for distance between
                                                # ego-vehicle and closest lidar
                                                # point directly in front or
                                                # behind ego-vehicle.
            "distance_threshold": 15,   # [m] distance at which
                                        # obstacle_distance_reward becomes
                                        # non-zero (~3 car lengths).
            "right_lane_reward": 1.0,   # The reward received when driving on
                                        # the right-most lanes, linearly mapped
                                        # to zero for other lanes.
            "high_speed_reward": 1.0,  # The reward received when driving at
                                        # full speed, linearly mapped to zero
                                        # for lower speeds according to
                                        # config["reward_speed_range"].
            "reward_speed_range": [20, 40],
            "smooth_driving_reward": 1.0,   # The reward received when IDLE is
                                            # chosen.
        })
        return cfg

    def _reset(self) -> None:
        super()._reset()
        self._distribute_vehicle_types()
        self.lidar_count = 0
        self.lidar_buffer = self.observation_type.observe()
        self.obs_step = 0
        self.indexes_to_update = []

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"],
                speed_limit=self.config["road_speed_limit"]),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"])

    def _distribute_vehicle_types(self):
        """
        Change vehicles distributed throughout the highway according to the
        config parameter vehicle type distribution.
        """
        vehicle_types = self.config["vehicle_type_distribution"]
        assert sum(vehicle_types.values()) == 1.0, \
            "the distribution of vehicle types must total 1.0."
        vehicles = self.road.vehicles[1:]
        f150_width, f150_length = 2.2, 6.0
        big_rig_width, big_rig_length = 2.6, 18.0
        vehicle_index = np.arange(len(vehicles))
        self.np_random.shuffle(vehicle_index)
        num_sedans = round(vehicle_types["sedan"] * len(vehicles))
        num_trucks = round(vehicle_types["truck"] * len(vehicles))
        truck_indices = vehicle_index[num_sedans:num_sedans + num_trucks]
        semi_indices = vehicle_index[num_sedans + num_trucks:]
        for i in truck_indices:
            vehicles[i].LENGTH = f150_length
            vehicles[i].WIDTH = f150_width
            vehicles[i].diagonal = np.sqrt(f150_width**2 + f150_length**2)
        # List semi_indices by vehicle location in decreasing x position.
        sorted_semi_indices = sorted(
            semi_indices, key=lambda i: vehicles[i].position[0], reverse=True)
        # Create semi trucks by modifying existing vehicles.
        for i in sorted_semi_indices:
            # Change vehicle parameters and goals to model a semi truck.
            vehicles[i].LENGTH = big_rig_length
            vehicles[i].DISTANCE_WANTED = self.vehicle.LENGTH * \
                2 + big_rig_length / 2
            vehicles[i].WIDTH = big_rig_width
            vehicles[i].diagonal = np.sqrt(
                big_rig_length**2 + big_rig_width**2)
            # Move semi trucks so they are 2 other car lengths away from other
            # vehicles so not initialized in a collision.
            front_vehicle, rear_vehicle = vehicles[i].road.neighbour_vehicles(
                vehicles[i], vehicles[i].lane_index)
            # If semi is initialized too close to the vehicle in front, move all
            # vehicles in that lane behind the front vehicle back.
            safe_distance = vehicles[i].LENGTH / 2. + self.vehicle.LENGTH * 2.5
            if front_vehicle and \
                    front_vehicle.position[0] - vehicles[i].position[0] <= \
                    front_vehicle.LENGTH / 2. + vehicles[i].LENGTH / 2. + \
                    self.vehicle.LENGTH * 2.:
                self._shift_all_vehicles_behind(front_vehicle, safe_distance)

            # If semi is initialized too close to the vehicle in rear, move all
            # vehicles in that lane behind the semi back.
            if rear_vehicle and \
                    vehicles[i].position[0] - rear_vehicle.position[0] <= \
                    vehicles[i].LENGTH / 2. + rear_vehicle.LENGTH / 2. + \
                    self.vehicle.LENGTH * 2.:
                self._shift_all_vehicles_behind(vehicles[i], safe_distance)

    def _shift_all_vehicles_behind(self, vehicle, distance):
        """Move all vehicles behind |vehicle| in the same lane back |distance|."""
        _, rear_vehicle = vehicle.road.neighbour_vehicles(
            vehicle, vehicle.lane_index)
        if rear_vehicle:
            # Move the vehicle behind rear_vehicle before moving rear_vehicle to
            # prevent skipping any vehicles.
            self._shift_all_vehicles_behind(rear_vehicle, distance)
            rear_vehicle.position[0] -= distance

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster obstacle avoidance and driving at 
        high speed.

        :param action: the last action performed
        :return: the corresponding reward
        """

        # Normalize the total reward.
        rewards = self._rewards(action)
        reward = sum(rewards.values())
        min_reward = self.config["obstacle_distance_reward"] \
            - self.config["smooth_driving_reward"]
        max_reward = self.config["high_speed_reward"] \
            + self.config["right_lane_reward"] \
            + self.config["smooth_driving_reward"]
        reward = utils.lmap(reward, [min_reward, max_reward], [0, 1])
        return reward

    def _rewards(self, action: Action) -> dict:
        """Reward function is based solely on LiDAR points."""
        # Define distance reward function.
        def dist_reward_func(dist): return \
            2 * self.config["obstacle_distance_reward"] / np.pi * np.arctan(
                0.5 * -dist) + self.config["obstacle_distance_reward"]
        collision_dist = self.vehicle.LENGTH
        dist_to_obstacle = max(self.find_closest_obstacle(), 0) \
            - collision_dist
        distance_reward = dist_reward_func(dist_to_obstacle)
        distance_reward = max(distance_reward,
                              self.config["obstacle_distance_reward"])

        # Reward for choosing IDLE, punish if changing lanes.
        actions = self.action_type.actions_indexes
        if action == actions["IDLE"]:
            smooth_reward = self.config["smooth_driving_reward"]
        elif action == actions["LANE_LEFT"] or action == actions["LANE_RIGHT"]:
            smooth_reward = -self.config["smooth_driving_reward"]
        else:
            smooth_reward = 0

        # Reward for traveling fast.
        scaled_speed = utils.lmap(self.vehicle.speed,
                                  self.config["reward_speed_range"], [0, 1])
        high_speed_reward = self.config["high_speed_reward"] * \
            np.clip(scaled_speed, 0, 1)

        # Reward for being in the right lane.
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if \
            isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        right_lane_reward = self.config["right_lane_reward"] * \
            lane / max(len(neighbours) - 1, 1)

        return {
            "distance": distance_reward, "right_lane": right_lane_reward,
            "high speed": high_speed_reward, "smooth": smooth_reward}

    def get_reward_breakdown(self, action: Action) -> dict:
        return self._rewards(action)

    def find_closest_obstacle(self) -> float:
        """
        Uses the ego-vehicle global position and the global position of
        buffered lidar points to find the lidar point directly in front of or 
        behind the ego-vehicle that is closest to the ego-vehicle.
        """
        # Use a lane buffer to capture vehicles immediately close to agent.
        lane_width = self.road.network.lanes_list()[0].DEFAULT_WIDTH
        lane_ratio = 0.7
        lane_buffer = lane_width * lane_ratio
        # x is the along the lane dimension, y is cross-lane.
        x_position = self.observation_type.POSITION_X
        y_position = self.observation_type.POSITION_Y
        # 1. get the current ego-vehicle x, y location.
        ego_position = self.vehicle.position
        # 2. isolate the lidar points in front and behind the ego vehicle.
        # 3. find the closest lidar point.
        # 4. get the distance to that point.
        lidar_points = self.lidar_buffer[:, [x_position, y_position]]
        distances = np.linalg.norm(lidar_points - ego_position, axis=1)
        points_in_buffer = np.logical_and(
            lidar_points[:, 1] - lane_buffer < ego_position[1],
            ego_position[1] < lidar_points[:, 1] + lane_buffer)
        try:
            closest_distance = np.min(distances[points_in_buffer])
        except ValueError:
            closest_distance = np.inf

        # closest_lidar = np.array([np.inf, np.inf])
        # closest_distance = np.linalg.norm(ego_position - closest_lidar)
        # for lidar_point in self.lidar_buffer[:, [x_position, y_position]]:
        #     if lidar_point[1] - lane_buffer < ego_position[1] \
        #             < lidar_point[1] + lane_buffer:
        #         distance = np.linalg.norm(ego_position - lidar_point)
        #         if distance < closest_distance:
        #             closest_distance = distance
        # 5. return that distance.
        return closest_distance

    def step(self, action: Action) -> \
            Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on 
        the road performs their default behaviour for several simulation 
        timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be \
                initialized in the environment implementation")

        self.steps += 1
        if action is not None and not self.config["manual_control"]:
            self.action_type.act(action)
        self._simulate(action)

        obs = self.lidar_buffer
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)

        return obs, reward, terminated, truncated, info

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        try:
            info = {
                "speed": self.vehicle.speed,
                "crashed": self.vehicle.crashed,
                "action": action,
                "rewards": self._rewards(action)
            }
        except NotImplementedError:
            pass
        return info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        assert self.config["simulation_frequency"] >= \
            self.config["policy_frequency"]
        assert self.config["simulation_frequency"] >= \
            self.config["base_lidar_frequency"]
        assert self.config["adaptive_observations"] != \
            self.config["constant_base_lidar"], \
            "Adaptive observations cannot be used when using a constant "\
            "base lidar frequency."
        frames = int(self.config["simulation_frequency"]
                     // self.config["policy_frequency"])
        for frame in range(frames):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1 / self.config["simulation_frequency"]
            # Account for floating point errors in comparisons.
            self.time = round(self.time, 10)

            # Implement observation logic for adaptive frequency. The base lidar
            # frequency is set by a config parameter. The highest lidar
            # frequency is set by the policy frequency.
            if self.time >= round(
                    self.obs_step / self.config["base_lidar_frequency"], 10):
                self._observe()
                self.obs_step += 1
            elif self.config["adaptive_observations"] and self.time >= round(
                    self.steps / self.config["policy_frequency"], 10):
                self._adaptively_observe()

            # Automatically render intermediate simulation steps if a viewer
            # has been launched. Ignored if the rendering is done offscreen
            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

    def _observe(self) -> None:
        """
        Updates self.lidar_buffer with all lidar samples.
        """
        # 1. Call AdaptiveLidarObservation.observe() so that all lidar samples
        # are collected.
        # 2. Update self.lidar_buffer with the new lidar data in absolute.
        self.lidar_buffer = self.observation_type.observe()
        # 4. Increase self.lidar_count.
        self.lidar_count += self.observation_type.cells
        self.indexes_to_update = list(range(0, self.observation_type.cells))

    def _get_center_distances_from_buffer(self) -> np.ndarray:
        """
        Computes the new center distance based on current ego position and the
        buffered lidar point locations. This will not always be the same as the
        buffered lidar point distances because the original may be old.
        """
        x_position = self.observation_type.POSITION_X
        y_position = self.observation_type.POSITION_Y
        center_distances = np.linalg.norm(
            self.lidar_buffer[:, [x_position, y_position]]
            - self.vehicle.position, axis=1)
        return center_distances

    def _adaptively_observe(self) -> None:
        """
        Updates self.lidar_buffer if a set of specific lidar points in the 
        buffer require update according to the discrete reactive strategy
        laid out in the paper.
        """
        # 1. Assess current self.lidar_buffer to see which indices should be
        # updated.
        center_distances = self._get_center_distances_from_buffer()
        distance_logic = (center_distances < self.config["reaction_distance"])
        velocity_logic = (abs(self.lidar_buffer[:, self.observation_type.VELOCITY])
                          > self.config["reaction_velocity"])
        self.indexes_to_update = np.where(distance_logic
                                          + velocity_logic)[0].tolist()
        # 2. Call AdaptiveLidarObservation.selectively_observe(indices) where
        # indices are the specific indices that need to be updated.
        updated_indexes = self.observation_type.selectively_observe(
            self.indexes_to_update)
        # 3. Update self.lidar_buffer with the new indices, without overwriting
        # indices that didn't need to be updated.
        self.lidar_buffer[self.indexes_to_update, :] = updated_indexes
        # 4. Increase self.lidar_count with the number of indices.
        self.lidar_count += len(self.indexes_to_update)


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)

register(
    id='highway-lidar-v0',
    entry_point='highway_env.envs:HighwayEnvLidar')
