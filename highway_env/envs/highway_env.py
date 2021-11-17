import numpy as np
import logging
from typing import List, Tuple, Optional, Callable
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle

Observation = np.ndarray

logger = logging.getLogger(__name__)

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
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
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
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


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

class HighwayEnvLidar(HighwayEnvFast):
    """
    A variant of highway-fast-v0 with Adaptive Lidar Observations:
        - a new reward function based on lidar observations
        - flexibility for the lidar sampling rate to be changed
        - faster simulation frequency
    """
    
    def __init__(self, config: dict = None) -> None:
        self.lidar_buffer = np.zeros((16, 4)) # stored lidar points
        self.lidar_count = 0
        super().__init__(config)
    
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                    "type": "AdaptiveLidarObservation"
                    },
            "simulation_frequency": 5, # [Hz]
            "policy_frequency": 1,  # [Hz]
            "duration": 30 * 1,  # [max steps per episode]
            "vehicles_density": 1.5,
            "show_trajectories": False,
            "obstacle_distance_reward": -10,    # reward for distance between
                                                # ego-vehicle and closest lidar
                                                # point directly in front or
                                                # behind ego-vehicle.
            "right_lane_reward": 0.0,   # The reward received when driving on 
                                        # the right-most lanes, linearly mapped
                                        # to zero for other lanes.
            "high_speed_reward": 0.0,  # The reward received when driving at 
                                        # full speed, linearly mapped to zero 
                                        # for lower speeds according to 
                                        # config["reward_speed_range"].
            "smooth_driving_reward": 0.1,   # The reward received when IDLE is
                                            # chosen and there aren't many lane
                                            # changes.
            "distance_threshold": 15,   # [m] distance at which
                                        # obstacle_distance_reward becomes 
                                        # non-zero (3 car lengths).
            "adaptive_observations": True,
            "base_lidar_frequency": 0.5,  # [Hz], <= policy_frequency
            "reaction_distance": 15,    # [m] distance at which higher sampling
                                        # rates are used for lidar indices.
            "reaction_velocity": 7.5,   # [m/s] velocity at which higher 
                                        # rates are used for lidar indices.
            })
        return cfg

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.lidar_count = 0
        self.lidar_buffer = np.zeros((16, 4)) 
    
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster obstacle avoidance and driving at 
        high speed.

        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, \
            self.config["reward_speed_range"], [0, 1])
        dist_to_obstacle = max(self._find_closest_obstacle(), 0)
        distance_reward = 2 * self.config["obstacle_distance_reward"] / np.pi \
            * np.arctan(0.5 * -dist_to_obstacle) \
            + self.config["obstacle_distance_reward"]
        if action == 1 or action == 4:
            smooth_reward = self.config["smooth_driving_reward"]
        elif action == 0 or action == 2:
            smooth_reward = -self.config["smooth_driving_reward"]
        else:
            smooth_reward = 0
        reward = max(distance_reward, self.config["obstacle_distance_reward"])\
                 + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
                 + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
                 + smooth_reward
        reward = utils.lmap(reward,
                            [self.config["obstacle_distance_reward"] \
                                - self.config["smooth_driving_reward"],
                            self.config["high_speed_reward"] \
                                + self.config["right_lane_reward"] \
                                + self.config["smooth_driving_reward"]],
                            [0, 1])
        return reward
    
    def _find_closest_obstacle(self) -> float:
        """
        Uses the ego-vehicle global position and the global position of
        buffered lidar points to find the lidar point directly in front of or 
        behind the ego-vehicle that is closest to the ego-vehicle.
        """

        lane_width = self.road.network.lanes_list()[0].DEFAULT_WIDTH
        x_position = self.observation_type.POSITION_X
        y_position = self.observation_type.POSITION_Y
        # 1. get the current ego-vehicle lane
        ego_position = self.controlled_vehicles[0].position
        # 2. isolate the lidar points in front and behind the ego vehicle.
        # 3. find the closest lidar point.
        # 4. get the distance to that point.
        closest_lidar = np.array([np.inf,np.inf])
        closest_distance = np.linalg.norm(ego_position - closest_lidar)
        for lidar_point in self.lidar_buffer[:,[x_position,y_position]]:
            if lidar_point[1] - lane_width < ego_position[1] \
                    < lidar_point[1] + lane_width:
                distance = np.linalg.norm(ego_position - lidar_point)
                if distance < closest_distance:
                    closest_distance = distance
        # 5. return that distance.
        return closest_distance
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
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
        self._simulate(action)

        obs = self.lidar_buffer
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info    
    
    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] \
            // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % frames == 0:
                self.action_type.act(action)
                
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1 # [in simulation steps, not action steps]
            
            if self.config["adaptive_observations"] and \
                    self.time % int(self.config["simulation_frequency"] \
                    // self.config["base_lidar_frequency"]) == 0:
                self._observe()

            # the fastest observation frequency is the policy frequency
            if self.config["adaptive_observations"]:
                self._adaptively_observe()
            else:
                self._observe()

            # Automatically render intermediate simulation steps if a viewer 
            # has been launched. Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through 
                                    # env.render() as usual
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
        
    def _adaptively_observe(self) -> None:
        """
        Updates self.lidar_buffer if a set of specific lidar points in the 
        buffer require update according to the discrete reactive strategy
        laid out in the paper.
        """
        # 1. Assess current self.lidar_buffer to see which indices should be 
        # updated.
        x_position = self.observation_type.POSITION_X
        y_position = self.observation_type.POSITION_Y
        center_distances = np.linalg.norm(self.lidar_buffer[:, \
            [x_position, y_position]] \
            - self.controlled_vehicles[0].position,axis=1)
        distance_logic = (center_distances < self.config["reaction_distance"]) 
        velocity_logic = (abs(self.lidar_buffer[:,self.observation_type.VELOCITY]) \
            > self.config["reaction_velocity"])
        indexes_to_update = np.where(distance_logic \
            + velocity_logic)[0].tolist()
        if len(indexes_to_update) != 0:
            logger.debug(f"Updating {indexes_to_update} according to adaptive sensor strategy.")
        # 2. Call AdaptiveLidarObservation.selectively_observe(indices) where 
        # indices are the specific indices that need to be updated.
        updated_indexes = self.observation_type.selectively_observe(
            indexes_to_update)
        # 3. Update self.lidar_buffer with the new indices, without overwriting
        # indices that didn't need to be updated.
        self.lidar_buffer[indexes_to_update,:] = updated_indexes
        # 4. Increase self.lidar_count with the number of indices.
        self.lidar_count += len(indexes_to_update)

    def get_available_actions(self) -> List[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        if not isinstance(self.action_type, DiscreteMetaAction):
            raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = [self.action_type.actions_indexes['IDLE']]
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_RIGHT'])
        if self.vehicle.speed_index < self.vehicle.SPEED_COUNT - 1 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['FASTER'])
        if self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['SLOWER'])
        return actions
        
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
