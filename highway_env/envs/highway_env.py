import numpy as np
from typing import List, Tuple, Optional, Callable
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle

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
        self.lidar_buffer = np.ndarray # stored lidar points
        self.total_collected_lidar_points = 0
        super().__init__(config)
    
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                    "type": "LidarObservation"
                    },
            "simulation_frequency": 100, # [Hz]
            "policy_frequency": 5,  # [Hz]
            "duration": 30 * 5,  # [max steps per episode]
            "vehicles_density": 1.5,
            "show_trajectories": False,
            "max_obstacle_dist_cost": 10,
            "distance_threshold": 13.5, # [m]
            "adaptive_observations": True,
            "base_lidar_frequency": 1, # [Hz], should be slower than policy, but faster than simulation frequency
            })
        return cfg
    
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster obstacle avoidance and highspeed.
        :param action: the last action performed
        :return: the corresponding reward
        """
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        cost_slope = self.config["max_obstacle_dist_cost"] / -self.config["distance_threshold"]
        cost_intercept = self.config["max_obstacle_dist_cost"]
        dist_2_obstacle = self._find_closest_obstacle()
        distance_cost = dist_2_obstacle * cost_slope + cost_intercept \
            if dist_2_obstacle < self.config["distance_threshold"] else 0
        reward = - distance_cost \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [-self.config["max_obstacle_dist_cost"],
                           self.config["high_speed_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward
    
    def _find_closest_obstacle(self) -> float:
        """
        Uses the ego-vehicle global position and the global position of
        buffered lidar points to find the lidar point that is closest to the
        ego-vehicle.
        """
        # get the current ego-vehicle global position
        # find an algorithm that will efficiently tell you which lidar point 
        #   is closest.
        # return that distance.
        return NotImplemented
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        self._simulate(action)

        obs = self.lidar_buffer
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info    
    
    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                # the fastest observation frequency is the policy frequency
                if self.config["adaptive_observations"]:
                    self._adaptively_observe()
                else:
                    self._observe()

                self.action_type.act(action)
                
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1 # [in simulation steps, not action steps]
            
            if self.config["adaptive_observations"] and \
                    self.time % int(self.config["simulation_frequency"] // self.config["base_lidar_frequency"]) == 0:            
                self._observe()

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False
        
    def _observe() -> None:
        """
        Updates self.lidar_buffer with all lidar samples.
        """
        # Call AdaptiveLidarObservation.observe() so that all lidar samples
        #   are collected.
        # Update self.lidar_buffer with the new lidar data.
        # Increase self.total_collected_lidar_points.
        return NotImplemented
        
    def _adaptively_observe(self) -> None:
        """
        Updates self.lidar_buffer if a set of specific lidar points in the 
        buffer require update according to the discrete reactive strategy
        laid out in the paper.
        """
        # Assess current self.lidar_buffer to see which indices should be 
        #   updated. 
        # Call AdaptiveLidarObservation.observe(traces) where traces are the 
        #   specific indices that need to be updated.
        # Update self.lidar_buffer with the new indices, without overwriting
        #   indices that didn't need to be updated.
        # Increase self.total_collected_lidar_points.
        return NotImplemented
        
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
