from highway_env.vehicle.dynamics import BicycleVehicle, CoupledDynamics
from os import replace
import warnings
from typing import List, Tuple, Optional, Callable
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split, relative_velocity
from highway_env.vehicle.controller import ControlledVehicle

Observation = np.ndarray
class CollisionEnv(HighwayEnv):
    """
    A highway driving environment with high probability of collisions.

    The vehicle is driving on a straight highway with several lanes, and is
    not allowed to act until an imminent collision is detected. It is 
    rewarded based on its ability to avoid that collision or mitigate the 
    damage of the collision.
    """
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.time_to_collision = np.inf
        self.active = 0 # State machine, 0 is inactive, 1 is active, 2 is transition
        self.time_since_avoidance = np.inf
        self.becomes_skynet = False  # change to true if self becomes Skynet

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "ContinuousAction",
                "vehicle_class": CoupledDynamics
            },
            "collision_avoided_reward": 1,
            "collision_imminent_reward": .00,
            "collision_max_reward": 0.3,
            "off_road_reward": 0.3,
            "collision_sensitivity": 1/40,
            "controlled_vehicles": 1,
            "duration": 20, # [s]
            "ego_spacing": 2,
            "initial_ego_speed": 20, # [m/s]
            "initial_lane_id": None,
            "lanes_count": 3,
            "look_ahead_distance": 50, # [m]
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "see_behind": False,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-45, 45],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted",
                "flatten": False,
                "observe_intentions": False,
            },
            "offroad_terminal": True,
            "policy_frequency": 15,  # [Hz]
            "road_friction": 1.0,  # Road-tire coefficient of friction (0,1]
            "simulation_frequency": 15,  # [Hz]
            "stopping_vehicles_count": 2,
            "time_after_collision": 0,  # [s] for capturing rear-end collisions
            "time_to_intervene": 5,  # [s]
            "vehicles_count": 40,
            "vehicles_density": 2,
            "control_time_after_avoid": 3,  # [s]
            "imminent_collision_distance": 7,  # within this distance is automatically imminent collisions, None for disabling this
            "sparse_reward": False,  # if true reward is ONLY given for avoidance.
        })
        return config

    def _create_vehicles(self) -> None:
        self.active = 0
        """Create some new random vehicles of a given type, and add them on the road."""
        if self.config["controlled_vehicles"] > 1:
            raise ValueError(f'{self.config["controlled_vehicles"]} controlled vehicles set, but CollisionEnv uses only 1.')
        
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        self.controlled_vehicles = []
        controlled_vehicle = self.action_type.vehicle_class.create_random(
            self.road,
            speed=self.config["initial_ego_speed"],
            lane_id=self.config["initial_lane_id"],
            spacing=self.config["ego_spacing"]
        )
        controlled_vehicle.mu = self.config["road_friction"]
        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)

        other_vehicles = []
        for _ in range(self.config["vehicles_count"]):
            other_vehicle = other_vehicles_type.create_random(self.road, \
                spacing = 1 / (self.config["vehicles_density"] * 2))
            other_vehicle.mu = self.config["road_friction"]
            self.road.vehicles.append(other_vehicle)
            other_vehicles.append(other_vehicle)

        ego_pos_x = controlled_vehicle.position[0]
        select_vehicles = self.road.np_random.choice(other_vehicles, \
            len(other_vehicles) // 2, replace=False)
        for select_vehicle in select_vehicles:
            dist_from_ego = select_vehicle.position[0] - ego_pos_x
            select_vehicle.position[0] = ego_pos_x - dist_from_ego
            

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the 
        road performs their default behaviour for several simulation timesteps 
        until the next decision making step.

        The ego-vehicle is not allowed to act until a collision is imminent.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        #state machine for controlling whether the car is 'active' or not
        GREEN = (50, 200, 0)
        ORANGE = (255, 150, 0)
        YELLOW = (200, 200, 0)
        if self.active == 0:
            self.controlled_vehicles[0].color = GREEN
            if self._imminent_collision():
                self.active = 1
            else:
                action = np.array([0, 0])
        if self.active == 1:
            self.controlled_vehicles[0].color = YELLOW
            if not self._imminent_collision():
                self.active = 2
                self.time_since_avoidance = self.time
        if self.active == 2:
            self.controlled_vehicles[0].color = ORANGE
            if (self.time - self.time_since_avoidance) > self.config["control_time_after_avoid"]:
                self.active = 0

        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        obs += 1 if self.active == 2 or self.active == 1 else 0
        obs += self.vehicle.lane_index[-1]
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            # Forward action to the vehicle
            if action is not None and not self.config["manual_control"]:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1/self.config["simulation_frequency"]

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

    def _imminent_collision(self) -> bool:
        """Returns True if a collision is about to happen."""
        front_vehicle,_ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        relative_distance = front_vehicle.position[0] - self.vehicle.position[0] if front_vehicle else np.inf
        if relative_distance > self.config["look_ahead_distance"]:
            return False
        relative_x_velocity = front_vehicle.velocity[0] - self.vehicle.velocity[0]
        self.time_to_collision = np.inf if relative_x_velocity >= 0 else (-relative_distance - self.vehicle.LENGTH)/ relative_x_velocity
        if self.config["imminent_collision_distance"]:
            return not self.time_to_collision > self.config["time_to_intervene"] or relative_distance < self.config["imminent_collision_distance"]
        else:
            return not self.time_to_collision > self.config["time_to_intervene"]

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster collision avoidance either by
        emergency braking or emergency steering.
        :param action: the last action performed
        :return: the corresponding reward
        """
        duration_reached = self.time >= self.config["duration"]

        if duration_reached and not self.vehicle.crashed:
            reward = self.config["collision_avoided_reward"]
        else:
            reward = 0
        if not self.config['sparse_reward']:
            if not duration_reached and not self.vehicle.crashed and self._imminent_collision():
                reward = max(reward, self.config["collision_imminent_reward"])
            if not duration_reached and self.vehicle.crashed:
                damage = 0
                for collisions in self.vehicle.log:
                    damage += collisions[1]
                t_reward = -self.config["collision_max_reward"] * damage / self.config["initial_ego_speed"] \
                         + self.config["collision_max_reward"]
                t_reward = t_reward if t_reward < self.config["collision_max_reward"] \
                    else self.config["collision_max_reward"]
                t_reward = 0 if t_reward < 0 else t_reward
                reward = max(reward, t_reward)
            if (self.config["offroad_terminal"] and not self.vehicle.on_road) or (duration_reached and not self.config["offroad_terminal"] and not self.vehicle.on_road):
                reward = max(reward, self.config["off_road_reward"])
            if self.becomes_skynet:
                reward = -999999
        return reward


    def _is_terminal(self) -> bool:
        """
        The episode is over if the time is out or after a set time after
        a collision.
        """
        if self.vehicle.crashed:
            end_time = self.time + self.config["time_after_collision"]
            while self.time < end_time:
                self._simulate(None)
        
        return self.vehicle.crashed or \
            self.time >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "action": action,
            "crashed": self.vehicle.crashed,
            "slip_values": self.vehicle.slip_values,
            "speed": np.array([self.vehicle.longitudinal_velocity, 
                               self.vehicle.lateral_velocity, 
                               self.vehicle.front_wheel_angular_velocity]),
            "time": self.time,
            "tire_forces": self.vehicle.tire_forces,
            "ttc": self.time_to_collision,
            "imminent": self._imminent_collision(),
            "onroad": self.vehicle.on_road,
            "active": self.active,
        }
        return info

utils.register_id_once('collision-v0','collision_env.envs:CollisionEnv')
