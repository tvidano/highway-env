import warnings
from typing import List, Tuple, Optional, Callable
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
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

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "ContinuousAction",
                "dynamical": True
            },
            "simulation_frequency": 50,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "lanes_count": 3,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 20, # [s]
            "ego_spacing": 2,
            "vehicles_density": 2,
            "collision_avoided_reward": 1,
            "collision_imminent_reward": 1/(20*10),
            "collision_max_reward": 0.8,
            "collision_sensitivity": 1/40,
            "time_after_collision": 5, # [s]
            "offroad_terminal": True,
            "stopping_vehicles_count": 2,
            "intervention_distance": 10 # [m]
        })
        return config

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

        if not self._imminent_collision():
            action = np.array([0,0])
        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
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
        """Determines if a collision is about to happen."""
        front_vehicle,_ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        return False if self.vehicle.lane_distance_to(front_vehicle) \
                > self.config["intervention_distance"] else True

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster collision avoidance either by
        emergency braking or emergency steering.
        :param action: the last action performed
        :return: the corresponding reward
        """
        duration_reached = self.steps >= self.config["duration"]

        if duration_reached and self.vehicle.crashed:
            reward = 0
        elif duration_reached and not self.vehicle.crashed:
            reward = self.config["collision_avoided_reward"]
        elif not duration_reached and not self.vehicle.crashed and self._imminent_collision():
            reward = self.config["collision_imminent_reward"]
        elif not duration_reached and not self.vehicle.crashed:
            reward = 0
        elif not duration_reached and self.vehicle.crashed:
            relative_velocity = 0
            for collisions in self.vehicle.log:
                relative_velocity += collisions[1]
            damage = relative_velocity/self.vehicle.MAX_SPEED
            reward = -self.config["collision_sensitivity"]*damage \
                    + self.config["collision_max_reward"]
        else:
            warnings.warn("Something went wrong.")
            reward = -1
        reward = 0 if not self.vehicle.on_road else reward
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

utils.register_id_once('collision-v0','collision_env.envs:CollisionEnv')
