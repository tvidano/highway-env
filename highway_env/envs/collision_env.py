from os import replace
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
# TODO{tvidano}: add tire models to bicycle car or fix dynamics model
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
            "initial_ego_speed": 30,
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
            "collision_imminent_reward": .05,
            "collision_max_reward": 0.8,
            "collision_sensitivity": 1/40,
            "time_after_collision": 5, # [s]
            "offroad_terminal": True,
            "stopping_vehicles_count": 2,
            "intervention_distance": 10 # [m]
        })
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=self.config["initial_ego_speed"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            other_vehicles = []
            for _ in range(others):
                other_vehicle = other_vehicles_type.create_random(self.road, \
                    spacing = 1 / (self.config["vehicles_density"] * 2))
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
        follow_distance = self.vehicle.lane_distance_to(front_vehicle) if front_vehicle else np.inf
        return False if follow_distance > self.config["intervention_distance"] else True

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster collision avoidance either by
        emergency braking or emergency steering.
        :param action: the last action performed
        :return: the corresponding reward
        """
        duration_reached = self.time >= self.config["duration"]

        if duration_reached and self.vehicle.crashed:
            reward = 0
        elif duration_reached and not self.vehicle.crashed:
            reward = self.config["collision_avoided_reward"]
        elif not duration_reached and not self.vehicle.crashed and self._imminent_collision():
            reward = self.config["collision_imminent_reward"]
        elif not duration_reached and not self.vehicle.crashed:
            reward = 0
        elif not duration_reached and self.vehicle.crashed:
            damage = 0
            for collisions in self.vehicle.log:
                damage += collisions[1]
            reward = -self.config["collision_max_reward"]*damage/self.config["initial_ego_speed"] \
                    + self.config["collision_max_reward"]
            reward = reward if reward < self.config["collision_max_reward"]\
                    else self.config["collision_max_reward"]
            reward = 0 if reward < 0 else reward
        else:
            warnings.warn("Something went wrong.")
            reward = 0
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
