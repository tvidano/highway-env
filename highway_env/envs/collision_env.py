from os import replace
import warnings
from typing import List, Tuple, Optional, Callable
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.observation import ObservationType
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.vehicle.objects import Obstacle
from highway_env.utils import near_split, relative_velocity
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import RoadObject, Obstacle
from highway_env.vehicle.dynamics import BicycleVehicle, CoupledDynamics

class CollisionEnv(HighwayEnv):
    """
    A highway driving environment with high probability of collisions.

    The vehicle is driving on a straight highway with several lanes, and is
    not allowed to act until an imminent collision is detected. It is 
    rewarded based on its ability to avoid that collision or mitigate the 
    damage of the collision.
    """
    ROAD_LENGTH = 10000

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.time_to_collision = np.inf
        self.active = 0 # State machine, 0 is inactive, 1 is active, 2 is transition
        self.time_since_avoidance = np.inf
        self.becomes_skynet = False  # change to true if self becomes Skynet
        self.did_run = False

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "ContinuousAction",
                "vehicle_class": CoupledDynamics
            },
            "collision_avoided_reward": 100,
            "collision_imminent_reward": .05,
            "collision_max_reward": 0.3,
            "collision_penalty": 1,
            "off_road_reward": 0.3,
            "off_road_penalty": 1,
            "collision_sensitivity": 1/40,
            "controlled_vehicles": 1,
            "duration": 15, # [s]
            "ego_spacing": 1.5,
            "initial_ego_speed": 20, # [m/s]
            "initial_lane_id": None,
            "lanes_count": 5,
            "look_ahead_distance": 50, # [m]
            "observation": {
                "type": "ADSObservation", # "Kinematics" "LidarObservation" "ADSObservation"
                # "vehicles_count": 15,
                # "see_behind": True,
                # "features": ["presence", "x", "y", "vx", "vy", "sin_h", "cos_h"],
                # # "features_range": {
                # #     "x": [-100, 100],
                # #     "y": [-100, 100],
                # #     "vx": [-45, 45],
                # #     "vy": [-20, 20]
                # # },
                # "absolute": False,
                # "order": "sorted",
                # "flatten": False,
                # "observe_intentions": False,
            },
            "offroad_terminal": False,
            "policy_frequency": 30,  # [Hz]
            "road_friction": 1.0,  # Road-tire coefficient of friction (0,1]
            "road_barriers": False,  # adds obstacles at the outside lane borders
            "lane_width": 6,  # None will set this to default of 4
            "simulation_frequency": 30,  # [Hz]
            "stopping_vehicles_count": 5,
            "time_after_collision": 0,  # [s] for capturing rear-end collisions
            "time_to_intervene": 8,
            "vehicles_count": 40,
            "reset_empty_lane": True, # moves a car in front if there isn't one
            "vehicles_density": 1.75,
            "control_time_after_avoid": 8,  # [s]
            "imminent_collision_distance": 10,  # [m] within this distance is automatically imminent collisions, None for disabling this
            "reward_type": "penalty_dense", # dense = reward is given on linear scale and for avoiding a collision.
                                     # sparse = reward is given ONLY for avoidance.
                                     # penalty = reward given for avoiding a collision, penalty given for collision
                                     # penalty_dense = reward for avoiding collision, penalize based on energy of crash and offroad
                                     # stop = linear reward to encourage vehicle to learn to come to a stop
        })
        return config

    def _create_road(self) -> None:

        """Create a road composed of straight adjacent lanes, and barrier objects at the edges"""
        width = self.config["lane_width"] if self.config["lane_width"] else StraightLane.DEFAULT_WIDTH
        road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30,width=width),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
        if self.config['road_barriers']:
            road.objects.append(Obstacle(road, road.network.get_lane(('0', '1', 0)).position(0, 0), 10000, 0.5))
            road.objects.append(Obstacle(road, road.network.get_lane(('0', '1', self.config["lanes_count"] - 1)).position(0, 0), 10000, 0.5))

        self.road = road


    def _create_vehicles(self) -> None:
        self.time = 0
        self.active = 0
        self.did_run = False
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

        if self.config["reset_empty_lane"]:
            front_vehicle,_ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            if not front_vehicle:
                select_vehicles = self.road.np_random.choice(other_vehicles, \
                                                             1, replace=False)
                dist_from_ego = abs(select_vehicles[0].position[0] - ego_pos_x)
                select_vehicles[0].position[0] = ego_pos_x + dist_from_ego
                select_vehicles[0].position[1] = controlled_vehicle.position[1]
                select_vehicles[0].velocity[0] = select_vehicles[0].velocity[0] - abs(select_vehicles[0].velocity[0] - controlled_vehicle.velocity[0])
            else:
                front_vehicle.velocity[0] = controlled_vehicle.velocity[0] - abs(front_vehicle.velocity[0] - controlled_vehicle.velocity[0])



    def step(self, action: Action) -> Tuple[ObservationType, float, bool, dict]:
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

        while self._update_state() == 0:
            # spin simulation
            self.steps += 1
            self._simulate(np.array([0, 0]))

            if self._is_terminal():
                if self.did_run:
                    obs = self.observation_type.observe()
                    reward = self._reward(action)
                    terminal = self._is_terminal()
                    info = self._info(obs, action)

                    return obs, reward, terminal, info
                else:
                    self.reset()

        else:
            # if self.did_run = False:
            # self.time = 0 <-- this could start the time at when the active starts
            self.did_run = True

        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()

        #obs += 1 if self.active == 2 or self.active == 1 else 0
        #obs += self.vehicle.position[0] # turns out that Kinematics gives this
        #obs += self.vehicle.position[1]
        #obs += self.vehicle.lane_index[-1]

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
        relative_x_velocity = front_vehicle.velocity[0] - self.vehicle.longitudinal_velocity
        self.time_to_collision = np.inf if relative_x_velocity >= 0 else (-relative_distance - self.vehicle.LENGTH)/ relative_x_velocity
        if self.config["imminent_collision_distance"]:
            return not self.time_to_collision > self.config["time_to_intervene"] or relative_distance < self.config["imminent_collision_distance"]
        else:
            return not self.time_to_collision > self.config["time_to_intervene"]

    def _update_state(self):
        #state machine for controlling whether the car is 'active' or not
        GREEN = (50, 200, 0)
        ORANGE = (255, 150, 0)
        YELLOW = (200, 200, 0)
        RED = (255, 100, 100)
        if self.active == 0:
            self.controlled_vehicles[0].color = GREEN
            if self._imminent_collision():
                self.active = 1
        if self.active == 1:
            self.controlled_vehicles[0].color = YELLOW
            if not self._imminent_collision():
                self.active = 2
                self.time_since_avoidance = self.time
            if self.vehicle.crashed:
                self.controlled_vehicles[0].color = RED
                self.active = 0
        if self.active == 2:
            self.controlled_vehicles[0].color = ORANGE
            if (self.time - self.time_since_avoidance) > self.config["control_time_after_avoid"]:
                self.active = 0
            if self.vehicle.crashed:
                self.controlled_vehicles[0].color = RED
                self.active = 0
        return self.active

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster collision avoidance either by
        emergency braking or emergency steering.
        :param action: the last action performed
        :return: the corresponding reward
        """
        reward = 0
        avoidance_rew = imminent_collision_rew = damage_mitigation_rew = survival_rew \
            = offroad_rew = velocity_rew = collision_pen = offroad_pen = damage_pen = variant = False
        if self.config['reward_type'] == 'sparse':
            avoidance_rew = True
        elif self.config['reward_type'] == 'dense':
            imminent_collision_rew = damage_mitigation_rew = survival_rew = True
        elif self.config['reward_type'] == 'penalty':
            avoidance_rew = collision_pen = offroad_pen = True
        elif self.config['reward_type'] == 'penalty_dense':
            avoidance_rew = damage_pen = True
        elif self.config['reward_type'] == 'stop':
            velocity_rew = True
        elif self.config['reward_type'] == 'variant':
            variant = True
        else:
            raise(NotImplementedError, f'{self.config["reward_type"]} reward type not implemented or misstyped.')
        
        duration_reached = self.time >= self.config["duration"]
        if avoidance_rew:
            reward += self.config["collision_avoided_reward"] if duration_reached and not self.vehicle.crashed and self.vehicle.on_road else 0
        if imminent_collision_rew:
            reward += self.config["collision_imminent_reward"] if self.active != 0 else 0
        if damage_mitigation_rew:
            if not duration_reached and self.vehicle.crashed:
                damage = 0
                for collisions in self.vehicle.log:
                    damage += collisions[1]
                mitigation_reward = self.config["collision_max_reward"] \
                           - self.config["collision_max_reward"] * damage / self.config["initial_ego_speed"]
                mitigation_reward = np.clip(mitigation_reward, 0, self.config["collision_max_reward"])
                reward += mitigation_reward
        if survival_rew:
            reward += self.config["collision_avoided_reward"] if duration_reached and not self.vehicle.crashed else 0
        if offroad_rew:
            if not self.config["offroad_terminal"]:
                print('Using a penalty for going offroad, but not ending episode when going offroad. Is this intended?')
            reward += self.config["off_road_reward"] if not self.vehicle.on_road else 0
        if velocity_rew and self._is_terminal():
            reward += -self.vehicle.longitudinal_velocity/self.config["initial_ego_speed"] + 1

        if collision_pen:
            reward -= self.config["collision_penalty"] if self.vehicle.crashed else 0
        if offroad_pen:
            if not self.config["offroad_terminal"]:
                print('Using a penalty for going offroad, but not ending episode when going offroad. Is this intended?')
            reward -= self.config["off_road_penalty"] if not self.vehicle.on_road else 0
        if damage_pen:
            damage = 0
            max_damage = 1/2*self.vehicle.mass*self.config["initial_ego_speed"]**2
            if not self.vehicle.on_road:
                damage = 1/2*self.vehicle.mass*np.linalg.norm(self.vehicle.state[[2,3], 0])**2
                damage = damage / max_damage
            if self.vehicle.crashed:
                for collisions in self.vehicle.log:
                    v1 = collisions[0].speed
                    v2 = np.linalg.norm(self.vehicle.state[[2,3], 0])
                    Ei = 1/2*self.vehicle.mass*(v1**2 + v2**2)
                    Ef = self.vehicle.mass*(v1/2 + v2/2)**2
                    damage += (Ei - Ef) / max_damage
            reward -= damage
        if variant:
            reward = 0
            reward += 0.25 * self.time
            if duration_reached and not self.vehicle.crashed:
                reward += 500
            if self.active == 2:
                reward += 10 - 2*abs(self.vehicle.lateral_velocity)
            if not self.vehicle.on_road:
                reward = -min(self.road.network.get_lane(('0', '1', 0)).lat_distance(self.vehicle.position),\
                                    self.road.network.get_lane(('0', '1', self.config["lanes_count"] - 1)).lat_distance(self.vehicle.position))
            if self.vehicle.crashed:
                reward = -500
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

    def _info(self, obs: ObservationType, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "action": action,
            "actuators": self.vehicle.actuators,
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

utils.register_id_once('collision-v0','highway_env.envs:CollisionEnv')
