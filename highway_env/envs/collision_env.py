import warnings
from typing import Tuple, Optional
import numpy as np

from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.common.observation import ObservationType
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.vehicle.dynamics import CoupledDynamics

class CollisionEnv(HighwayEnv):
    """
    A highway driving environment with high probability of collisions.
    The vehicle is driving on a straight highway with several lanes, and is
    not allowed to act until an imminent collision is detected. It is
    rewarded based on its ability to avoid that collision or mitigate the
    damage of the collision.
    """
    ROAD_LENGTH = 10000

    COLORS = [(50, 200, 0), (255, 150, 0), (200, 200, 0), (255, 100, 100)]  # Green, Yellow, Orange, and Red

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "ContinuousAction",
                "vehicle_class": "CoupledDynamics",
            },
            "observation": {
                "type": "LidarKinematicObservation",  # "Kinematics" "LidarObservation" "LidarKinematicObservation"
                'ego_position': True,
                'ego_velocity': True,
            },
            "collision_avoided_reward": 100,  # reward received for staying alive for the full time
            "time_alive_reward": 0.25,  # multiplied by self.time for time alive reward, 0 to disable
            "lateral_velocity_reward": -2,  # multiplied by lateral velocity after avoiding collision, 0 to disable, negative to penalize
            "crashed_reward": -100,  # reward given if vehicle crashed, negative to penalize
            "offroad_reward": -100,  # reward given if vehicle offroad, negative to penalize
            "duration": 15,  # [s]
            "policy_frequency": 15,  # [Hz]
            "simulation_frequency": 30,  # [Hz]
            "vehicles_count": 30,
            "ego_spacing": 1.5,
            "vehicles_density": 1.75,
            "initial_ego_speed": 20,  # [m/s]
            "initial_lane_id": None,
            "stopping_vehicles_count": 5,
            "lanes_count": 5,
            "lane_width": 6,  # None will set this to default of 4
            "road_friction": 1.0,  # Road-tire coefficient of friction (0,1]
            "reset_empty_lane": True,  # moves a car in front if there isn't one
            "offroad_terminal": False,
            "time_after_collision": 0,  # [s] for capturing rear-end collisions
            "time_to_intervene": 8,
            "look_ahead_distance": 50,  # [m]
            "control_time_after_avoid": 8,  # [s]
            "imminent_collision_distance": 10,  # [m] within this distance is automatically imminent collisions, None for disabling this
        })
        return config

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes"""
        width = self.config["lane_width"] if self.config["lane_width"] else StraightLane.DEFAULT_WIDTH
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30, width=width),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        self.time = 0
        self.active = 0   # State machine, 0 is inactive, 1 is active, 2 is transition
        self.did_run = False
        self.time_since_avoidance = np.inf
        self.time_to_collision = np.inf

        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        self.controlled_vehicles = []
        controlled_vehicle = self.action_type.vehicle_class.create_random(
            self.road,
            speed=self.config["initial_ego_speed"],
            lane_id=self.config["initial_lane_id"],
            spacing=self.config["ego_spacing"],
            mu=self.config["road_friction"]
        )
        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)

        other_vehicles = []
        for _ in range(self.config["vehicles_count"]):
            other_vehicle = other_vehicles_type.create_random(self.road, \
                        spacing=(1 / (self.config["vehicles_density"] * 2)),
                        mu=self.config["road_friction"],
                        enable_mu=True,)
            self.road.vehicles.append(other_vehicle)
            other_vehicles.append(other_vehicle)

        ego_pos_x = controlled_vehicle.position[0]
        select_vehicles = self.road.np_random.choice(other_vehicles, len(other_vehicles) // 3, replace=False)
        for select_vehicle in select_vehicles:
            dist_from_ego = select_vehicle.position[0] - ego_pos_x
            select_vehicle.position[0] = ego_pos_x - dist_from_ego

        if self.config["reset_empty_lane"]:
            front_vehicle,_ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            if not front_vehicle:
                select_vehicles = self.road.np_random.choice(other_vehicles, 1, replace=False)
                dist_from_ego = abs(select_vehicles[0].position[0] - ego_pos_x)
                select_vehicles[0].position[0] = ego_pos_x + dist_from_ego
                select_vehicles[0].position[1] = controlled_vehicle.position[1]
                select_vehicles[0].velocity[0] = select_vehicles[0].velocity[0] - abs(select_vehicles[0].velocity[0] - controlled_vehicle.velocity[0])
            else:
                front_vehicle.velocity[0] = controlled_vehicle.velocity[0] - abs(front_vehicle.velocity[0] - controlled_vehicle.velocity[0])

        """Choose stopping vehicles: randomly choose non-controlled vehicles to stop abruptly."""
        stopping_vehicles_count = self.config["stopping_vehicles_count"]
        try:
            chosen_vehicles = self.road.np_random.choice(other_vehicles, stopping_vehicles_count, replace=False)
        except ValueError:
            warnings.warn(f'Chose {stopping_vehicles_count} vehicles to stop abruptly when only '
                          f'{len(other_vehicles)} vehicles are uncontrolled. '
                          'Selecting all uncontrolled vehicles...')
            chosen_vehicles = other_vehicles
        for chosen_vehicle in chosen_vehicles:
            chosen_vehicle.target_speed = 0

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

        self._spin(action)
        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info

    def _spin(self, action):
        """Run simulation while vehicle is inactive"""
        while self._update_state() == 0:
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
            self.did_run = True

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            # Forward action to the vehicle
            if action is not None and not self.config["manual_control"]:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += (1/self.config["simulation_frequency"])

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
        self.time_to_collision = np.inf if relative_x_velocity >= 0 else (-relative_distance - self.vehicle.LENGTH)/relative_x_velocity
        return not self.time_to_collision > self.config["time_to_intervene"] or \
            (self.config["imminent_collision_distance"] and relative_distance < self.config["imminent_collision_distance"])

    def _update_state(self):
        """
        State machine for controlling whether the car is 'active' or not.
        If a collision is not imminent, state is 0 and the car takes the default action.
        If a collision is imminent, state is 1 and the agent has control.
        For a specified time after avoidance, state is 2 and the agent still has control to right itself before returning to state 0.
        """
        if self.active == 0:
            if self._imminent_collision():
                self.active = 1
        if self.active == 1:
            if not self._imminent_collision():
                self.active = 2
                self.time_since_avoidance = self.time
        if self.active == 2:
            if (self.time - self.time_since_avoidance) > self.config["control_time_after_avoid"]:
                self.active = 0
        if self.vehicle.crashed:
            self.active = 4
        self.controlled_vehicles[0].color = self.COLORS[self.active]

        return self.active

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster collision avoidance either by
        emergency braking or emergency steering.
        :param action: the last action performed
        :return: the corresponding reward
        """
        duration_reached = self.time >= self.config["duration"]
        reward = self.config["collision_avoided_reward"] if duration_reached and not self.vehicle.crashed and self.vehicle.on_road else 0
        reward += self.config["time_alive_reward"] * self.time
        reward += self.config["lateral_velocity_reward"] * abs(self.vehicle.lateral_velocity) if self.active == 2 else 0
        reward = self.config["offroad_reward"] if not self.on_road else reward
        reward = self.config["crashed_reward"] if self.vehicle.crashed else reward
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


utils.register_id_once(
    id='collision-v0',
    entry_point='highway_env.envs:CollisionEnv',
)
