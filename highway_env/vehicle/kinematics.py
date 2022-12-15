from typing import List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import copy
from collections import deque

from highway_env import utils
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.objects import MirroredObject, RoadObject, Obstacle, Landmark
from highway_env.utils import Vector

if TYPE_CHECKING:
    from highway_env.road.lane import AbstractLane
class Vehicle(RoadObject):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -40.
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed)
        self.prediction_type = predition_type
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)

    @classmethod
    def create_random(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      spacing: float = 1,
                      add_backwards: bool = False) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(
            list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(
            list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(
            len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7*lane.speed_limit, 1.0*lane.speed_limit)
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0],
                    Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12+1.0*speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        if hasattr(vehicle, 'color'):
            v.color = vehicle.color
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the
        steering wheel dynamics. If the vehicle is crashed, the actions are
        overridden with erratic steering and braking until complete stop. The 
        vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(
                self.action['acceleration'],
                1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < self.MIN_SPEED:
            self.action['acceleration'] = max(
                self.action['acceleration'],
                1.0 * (self.MIN_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(
                self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(self, times: np.ndarray) \
            -> Tuple[List[np.ndarray], List[float]]:
        if self.prediction_type == 'zero_steering':
            action = {'acceleration': 0.0, 'steering': 0.0}
        elif self.prediction_type == 'constant_steering':
            action = {'acceleration': 0.0, 'steering': self.action['steering']}
        else:
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

    @property
    def velocity(self) -> np.ndarray:
        # TODO: slip angle beta should be used here
        return self.speed * self.direction

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index \
                if last_lane_index[-1] is not None else (
                    *last_lane_index[:-1], 0)
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / \
                np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang])
        else:
            return np.zeros((3,))

    def to_dict(self, origin_vehicle: "Vehicle" = None,
                observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading,
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1],
            'long_off': self.lane_offset[0],
            'lat_off': self.lane_offset[1],
            'ang_off': self.lane_offset[2],
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__,
                                   id(self) % 1000,
                                   self.position)

    def __repr__(self):
        return self.__str__()

class CyclicVehicle(Vehicle):

    """A kinematic vehicle that when past the road edge, will restart at the
    road origin."""

    SPEED_GAINS = [0.9, 1.0]
    """Range of speed limit scaling when creating random vehicle."""

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed, predition_type)
        # Create imaginary vehicles past the edges of the road to support
        # different kinds of observations.
        # Assume all lanes are equal length and the length of the lane is the
        # edge of the road.
        self.road_edge = self.road.network.get_lane(self.lane_index).length
        imaginary_position = np.array([self.position[0] + self.road_edge, 
                                       self.position[1]])
        self.front_mirrored_vehicle = MirroredObject(
            self.road, imaginary_position, length=self.LENGTH, 
            width=self.WIDTH)
        self.road.objects.append(self.front_mirrored_vehicle)
        imaginary_position = np.array([self.position[0] - self.road_edge,
                                       self.position[1]])
        self.rear_mirrored_vehicle = MirroredObject(
            self.road, imaginary_position, length=self.LENGTH, 
            width=self.WIDTH)
        self.road.objects.append(self.rear_mirrored_vehicle)
        self.is_front_most_vehicle = False
        self.is_rear_most_vehicle = False
        self._update_if_front_rear_edge()

    def _update_if_front_rear_edge(self):
        """Define bools if front or rear edge vehicle."""
        veh_pos_in_lane = [v.position[0] for v in self.road.vehicles \
            if v.lane_index == self.lane_index]
        if len(veh_pos_in_lane) == 0:
            self.is_front_most_vehicle = True
            self.is_rear_most_vehicle = True
            return
        self.is_rear_most_vehicle = self.position[0] <= min(veh_pos_in_lane)
        self.is_front_most_vehicle = self.position[0] >= max(veh_pos_in_lane)

    # Override RoadObject's lane_distance_to
    def lane_distance_to(self, other: 'CyclicVehicle',
            lane: 'AbstractLane' = None, return_both: bool=False) -> float:
        """
        Compute the signed distance to another object along lane as per the
        cyclic road type. Assuming linear road.

        :param other: the other object
        :param lane: a lane
        :return: the distance to the other other [m]
        """
        if other is None:
            return np.nan
        if lane is None:
            lane = self.lane
        # Compute the distance.
        distance = other.position[0] - self.position[0]
        # Compute the distance using the cyclic road.
        if distance >= 0:
            cycle_distance = distance - self.road_edge
        else:
            cycle_distance = self.road_edge + distance
        if return_both:
            return distance, cycle_distance
        # Return the distance with the smallest magnitude.
        else:
            return distance if abs(distance) < abs(cycle_distance) \
                else cycle_distance

        # front_most_vehicle, rear_most_vehicle = \
        #     self._get_edge_vehicles(self.lane_index)
        # if (self is front_most_vehicle and other is rear_most_vehicle) or \
        #         (self is rear_most_vehicle and other is front_most_vehicle):
        #     dist_to_edge = self._get_distance_to_road_edge()
        #     other_dist_to_edge = other._get_distance_to_road_edge()
        #     assert dist_to_edge > 0
        #     assert other_dist_to_edge > 0
        #     return dist_to_edge + other_dist_to_edge
        # else:
        #     return other.position[0] - self.position[0]

    def _get_edge_vehicles(self, lane_index: LaneIndex) \
            -> Tuple['CyclicVehicle']:
        """Gets the collision-free front and rear-most vehicles in a lane."""
        front_most_vehicle = None
        rear_most_vehicle = None
        for v in self.road.vehicles:
            v._update_if_front_rear_edge()
            if v.lane_index == lane_index:
                if v.is_front_most_vehicle:
                    front_most_vehicle = v
                if v.is_rear_most_vehicle:
                    rear_most_vehicle = v
            if front_most_vehicle is not None \
                    and rear_most_vehicle is not None:
                break
        return (front_most_vehicle, rear_most_vehicle)

    def _get_distance_to_road_edge(self) -> float:
        """Computes the signed distance to road's edge in the cyclic road."""
        return self.road_edge - self.position[0]

    # Override Vehicle's step
    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the
        steering wheel dynamics. If the vehicle is crashed, the actions are
        overridden with erratic steering and braking until complete stop. The 
        vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
            self.road.vehicles.remove(self)
            self.road.objects.remove(self.front_mirrored_vehicle)
            self.road.objects.remove(self.rear_mirrored_vehicle)
            collision_landmark = Landmark(self.road, self.position)
            self.road.objects.append(collision_landmark)
        # Reset position to the distance from the road edge if past road edge.
        if self.position[0] >= self.road_edge:
            self.position[0] = self._get_distance_to_road_edge()
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        # Update the position of the mirrored vehicles.
        self.front_mirrored_vehicle.position[0] = self.position[0] + self.road_edge
        self.rear_mirrored_vehicle.position[0] = self.position[0] - self.road_edge
        self._update_if_front_rear_edge()
        self.on_state_update()

    # Override Vehicle's create_random
    @classmethod
    def create_random(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      spacing: float = 1) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position
        is chosen behind the last vehicle in the road with density based on the
        number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: the number of seconds between ego car and front car.
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(
            list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(
            list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(
            len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    cls.SPEED_GAINS[0]*lane.speed_limit, 
                    cls.SPEED_GAINS[1]*lane.speed_limit)
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0],
                    Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        # Consider spacing argument as the number of seconds of travel between
        # front edge of ego vehicle and rear edge of forward car.
        offset = spacing * speed + cls.LENGTH / 2
        in_lane_x0 = [v.position[0] + v.LENGTH / 2 for v in road.vehicles \
            if v.lane_index[2] == _id]
        # Position of the front edge of the forward most vehicle.
        x0 = np.max(in_lane_x0) if len(in_lane_x0) > 0 else 0
        x0 += offset * road.np_random.uniform(0.9, 1.0)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v
