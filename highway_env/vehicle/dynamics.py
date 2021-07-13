from typing import Callable, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt

from highway_env.road.road import Road
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.tires import AbstractTire, ConstantPacTire

class BicycleVehicle(Vehicle):
    """
    A dynamical bicycle model, with tire friction and slipping.
    
    See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)
    """
    MASS: float = 1  # [kg]
    LENGTH_A: float = Vehicle.LENGTH / 2  # [m]
    LENGTH_B: float = Vehicle.LENGTH / 2  # [m]
    INERTIA_Z: float = 1/12 * MASS * (Vehicle.LENGTH ** 2 + Vehicle.WIDTH ** 2)  # [kg.m2]
    FRICTION_FRONT: float = 15.0 * MASS  # [N]
    FRICTION_REAR: float = 15.0 * MASS  # [N]

    MAX_ANGULAR_SPEED: float = 2 * np.pi  # [rad/s]
    MAX_SPEED: float = 45 # [m/s]

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

    @property
    def state(self) -> np.ndarray:
        return np.array([[self.position[0]],
                         [self.position[1]],
                         [self.heading],
                         [self.speed],
                         [self.lateral_speed],
                         [self.yaw_rate]])

    @property
    def derivative(self) -> np.ndarray:
        """
        See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)

        :return: the state derivative
        """
        delta_f = self.action["steering"]
        delta_r = 0
        theta_vf = np.arctan2(self.lateral_speed + self.LENGTH_A * self.yaw_rate, self.speed)  # (2.27)
        theta_vr = np.arctan2(self.lateral_speed - self.LENGTH_B * self.yaw_rate, self.speed)  # (2.28)
        f_yf = 2*self.FRICTION_FRONT * (delta_f - theta_vf)  # (2.25)
        f_yr = 2*self.FRICTION_REAR * (delta_r - theta_vr)  # (2.26)
        if abs(self.speed) < 1:  # Low speed dynamics: damping of lateral speed and yaw rate
            f_yf = - self.MASS * self.lateral_speed - self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
            f_yr = - self.MASS * self.lateral_speed + self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
        d_lateral_speed = 1/self.MASS * (f_yf + f_yr) - self.yaw_rate * self.speed  # (2.21)
        d_yaw_rate = 1/self.INERTIA_Z * (self.LENGTH_A * f_yf - self.LENGTH_B * f_yr)  # (2.22)
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]],
                         [speed[1]],
                         [self.yaw_rate],
                         [self.action['acceleration']],
                         [d_lateral_speed],
                         [d_yaw_rate]])

    @property
    def derivative_linear(self) -> np.ndarray:
        """
        Linearized lateral dynamics.
            
        This model is based on the following assumptions:
        - the vehicle is moving with a constant longitudinal speed
        - the steering input to front tires and the corresponding slip angles are small
        
        See https://pdfs.semanticscholar.org/bb9c/d2892e9327ec1ee647c30c320f2089b290c1.pdf, Chapter 3.
        """
        x = np.array([[self.lateral_speed], [self.yaw_rate]])
        u = np.array([[self.action['steering']]])
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]], [speed[1]], [self.yaw_rate], [self.action['acceleration']], dx[0], dx[1]])

    def step(self, dt: float) -> None:
        self.clip_actions()
        derivative = self.derivative
        self.position += derivative[0:2, 0] * dt
        self.heading += self.yaw_rate * dt
        self.speed += self.action['acceleration'] * dt
        self.lateral_speed += derivative[4, 0] * dt
        self.yaw_rate += derivative[5, 0] * dt

        self.on_state_update()

    def clip_actions(self) -> None:
        super().clip_actions()
        # Required because of the linearisation
        self.action["steering"] = np.clip(self.action["steering"], -np.pi/2, np.pi/2)
        self.yaw_rate = np.clip(self.yaw_rate, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

    def lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A0, phi, B such that dx = (A0 + theta^T phi)x + B u
        """
        B = np.array([
            [2*self.FRICTION_FRONT / self.MASS],
            [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z]
        ])

        speed_body_x = self.speed
        A0 = np.array([
            [0, -speed_body_x],
            [0, 0]
        ])

        if abs(speed_body_x) < 1:
            return A0, np.zeros((2, 2, 2)), B*0

        phi = np.array([
            [
                [-2 / (self.MASS*speed_body_x), -2*self.LENGTH_A / (self.MASS*speed_body_x)],
                [-2*self.LENGTH_A / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_A**2 / (self.INERTIA_Z*speed_body_x)]
            ], [
                [-2 / (self.MASS*speed_body_x), 2*self.LENGTH_B / (self.MASS*speed_body_x)],
                [2*self.LENGTH_B / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_B**2 / (self.INERTIA_Z*speed_body_x)]
            ],
        ])
        return A0, phi, B

    def lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([self.FRICTION_FRONT, self.FRICTION_REAR])
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

    def full_lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, phi, B
        """
        A_lat, phi_lat, B_lat = self.lateral_lpv_structure()

        speed_body_x = self.speed
        A_top = np.array([
            [0, speed_body_x, 1, 0],
            [0, 0, 0, 1]
        ])
        A0 = np.concatenate((A_top, np.concatenate((np.zeros((2, 2)), A_lat), axis=1)))
        phi = np.array([np.concatenate((np.zeros((2, 4)), np.concatenate((np.zeros((2, 2)), phi_i), axis=1)))
                        for phi_i in phi_lat])
        B = np.concatenate((np.zeros((2, 1)), B_lat))
        return A0, phi, B

    def full_lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.full_lateral_lpv_structure()
        self.theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B


class CoupledDynamics(Vehicle):
    """Non-linear single track vehicle model for lateral dynamics and single wheel model for longitudinal dynamics."""

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0, tire: AbstractTire = ConstantPacTire) -> None:
        """
        :param road: the road instance where the object is placed in
        :param position: global cartesian position of object in the surface
        :param heading: the angle from positive direction of global horizontal axis
        :param speed: global cartesian speed of object in the surface
        :param tire: tire object
        """
        super().__init__(road, position, heading, speed)
        self.mass = 12000/9.81 # kg
        self.inertia_z = 2000 # kg-m^2
        self.length = 4.5  # Meters
        self.wheel_radius = 0.33 # Meters
        self.wheel_inertia = 0.47 # kg-m^2
        self.max_engine_torque = 250 # N-m
        self.max_brake_torque = 400 # N-m
        self.cg_a = 2 # Location of center of mass from front of the vehicle, Meters
        self.cg_b = self.length - self.cg_a
        self.Fz_front = 9.81/2*self.mass*self.cg_b/self.length
        self.Fz_rear = 9.81/2*self.mass*self.cg_a/self.length
        self.mu = 1.0
        self.front_tire = tire(init_state=np.array([0,0,self.Fz_front,self.mu]))
        self.rear_tire = tire(init_state=np.array([0,0,self.Fz_rear,self.mu]))

        self.longitudinal_velocity = self.speed # m/s
        self.lateral_velocity = 0 # m/s
        self.yaw_rate = 0 # rad/s
        self.front_wheel_angular_velocity = self.rear_wheel_angular_velocity = self.longitudinal_velocity/self.wheel_radius
        self.tire_forces = np.array([self.front_tire.get_forces(), self.rear_tire.get_forces()])
        self.is_braking = False
    
    @property
    def state(self) -> np.ndarray:
        return np.array([[self.position[0]],
                         [self.position[1]],
                         [self.longitudinal_velocity],
                         [self.lateral_velocity],
                         [self.yaw_rate],
                         [self.front_wheel_angular_velocity],
                         [self.rear_wheel_angular_velocity]])

    def step(self, dt: float) -> None:
        self.clip_actions()
        if self.action["acceleration"] >= 0:
            self.is_braking = False
        else:
            self.is_braking = True
        n = 10
        for _ in range(n):
            self.longitudinal_velocity, self.front_wheel_angular_velocity =  \
                self.RK4(self.long_dynamics, self.state[[2,5], 0], dt/n)
            if self.longitudinal_velocity < 1:
                self.longitudinal_velocity = 0
            if self.front_wheel_angular_velocity < 0:
                self.front_wheel_angular_velocity = 0
            self.lateral_velocity, self.yaw_rate = self.RK4(self.lateral_dynamics, self.state[[3,4], 0], dt/n)
        self.heading += self.yaw_rate*dt
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        velocity = R @ np.array([self.longitudinal_velocity, self.lateral_velocity])
        self.position[0] += velocity[0]*dt
        self.position[1] += velocity[1]*dt

        self.front_tire.state = np.array([self.compute_long_slip(self.state[[2,5],0]),
                                          self.compute_lat_slip(*self.state[[2,3,4],0])[0],
                                          self.Fz_front,
                                          self.mu])        
        self.rear_tire.state = np.array([self.compute_long_slip(self.state[[2,6],0]),
                                         self.compute_lat_slip(*self.state[[2,3,4],0])[1],
                                         self.Fz_rear,
                                         self.mu])
        self.tire_forces = np.array([*self.front_tire.get_forces(), *self.rear_tire.get_forces()])
        self.slip_values = np.array([self.compute_long_slip(self.state[[2,5],0]),
                                     *self.compute_lat_slip(*self.state[2:5])])
        self.on_state_update()

    def long_dynamics(self, current_long_state: np.ndarray) -> np.ndarray:
        """
        Longitudinal dynamics defined as two wheels. States are:
            0. longitudinal_velocity
            1. front_wheel_angular_velocity
        """
        assert(1 >= self.action["acceleration"] >= -1)
        if self.is_braking:
            front_torque = 2*self.action["acceleration"]*self.max_brake_torque
            F_drag = 1/2*1.225*(1.6 + 0.00056*(self.mass - 765))*self.longitudinal_velocity**2
            full_braking_compensation = 1.3
        else:
            front_torque = self.action["acceleration"]*self.max_engine_torque
            F_drag = 0 # assume driver maintains speed, negating drag.
            full_braking_compensation = 1
        delta = self.action["steering"]

        assert((current_long_state.ndim == 1) & (current_long_state.size == 2))
        front_wheel_state = np.array(current_long_state[[0,1]])
        front_kappa = self.compute_long_slip(front_wheel_state)
        front_tire_state = self.front_tire.state
        front_Fx, front_Fy = self.front_tire.get_forces(
            np.insert(front_tire_state[1:],0,front_kappa))
        
        front_F = front_Fx*np.cos(delta) - front_Fy*np.sin(delta)
        front_F *= full_braking_compensation
        d_U = 2*(front_F - F_drag)/self.mass
        d_omega_front = (front_torque - self.wheel_radius*front_Fx)/self.wheel_inertia
        return np.array([d_U, d_omega_front])

    def compute_long_slip(self, wheel_state: np.ndarray) -> float:
        """
        States:
            0. wheel center longitudinal velocity [m/s]
            1. wheel rotational velocity [rad/s]
        """
        tangent_vel = self.wheel_radius*wheel_state[1]
        if math.isclose(wheel_state[0],0,abs_tol=1e-3):
            return 0
        elif math.isclose(tangent_vel,0,abs_tol=1e-3):
            return 0
        if self.is_braking:
            kappa = (tangent_vel - wheel_state[0])/wheel_state[0]
        else:
            kappa = (tangent_vel - wheel_state[0])/tangent_vel
        return kappa

    def lateral_dynamics(self, current_lat_state: np.ndarray) -> np.ndarray:
        """
        Lateral dynamics defined by nonlinear bicycle model. States are:
            0. lateral velocity [m/s]
            1. yaw rate [rad/s]
        """
        delta = self.action["steering"]
        V_x = self.longitudinal_velocity
        V_y, d_psi = current_lat_state
        front_alpha, rear_alpha = self.compute_lat_slip(V_x, V_y, d_psi)
        kappa_f, _, Fz_f, mu_f = self.front_tire.state
        kappa_r, _, Fz_r, mu_r = self.rear_tire.state
        front_Fx, front_Fy = self.front_tire.get_forces(np.array([kappa_f, front_alpha, Fz_f, mu_f]))
        _, rear_Fy = self.rear_tire.get_forces(np.array([kappa_r, rear_alpha, Fz_r, mu_r]))
        if abs(self.longitudinal_velocity) < 1:  # Low speed dynamics: damping of lateral speed and yaw rate
            front_Fy = - self.mass * self.lateral_velocity - self.inertia_z/self.cg_a* self.yaw_rate
            rear_Fy = - self.mass * self.lateral_velocity + self.inertia_z/self.cg_a * self.yaw_rate

        d_V_y = 1/self.mass*(front_Fy*np.cos(delta) + rear_Fy) - d_psi*V_x #- front_Fx*np.sin(delta) 
        dd_psi = 1/self.inertia_z*(self.cg_a*(front_Fy*np.cos(delta) ) #- front_Fx*np.sin(delta)
            - self.cg_b*rear_Fy)
        return np.array([d_V_y, dd_psi])

    def compute_lat_slip(self, V_x: float, V_y: float, d_psi: float) -> tuple:
        if math.isclose(V_x, 0, abs_tol=1e-2):
            V_x = math.copysign(1e-2, V_x)
        theta_vf = np.arctan2(V_y + self.cg_a * d_psi, V_x)
        theta_vr = np.arctan2(V_y - self.cg_b * d_psi, V_x)
        front_alpha = self.action["steering"] - theta_vf
        rear_alpha = - theta_vr
        return np.array([front_alpha, rear_alpha])

    def RK4(self, ode_func: Callable, current_state: np.ndarray, dt: float) -> np.ndarray:
        """4th Order Runge-Kutta Numerical Integration."""
        K1 = ode_func(current_state)
        K2 = ode_func(current_state + 0.5*dt*K1)
        K3 = ode_func(current_state + 0.5*dt*K2)
        K4 = ode_func(current_state + dt*K3)
        return current_state + (1/6)*(K1 + 2*K2 + 2*K3 + K4)*dt

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.longitudinal_velocity
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        # Impose actuator limits
        self.action['steering'] = np.clip(self.action['steering'], -np.pi/6, np.pi/6)
        self.action['acceleration'] = np.clip(self.action['acceleration'], -1, 1)
    
    def set_friction(self, mu) -> None:
        self.mu = mu


def simulate(dt: float = 0.1) -> None:
    import control
    time = np.arange(0, 20, dt)
    vehicle = BicycleVehicle(road=None, position=[0, 5], speed=8.3)
    xx, uu = [], []
    from highway_env.interval import LPV
    A, B = vehicle.full_lateral_lpv_dynamics()
    K = -np.asarray(control.place(A, B, -np.arange(1, 5)))
    lpv = LPV(x0=vehicle.state[[1, 2, 4, 5]].squeeze(), a0=A, da=[np.zeros(A.shape)], b=B,
              d=[[0], [0], [0], [1]], omega_i=[[0], [0]], u=None, k=K, center=None, x_i=None)

    for t in time:
        # Act
        u = K @ vehicle.state[[1, 2, 4, 5]]
        omega = 2*np.pi/20
        u_p = 0*np.array([[-20*omega*np.sin(omega*t) * dt]])
        u += u_p
        # Record
        xx.append(np.array([vehicle.position[0], vehicle.position[1], vehicle.heading])[:, np.newaxis])
        uu.append(u)
        # Interval
        lpv.set_control(u, state=vehicle.state[[1, 2, 4, 5]])
        lpv.step(dt)
        # x_i_t = lpv.change_coordinates(lpv.x_i_t, back=True, interval=True)
        # Step
        vehicle.act({"acceleration": 0, "steering": u})
        vehicle.step(dt)

    xx, uu = np.array(xx), np.array(uu)
    plot(time, xx, uu)


def plot(time: np.ndarray, xx: np.ndarray, uu: np.ndarray) -> None:
    pos_x, pos_y = xx[:, 0, 0], xx[:, 1, 0]
    psi_x, psi_y = np.cos(xx[:, 2, 0]), np.sin(xx[:, 2, 0])
    dir_x, dir_y = np.cos(xx[:, 2, 0] + uu[:, 0, 0]), np.sin(xx[:, 2, 0] + uu[:, 0, 0])
    _, ax = plt.subplots(1, 1)
    ax.plot(pos_x, pos_y, linewidth=0.5)
    dir_scale = 1/5
    ax.quiver(pos_x[::20]-0.5/dir_scale*psi_x[::20],
              pos_y[::20]-0.5/dir_scale*psi_y[::20],
              psi_x[::20], psi_y[::20],
              angles='xy', scale_units='xy', scale=dir_scale, width=0.005, headwidth=1)
    ax.quiver(pos_x[::20]+0.5/dir_scale*psi_x[::20], pos_y[::20]+0.5/dir_scale*psi_y[::20], dir_x[::20], dir_y[::20],
              angles='xy', scale_units='xy', scale=0.25, width=0.005, color='r')
    ax.axis("equal")
    ax.grid()
    plt.show()
    plt.close()


def main() -> None:
    simulate()


if __name__ == '__main__':
    main()
