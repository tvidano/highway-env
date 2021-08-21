import numpy as np


class AbstractTire(object):
    """General object for tire model."""

    def __init__(self, init_state: np.ndarray = None, params: dict = None) -> None:
        # Configure
        self.params = self.default_params()
        self.configure(params)
        self.state = init_state if init_state else np.array([0, 0])

    def default_params(self):
        """Physically reasonable tire parameters."""
        raise NotImplementedError

    def configure(self, params: dict) -> None:
        if params:
            self.params.update(params)

    @property
    def state(self) -> np.ndarray:
        """The first state is long. slip the second is lat. slip."""
        return self._state

    @state.setter
    def state(self, new_state) -> None:
        assert (new_state.size == 2)
        assert (1 >= new_state[0] >= -1)
        self._state = new_state

    def get_forces(self) -> np.ndarray:
        """
        Calculate tire forces with tire_state as input.

        :param tire_state: current state of the tire
        """
        raise NotImplementedError


class LinearTire(AbstractTire):
    """
    Linear Tire Model where the tire state is parameterized by
    longitudinal slip and lateral slip.
    """

    def __init__(self, init_state: np.ndarray, params: dict = None) -> None:
        super().__init__(init_state, params)

    def default_params(self):
        params = {
            "long_stiffness": 15,
            "lat_stiffness": 15
        }
        return params

    def get_forces(self) -> np.ndarray:
        long_force = self.params["long_stiffness"] * self.state
        lat_force = self.params["lat_stiffness"] * self.state
        return np.array([long_force, lat_force])


# TODO{tvidano}: Complete this implementation.
class DugoffTire(AbstractTire):
    """Dugoff Tire Model."""

    def __init__(self, params: dict = None) -> None:
        super().__init__(params)

    def default_params(self):
        params = {
            "long_stiffness": 15,
            "lat_stiffness": 15,
            "epsilon": 0.01
        }
        return params

    def get_forces(self, tire_state: dict) -> dict:
        mu = tire_state["mu"]
        Fz = tire_state["normal_force"]
        kappa = tire_state["long_slip"]
        alpha = tire_state["lat_slip"]
        Cy = self.params["lat_stiffness"]
        Cx = self.params["long_stiffness"]
        Fy = Cy * np.tan(alpha) / (1 - abs(kappa))
        Fx = Cx * kappa / (1 - abs(kappa))
        lam = mu / (2 * np.sqrt((Fy / Fz) ** 2 + (Fx / Fz) ** 2))
        if lam < 1:
            Fy *= 2 * lam * (1 - lam / 2)
            Fx *= 2 * lam * (1 - lam / 2)
        return (Fx, Fy)


class ConstantPacTire(AbstractTire):
    """
    Pacejka Tire Model with Constant Coefficients using a circle for combined
    lateral and longitudinal forces.
    """

    def __init__(self, init_state: np.ndarray = None, params: dict = None) -> None:
        if init_state is None:
            init_state = np.array([0, 0, 1500, 1.0])
        super().__init__(init_state, params)

    def default_params(self):
        params = {
            "Bx": 9,
            "Cx": 1.3,
            "Ex": -0.9,  # 0.95,
            "By": 8,
            "Cy": 1.4,
            "Ey": -2
        }
        return params

    @property
    def state(self) -> np.ndarray:
        """
        The states are as follows:
            0. longitudinal slip [-1,1],
            1. lateral slip [rad],
            2. normal force [N],
            3. tire-road coefficient of friction (0,1]
        """
        return self._state

    @state.setter
    def state(self, new_state) -> None:
        assert (new_state.size == 4)
        # assert(1 >= new_state[0] >= -1)
        assert (new_state[2] > 0)
        assert (1 >= new_state[3] > 0)
        self._state = new_state

    def get_forces(self, new_state: np.ndarray = None) -> np.ndarray:
        if new_state is not None:
            kappa, alpha, Fz, mu = new_state
        else:
            kappa, alpha, Fz, mu = self.state
        F_max = mu * Fz
        pac = lambda Fz, mu, C, B, E, input: Fz * mu * np.sin(C * np.arctan(B * input - E * (B * input
                                                                                             - np.arctan(B * input))))
        Fx = pac(Fz, mu, self.params["Cx"], self.params["Bx"],
                 self.params["Ex"], kappa)
        Fy = pac(Fz, mu, self.params["Cy"], self.params["By"],
                 self.params["Ey"], alpha)
        if ((Fy / F_max) ** 2 + (Fx / F_max) ** 2) > 1:
            Fx = Fx / np.linalg.norm(np.array([Fx, Fy])) * F_max
            Fy = Fy / np.linalg.norm(np.array([Fx, Fy])) * F_max
        return np.array([Fx, Fy])

    def get_tire_curves(self, tire_states: np.ndarray = None) -> list:
        """
        Calculates tire forces for the given a time series of tire_state.
        """
        if tire_states is None:
            n = 100
            kappas = np.linspace(-1, 1, n)
            alphas = np.linspace(-15, 15, n) * np.pi / 180
            Fzs = np.ones(n) * 1500
            mus = np.ones(n) * 1.0
            tire_states = np.vstack([kappas, alphas, Fzs, mus])
        pure_long_inputs = np.vstack([tire_states[0, :], np.zeros(n), tire_states[2:, :]])
        pure_lat_inputs = np.vstack([np.zeros(n), tire_states[1:, :]])
        input_list = [pure_long_inputs, pure_lat_inputs, tire_states]
        tire_curves = []
        for i, inputs in enumerate(input_list):
            self.state = inputs[:, 0]
            Fx, Fy = self.get_forces()
            if i == 0:
                tire_curve = np.array([inputs[0, 0], Fx])
            elif i == 1:
                tire_curve = np.array([inputs[1, 0], Fy])
            elif i == 2:
                tire_curve = np.array([Fx, Fy])
            for j in range(np.size(inputs, 1) - 1):
                self.state = inputs[:, j + 1]
                Fx, Fy = self.get_forces()
                if i == 0:
                    output = np.array([inputs[0, j + 1], Fx])
                elif i == 1:
                    output = np.array([inputs[1, j + 1], Fy])
                elif i == 2:
                    output = np.array([Fx, Fy])
                tire_curve = np.vstack([tire_curve, [output]])
            tire_curves.append(tire_curve)
        return tire_curves