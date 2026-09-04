from typing import Optional, Callable
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class ExtendedKalmanFilter(BaseFilter):
    """
    Extended Kalman Filter (EKF) implementation for non-linear systems.

    The EKF handles non-linearity by linearizing the system at the current estimate
    using Jacobians.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        transition_function: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        measurement_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        transition_jacobian: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        measurement_jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize the EKF.

        Args:
            state: Initial state vector (x).
            covariance: Initial state covariance matrix (P).
            transition_covariance: Process noise covariance (Q).
            measurement_covariance: Measurement noise covariance (R).
            transition_function: Function f(x, dt) returning the predicted state.
            measurement_function: Function h(x) returning the measurement from state.
            transition_jacobian: Function F(x, dt) returning the Jacobian of f.
            measurement_jacobian: Function H(x) returning the Jacobian of h.
        """
        # We pass None for matrices that are dynamic/non-linear
        super().__init__(
            state=state,
            covariance=covariance,
            transition_covariance=transition_covariance,
            measurement_covariance=measurement_covariance,
            transition_matrix=None,
            measurement_matrix=None,
        )

        self.transition_function = transition_function
        self.measurement_function = measurement_function
        self.transition_jacobian = transition_jacobian
        self.measurement_jacobian = measurement_jacobian

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step for EKF.

        Args:
            dt: Time step.
            kwargs:
                f: Optional override for transition function.
                F: Optional override for transition Jacobian.
        """
        # 1. State Prediction: x = f(x, u)
        f_func = kwargs.get("f", self.transition_function)
        if f_func is None:
            raise ValueError(
                "Transition function 'f' must be defined either in init or kwargs."
            )

        # We assume f takes (state, dt)
        self.state = f_func(self.state, dt)

        # 2. Covariance Prediction: P = FPF' + Q
        F_func = kwargs.get("F", self.transition_jacobian)

        # If F is a matrix (constant Jacobian), use it directly
        if isinstance(F_func, np.ndarray):
            F = F_func
        elif callable(F_func):
            F = F_func(self.state, dt)
        else:
            raise ValueError(
                "Transition Jacobian 'F' must be defined either in init or kwargs."
            )

        Q = self.transition_covariance
        self.covariance = F @ self.covariance @ F.T + Q

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step for EKF.

        Args:
            measurement: Measurement vector z.
            kwargs:
                h: Optional override for measurement function.
                H: Optional override for measurement Jacobian.
        """
        # 1. Innovation: y = z - h(x)
        h_func = kwargs.get("h", self.measurement_function)
        if h_func is None:
            raise ValueError(
                "Measurement function 'h' must be defined either in init or kwargs."
            )

        z = measurement
        y = z - h_func(self.state)

        # 2. Innovation Covariance
        H_func = kwargs.get("H", self.measurement_jacobian)

        if isinstance(H_func, np.ndarray):
            H = H_func
        elif callable(H_func):
            H = H_func(self.state)
        else:
            raise ValueError(
                "Measurement Jacobian 'H' must be defined either in init or kwargs."
            )

        R = self.measurement_covariance
        P = self.covariance

        S = H @ P @ H.T + R

        # 3. Kalman Gain: K = PH'S^-1
        K = P @ H.T @ safe_inv(S)

        # 4. State Update: x = x + Ky
        self.state = self.state + K @ y

        # 5. Covariance Update (Joseph form for stability): P = (I - KH)P(I - KH)' + KRK'
        identity = np.eye(len(self.state))
        I_KH = identity - K @ H
        self.covariance = I_KH @ P @ I_KH.T + K @ R @ K.T

        # Enforce symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        # Save for IMM and diagnostics
        self.last_y = y
        self.last_S = S

        return self.state
