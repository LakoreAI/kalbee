import numpy as np
from typing import Optional
from kalbee.modules.filters.base import BaseFilter


class AlphaBetaGammaFilter(BaseFilter):
    """
    Implementation of the Alpha-Beta-Gamma filter.
    Used for tracking position, velocity, and acceleration with constant gains.
    """

    def __init__(
        self,
        state: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        initial_covariance: Optional[np.ndarray] = None,
    ):
        """
        Initialize the alpha-beta-gamma filter.

        Args:
            state: Initial state vector [x, v, a]^T
            alpha: Position gain
            beta: Velocity gain
            gamma: Acceleration gain
            initial_covariance: Optional initial covariance (P)
        """
        if initial_covariance is None:
            # Default identity covariance if not provided
            initial_covariance = np.eye(len(state))

        super().__init__(state, initial_covariance)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Advance the state using the kinematics model:
        x_k = x_{k-1} + v_{k-1}*dt + 0.5*a_{k-1}*dt^2
        v_k = v_{k-1} + a_{k-1}*dt
        a_k = a_{k-1}
        """
        x, v, a = self.state.flatten()

        new_x = x + v * dt + 0.5 * a * (dt**2)
        new_v = v + a * dt
        new_a = a

        self.state = np.array([[new_x], [new_v], [new_a]])

        # In ABG filters, we typically don't update P during predict
        # as gains are fixed.
        return self.state

    def update(self, measurement: np.ndarray, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Update the state using the residual and fixed gains alpha, beta, gamma.
        """
        # Residual (innovation) - comparing measurement to predicted position
        z = np.asanyarray(measurement).flatten()[0]
        x_pred = self.state[0, 0]
        residual = z - x_pred

        # Apply corrections
        self.state[0, 0] += self.alpha * residual
        self.state[1, 0] += (self.beta / dt) * residual
        self.state[2, 0] += (self.gamma / (2 * dt**2)) * residual

        return self.state
