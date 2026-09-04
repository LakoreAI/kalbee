"""
Fading Memory (Discounted) Kalman Filter.

A variant of the standard Kalman Filter that applies a discount factor to
the predicted covariance, giving more weight to recent measurements.
Useful for tracking maneuvering targets where the process model is imperfect
and older predictions should be "forgotten" faster.

The fading factor alpha > 1 inflates the predicted covariance:
    P_pred = alpha * F * P * F' + Q

When alpha = 1, this reduces to the standard Kalman Filter.

References:
    - Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
      Estimation with Applications to Tracking and Navigation, Section 6.4.
    - Kailath, T., Sayed, A. H., & Hassibi, B. (2000).
      Linear Estimation, Section 11.2.
"""

from typing import Optional
import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter


class FadingMemoryKalmanFilter(KalmanFilter):
    """
    Fading Memory Kalman Filter with covariance discounting.

    Extends the standard KF by inflating the predicted covariance with a
    fading factor alpha >= 1. This prevents the filter from becoming
    overconfident in its model predictions, which is critical when:
    - The target performs maneuvers not captured by the motion model
    - There are unmodeled dynamics or disturbances
    - The process noise Q is underestimated

    A typical value is alpha = 1.01 to 1.1 for mild fading.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        control_matrix: Optional[np.ndarray] = None,
        fading_factor: float = 1.05,
    ):
        """
        Initialize the Fading Memory Kalman Filter.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_matrix: State transition matrix F (n x n).
            transition_covariance: Process noise covariance Q (n x n).
            measurement_matrix: Measurement matrix H (m x n).
            measurement_covariance: Measurement noise covariance R (m x m).
            control_matrix: Optional control input matrix B (n x n_u).
            fading_factor: Discount factor alpha >= 1. Values > 1 inflate
                          the predicted covariance. Default is 1.05.
        """
        if fading_factor < 1.0:
            raise ValueError(f"Fading factor must be >= 1.0, got {fading_factor}")

        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
            control_matrix=control_matrix,
        )
        self.fading_factor = fading_factor

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step with fading memory covariance inflation.

        P_pred = alpha * (F * P * F') + Q

        Args:
            dt: Time step.
            **kwargs:
                F: Optional override for transition matrix.
                u: Optional control input vector.
                B: Optional override for control matrix.

        Returns:
            The predicted state vector.
        """
        F = self.transition_matrix
        Q = self.transition_covariance

        if "F" in kwargs:
            F = kwargs["F"]

        self.state = F @ self.state

        # Apply control input if provided
        u = kwargs.get("u", None)
        B = kwargs.get("B", self.control_matrix)
        if u is not None and B is not None:
            self.state = self.state + B @ np.asarray(u).reshape(-1, 1)

        # Fading memory: inflate predicted covariance
        self.covariance = self.fading_factor * (F @ self.covariance @ F.T) + Q

        return self.state
