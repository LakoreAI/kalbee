from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import json
import numpy as np


class BaseFilter(ABC):
    """
    Abstract base class for state estimation filters.

    This class defines the interface for filters like Kalman Filter,
    Extended Kalman Filter, and alpha-beta-gamma filters.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: Optional[np.ndarray] = None,
        transition_covariance: Optional[np.ndarray] = None,
        measurement_matrix: Optional[np.ndarray] = None,
        measurement_covariance: Optional[np.ndarray] = None,
    ):
        """
        Initialize the filter.

        Args:
            state: Initial state vector (n x 1)
            covariance: Initial state uncertainty matrix (n x n)
            transition_matrix: Matrix F that defines state progression.
            transition_covariance: Process noise covariance matrix Q.
            measurement_matrix: Matrix H that maps state to measurement.
            measurement_covariance: Measurement noise covariance matrix R.
        """
        self.state = np.asanyarray(state).astype(float)
        self.covariance = np.asanyarray(covariance).astype(float)

        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.measurement_matrix = measurement_matrix
        self.measurement_covariance = measurement_covariance

    @property
    def x(self) -> np.ndarray:
        """Current state estimate."""
        return self.state

    @property
    def P(self) -> np.ndarray:
        """Current state covariance."""
        return self.covariance

    @abstractmethod
    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict the next state of the system.

        Args:
            dt: Time step since the last update.
            **kwargs: Additional parameters for specific implementations.

        Returns:
            The predicted state vector.
        """
        pass

    @abstractmethod
    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update the state estimate with a new measurement.

        Args:
            measurement: The observed measurement vector.
            **kwargs: Additional parameters for specific implementations.

        Returns:
            The updated state vector.
        """
        pass

    def measure(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Map a state to the measurement space.

        Args:
            state: The state to map. If None, uses internal state.

        Returns:
            The expected measurement.
        """
        if state is None:
            state = self.state
        if self.measurement_matrix is None:
            raise ValueError("Measurement matrix H is not defined.")
        return self.measurement_matrix @ state

    def predict_only(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Perform only the prediction step without modifying filter diagnostics.

        This is useful when you need to predict the state ahead without
        updating (e.g., for planning or trajectory prediction).

        Args:
            dt: Time step.
            **kwargs: Additional parameters passed to predict().

        Returns:
            The predicted state vector.
        """
        # Save current state for restoration
        saved_state = self.state.copy()
        saved_cov = self.covariance.copy()

        # Perform prediction
        self.predict(dt=dt, **kwargs)
        predicted_state = self.state.copy()

        # Restore original state (this was just a prediction, not an update)
        self.state = saved_state
        self.covariance = saved_cov

        return predicted_state

    def reset(
        self,
        state: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
    ):
        """
        Reset the filter to a new state without recreating the object.

        Useful when tracking a new target or reinitializing after divergence.

        Args:
            state: New initial state (n x 1). If None, uses zeros.
            covariance: New initial covariance (n x n). If None, uses large
                       uncertainty (100 * I).
        """
        n = self.state.shape[0]

        if state is not None:
            self.state = np.asarray(state, dtype=float).reshape(n, 1)
        else:
            self.state = np.zeros((n, 1))

        if covariance is not None:
            self.covariance = np.asarray(covariance, dtype=float)
        else:
            self.covariance = np.eye(n) * 100.0

        # Clear any cached diagnostics
        for attr in ["last_y", "last_S", "K"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

    def filter_sequence(
        self, measurements: np.ndarray, dt: float = 1.0, missing: Optional[float] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process a full sequence of measurements through predict-update cycle.

        Args:
            measurements: Array of shape (T, m) or (T,) for scalar measurements.
                          Use ``missing`` value to indicate missing measurements.
            dt: Time step for prediction.
            missing: Value that indicates a missing measurement (e.g., NaN).
                     If a row equals this value, only predict is performed.

        Returns:
            Tuple of (state_history, covariance_history) where state_history
            is shape (T, n, 1) and covariance_history is a list of (n, n) arrays.
        """
        z = np.asarray(measurements, dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        T = z.shape[0]
        n = self.state.shape[0]

        state_history = np.zeros((T, n, 1))
        cov_history = []

        for t in range(T):
            self.predict(dt=dt)

            is_missing = False
            if missing is not None:
                if np.all(np.asarray(z[t]) == missing) or np.any(np.isnan(z[t])):
                    is_missing = True

            if not is_missing:
                self.update(z[t].reshape(-1, 1))

            state_history[t] = self.state.copy()
            cov_history.append(self.covariance.copy())

        return state_history, cov_history

    def save_state(self, filepath: str) -> None:
        """
        Save filter state to a JSON file.

        Args:
            filepath: Path to save the state file.
        """
        state_dict = {
            "state": self.state.tolist(),
            "covariance": self.covariance.tolist(),
        }
        if self.transition_matrix is not None:
            state_dict["transition_matrix"] = self.transition_matrix.tolist()
        if self.transition_covariance is not None:
            state_dict["transition_covariance"] = self.transition_covariance.tolist()
        if self.measurement_matrix is not None:
            state_dict["measurement_matrix"] = self.measurement_matrix.tolist()
        if self.measurement_covariance is not None:
            state_dict["measurement_covariance"] = self.measurement_covariance.tolist()

        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2)

    def load_state(self, filepath: str) -> None:
        """
        Load filter state from a JSON file.

        Args:
            filepath: Path to the state file.
        """
        with open(filepath, "r") as f:
            state_dict = json.load(f)

        self.state = np.array(state_dict["state"])
        self.covariance = np.array(state_dict["covariance"])

        if "transition_matrix" in state_dict and self.transition_matrix is not None:
            self.transition_matrix = np.array(state_dict["transition_matrix"])
        if "transition_covariance" in state_dict and self.transition_covariance is not None:
            self.transition_covariance = np.array(state_dict["transition_covariance"])
        if "measurement_matrix" in state_dict and self.measurement_matrix is not None:
            self.measurement_matrix = np.array(state_dict["measurement_matrix"])
        if "measurement_covariance" in state_dict and self.measurement_covariance is not None:
            self.measurement_covariance = np.array(state_dict["measurement_covariance"])
