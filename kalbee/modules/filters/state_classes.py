from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json


@dataclass
class FilterState:
    """
    Typed state class for filter snapshots.

    Provides a clean dataclass interface for accessing filter state
    with type hints and serialization support.
    """

    state_mean: np.ndarray
    state_covariance: np.ndarray
    timestamp: Optional[int] = None
    innovation: Optional[np.ndarray] = None
    innovation_covariance: Optional[np.ndarray] = None
    kalman_gain: Optional[np.ndarray] = None

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return len(self.state_mean)

    @property
    def measurement_dim(self) -> int:
        """Measurement dimension (from innovation)."""
        if self.innovation is not None:
            return len(self.innovation)
        return 0

    @property
    def covariance_trace(self) -> float:
        """Trace of state covariance (overall uncertainty)."""
        return float(np.trace(self.state_covariance))

    @property
    def state_std(self) -> np.ndarray:
        """Standard deviation of state elements."""
        return np.sqrt(np.diag(self.state_covariance))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "state_mean": self.state_mean.tolist(),
            "state_covariance": self.state_covariance.tolist(),
            "timestamp": self.timestamp,
            "covariance_trace": self.covariance_trace,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "FilterState":
        """Create from dictionary."""
        return cls(
            state_mean=np.array(d["state_mean"]),
            state_covariance=np.array(d["state_covariance"]),
            timestamp=d.get("timestamp"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FilterState":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FilterConfig:
    """
    Configuration dataclass for filter setup.

    Enables easy serialization and reproducibility of filter configurations.
    """

    filter_type: str
    state_dim: int
    measurement_dim: int
    transition_matrix: Optional[np.ndarray] = None
    transition_covariance: Optional[np.ndarray] = None
    measurement_matrix: Optional[np.ndarray] = None
    measurement_covariance: Optional[np.ndarray] = None
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "filter_type": self.filter_type,
            "state_dim": self.state_dim,
            "measurement_dim": self.measurement_dim,
            "parameters": self.parameters,
        }
        if self.transition_matrix is not None:
            d["transition_matrix"] = self.transition_matrix.tolist()
        if self.transition_covariance is not None:
            d["transition_covariance"] = self.transition_covariance.tolist()
        if self.measurement_matrix is not None:
            d["measurement_matrix"] = self.measurement_matrix.tolist()
        if self.measurement_covariance is not None:
            d["measurement_covariance"] = self.measurement_covariance.tolist()
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "FilterConfig":
        """Create from dictionary."""
        return cls(
            filter_type=d["filter_type"],
            state_dim=d["state_dim"],
            measurement_dim=d["measurement_dim"],
            transition_matrix=np.array(d["transition_matrix"])
            if "transition_matrix" in d
            else None,
            transition_covariance=np.array(d["transition_covariance"])
            if "transition_covariance" in d
            else None,
            measurement_matrix=np.array(d["measurement_matrix"])
            if "measurement_matrix" in d
            else None,
            measurement_covariance=np.array(d["measurement_covariance"])
            if "measurement_covariance" in d
            else None,
            parameters=d.get("parameters", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FilterConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
