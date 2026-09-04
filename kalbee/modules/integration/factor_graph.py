from typing import List, Dict, Any
import json
import numpy as np


class FactorGraphExporter:
    """
    Exports kalbee trajectory histories into Factor Graph format.

    Creates nodes (states) and factors (motion priors, measurement constraints)
    for global non-linear factor graph optimization solvers (GTSAM / Ceres / g2o).
    """

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.factors: List[Dict[str, Any]] = []

    def add_state_node(self, step_idx: int, state: np.ndarray, covariance: np.ndarray):
        """Add a state variable node at step k."""
        self.nodes.append(
            {
                "id": f"x_{step_idx}",
                "type": "state",
                "step": step_idx,
                "val": state.tolist(),
                "cov": covariance.tolist(),
            }
        )

    def add_motion_factor(
        self,
        step_from: int,
        step_to: int,
        transition_matrix: np.ndarray,
        process_noise: np.ndarray,
    ):
        """Add a motion factor connecting x_{step_from} to x_{step_to}."""
        self.factors.append(
            {
                "type": "motion_prior",
                "from": f"x_{step_from}",
                "to": f"x_{step_to}",
                "F": transition_matrix.tolist(),
                "Q": process_noise.tolist(),
            }
        )

    def add_measurement_factor(
        self,
        step_idx: int,
        measurement: np.ndarray,
        measurement_matrix: np.ndarray,
        noise: np.ndarray,
    ):
        """Add a measurement factor for x_{step_idx}."""
        self.factors.append(
            {
                "type": "measurement",
                "target": f"x_{step_idx}",
                "z": measurement.tolist(),
                "H": measurement_matrix.tolist(),
                "R": noise.tolist(),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return factor graph dictionary representation."""
        return {
            "nodes": self.nodes,
            "factors": self.factors,
        }

    def save_json(self, filepath: str):
        """Save factor graph schema to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
