from typing import List
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.fusion.covariance_intersection import covariance_intersection


class FederatedKalmanFilter:
    """
    Federated Kalman Filter for distributed multi-sensor fusion.

    Hierarchical fusion where local filters process sensor data independently,
    then a master filter fuses their estimates using covariance intersection.

    Architecture:
        Sensor 1 → Local Filter 1 ─┐
        Sensor 2 → Local Filter 2 ─┼→ Federated Fusion → Global Estimate
        Sensor 3 → Local Filter 3 ─┘

    Usage:
        federated = FederatedKalmanFilter(local_filters, global_filter)
        for measurements in sensor_stream:
            federated.predict()
            fused = federated.update(measurements)
    """

    def __init__(
        self,
        local_filters: List[BaseFilter],
        global_filter: BaseFilter,
        omega: float = 0.5,
    ):
        """
        Initialize the Federated KF.

        Args:
            local_filters: List of local filter instances (one per sensor).
            global_filter: Global filter for fused state.
            omega: Weight for covariance intersection (0-1). None for auto-optimal.
        """
        self.local_filters = local_filters
        self.global_filter = global_filter
        self.omega = omega
        self.n_sensors = len(local_filters)

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step for all local and global filters.

        Args:
            dt: Time step.

        Returns:
            Global predicted state.
        """
        for lf in self.local_filters:
            lf.predict(dt=dt, **kwargs)

        self.global_filter.predict(dt=dt, **kwargs)
        return self.global_filter.x

    def update(self, measurements: List[np.ndarray], **kwargs) -> np.ndarray:
        """
        Update all local filters and fuse their estimates.

        Args:
            measurements: List of measurement vectors (one per sensor).
                          Use None for sensors with no measurement this step.

        Returns:
            Fused global state estimate.
        """
        if len(measurements) != self.n_sensors:
            raise ValueError(
                f"Expected {self.n_sensors} measurements, got {len(measurements)}"
            )

        # Update local filters
        local_estimates = []
        for lf, z in zip(self.local_filters, measurements):
            if z is not None:
                lf.update(z, **kwargs)
            local_estimates.append((lf.x.copy(), lf.P.copy()))

        # Fuse using covariance intersection
        fused_mean, fused_cov = local_estimates[0]
        for mean_i, cov_i in local_estimates[1:]:
            fused_mean, fused_cov = covariance_intersection(
                fused_mean, fused_cov,
                mean_i, cov_i,
                omega=self.omega,
            )

        # Update global filter state with fused estimate
        self.global_filter.state = fused_mean
        self.global_filter.covariance = fused_cov

        return self.global_filter.x

    @property
    def state(self) -> np.ndarray:
        """Get global fused state."""
        return self.global_filter.x

    @property
    def covariance(self) -> np.ndarray:
        """Get global fused covariance."""
        return self.global_filter.P
