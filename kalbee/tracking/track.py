"""
Single-target track lifecycle management.

A :class:`Track` wraps any linear :class:`~kalbee.modules.filters.base.BaseFilter`
(a constant-velocity Kalman filter is the classic SORT choice) and adds the
bookkeeping a multi-object tracker needs: a tentative -> confirmed -> deleted
state machine driven by hit/miss streaks.
"""

import numpy as np

from kalbee.modules.filters.base import BaseFilter

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
DELETED = "deleted"


class Track:
    """A single tracked object backed by a Kalman-family filter."""

    def __init__(
        self,
        filter_obj: BaseFilter,
        track_id: int,
        n_init: int = 3,
        max_age: int = 30,
    ):
        """
        Args:
            filter_obj: The per-track filter, already initialized to the first
                detection. Must expose ``measurement_matrix`` and
                ``measurement_covariance`` (i.e. a linear filter).
            track_id: Unique identifier for this track.
            n_init: Consecutive hits required to promote tentative -> confirmed.
            max_age: Consecutive misses tolerated before a confirmed track is
                deleted.
        """
        self.filter = filter_obj
        self.id = track_id
        self.n_init = n_init
        self.max_age = max_age

        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.status = TENTATIVE

    @property
    def state(self) -> np.ndarray:
        return self.filter.state

    @property
    def covariance(self) -> np.ndarray:
        return self.filter.covariance

    @property
    def is_confirmed(self) -> bool:
        return self.status == CONFIRMED

    @property
    def is_deleted(self) -> bool:
        return self.status == DELETED

    def predicted_measurement(self) -> np.ndarray:
        """Predicted measurement ``H @ x`` for the current (predicted) state."""
        H = self.filter.measurement_matrix
        return H @ self.filter.state

    def innovation_covariance(self) -> np.ndarray:
        """Innovation covariance ``S = H P Hᵀ + R`` for gating/association."""
        H = self.filter.measurement_matrix
        R = self.filter.measurement_covariance
        P = self.filter.covariance
        return H @ P @ H.T + R

    def predict(self, dt: float = 1.0) -> None:
        """Advance the filter one step and age the track."""
        self.filter.predict(dt=dt)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, measurement: np.ndarray) -> None:
        """Correct the filter with a matched detection and register the hit."""
        self.filter.update(measurement)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        if self.status == TENTATIVE and self.hits >= self.n_init:
            self.status = CONFIRMED

    def mark_missed(self) -> None:
        """Register a missed association and update the lifecycle state."""
        if self.status == TENTATIVE:
            # A tentative track that fails to associate is discarded immediately.
            self.status = DELETED
        elif self.time_since_update > self.max_age:
            self.status = DELETED
