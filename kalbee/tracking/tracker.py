"""
SORT-style multi-object tracker built on top of the kalbee filter core.

The tracker is deliberately thin: it owns track lifecycle and data association,
and delegates all state estimation to whatever :class:`BaseFilter` a user-supplied
factory produces. This keeps the estimation core untouched while turning it into a
practical detection-based tracker (the pattern SORT/ByteTrack/StrongSORT use).
"""

from typing import Callable, List, Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.tracking.association import mahalanobis_matrix, associate
from kalbee.tracking.track import Track


class MultiObjectTracker:
    """
    Online multi-object tracker with Mahalanobis-gated Hungarian association.

    Example::

        import numpy as np
        from kalbee import KalmanFilter, MultiObjectTracker
        from kalbee.models import constant_velocity, position_measurement_model

        F, Q = constant_velocity(dt=1.0, process_var=1.0, n_dims=2)
        H, R = position_measurement_model(order=1, n_dims=2, measurement_var=1.0)

        def factory(z):
            # z is a measured position (2,); seed the state with zero velocity.
            x0 = np.array([[z[0]], [0.0], [z[1]], [0.0]])
            return KalmanFilter(x0, np.eye(4) * 10.0, F, Q, H, R)

        tracker = MultiObjectTracker(factory, n_init=3, max_age=5)
        confirmed = tracker.update(np.array([[1.0, 2.0], [5.0, 6.0]]))
    """

    def __init__(
        self,
        filter_factory: Callable[[np.ndarray], BaseFilter],
        n_init: int = 3,
        max_age: int = 30,
        gate: float = 5.0,
    ):
        """
        Args:
            filter_factory: ``factory(measurement) -> BaseFilter`` building a new
                track's filter, initialized to the given detection. The filter
                must be linear (exposes ``measurement_matrix`` / ``measurement_covariance``).
            n_init: Consecutive hits to confirm a track.
            max_age: Consecutive misses before a confirmed track is deleted.
            gate: Maximum Mahalanobis distance for a valid track/detection match.
        """
        self.filter_factory = filter_factory
        self.n_init = n_init
        self.max_age = max_age
        self.gate = gate

        self.tracks: List[Track] = []
        self._next_id = 0

    def _spawn(self, measurement: np.ndarray) -> None:
        filt = self.filter_factory(np.asarray(measurement, dtype=float))
        self.tracks.append(
            Track(filt, self._next_id, n_init=self.n_init, max_age=self.max_age)
        )
        self._next_id += 1

    def update(
        self,
        detections: Optional[np.ndarray],
        dt: float = 1.0,
    ) -> List[Track]:
        """
        Advance the tracker by one frame.

        Args:
            detections: Array of shape (D, m) of measurement vectors, or None/empty.
            dt: Time step for the prediction.

        Returns:
            The list of currently confirmed tracks.
        """
        detections = (
            np.zeros((0, 0))
            if detections is None
            else np.asarray(detections, dtype=float)
        )
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)
        n_det = detections.shape[0]

        # 1. Predict every existing track forward.
        for track in self.tracks:
            track.predict(dt=dt)

        # 2. Associate predicted tracks with detections (Mahalanobis + Hungarian).
        if self.tracks and n_det > 0:
            predicted = [t.predicted_measurement() for t in self.tracks]
            innov_covs = [t.innovation_covariance() for t in self.tracks]
            cost = mahalanobis_matrix(predicted, innov_covs, detections)
            matches, unmatched_tracks, unmatched_dets = associate(cost, self.gate)
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(n_det))

        # 3. Correct matched tracks.
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx].reshape(-1, 1))

        # 4. Age out unmatched tracks.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 5. Start tentative tracks for unmatched detections.
        for det_idx in unmatched_dets:
            self._spawn(detections[det_idx])

        # 6. Drop dead tracks.
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        return self.confirmed_tracks()

    def confirmed_tracks(self) -> List[Track]:
        """Return the subset of tracks that are currently confirmed."""
        return [t for t in self.tracks if t.is_confirmed]
