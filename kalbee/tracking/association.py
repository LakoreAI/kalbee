"""
Data association for multi-object tracking.

Provides the two ingredients every detection-based tracker (SORT, ByteTrack,
StrongSORT, ...) needs on top of a motion filter:

* cost functions — IoU for bounding boxes, Mahalanobis for gated distance;
* an assignment solver — optimal matching via the Hungarian algorithm with a
  cost gate that rejects implausible pairings.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from kalbee.modules.utils.linalg import safe_inv


def iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """
    Pairwise Intersection-over-Union between track and detection boxes.

    Args:
        tracks: Array of shape (T, 4) in ``[x1, y1, x2, y2]`` format.
        detections: Array of shape (D, 4) in ``[x1, y1, x2, y2]`` format.

    Returns:
        IoU matrix of shape (T, D) with values in [0, 1].
    """
    tracks = np.asarray(tracks, dtype=float).reshape(-1, 4)
    detections = np.asarray(detections, dtype=float).reshape(-1, 4)
    if tracks.size == 0 or detections.size == 0:
        return np.zeros((tracks.shape[0], detections.shape[0]))

    # Broadcast to (T, D, 4)
    t = tracks[:, None, :]
    d = detections[None, :, :]

    xx1 = np.maximum(t[..., 0], d[..., 0])
    yy1 = np.maximum(t[..., 1], d[..., 1])
    xx2 = np.minimum(t[..., 2], d[..., 2])
    yy2 = np.minimum(t[..., 3], d[..., 3])

    inter_w = np.clip(xx2 - xx1, 0.0, None)
    inter_h = np.clip(yy2 - yy1, 0.0, None)
    inter = inter_w * inter_h

    area_t = np.clip(t[..., 2] - t[..., 0], 0.0, None) * np.clip(
        t[..., 3] - t[..., 1], 0.0, None
    )
    area_d = np.clip(d[..., 2] - d[..., 0], 0.0, None) * np.clip(
        d[..., 3] - d[..., 1], 0.0, None
    )
    union = area_t + area_d - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def mahalanobis_matrix(
    predicted_measurements: List[np.ndarray],
    innovation_covariances: List[np.ndarray],
    detections: np.ndarray,
) -> np.ndarray:
    """
    Pairwise Mahalanobis distance between predicted measurements and detections.

    Args:
        predicted_measurements: List of ``T`` predicted measurement vectors
            ``H @ x`` (each shape (m,) or (m, 1)).
        innovation_covariances: List of ``T`` innovation covariances ``S``
            (each shape (m, m)) — the ``last_S`` a filter exposes.
        detections: Array of shape (D, m) of measured values.

    Returns:
        Distance matrix of shape (T, D).
    """
    detections = np.asarray(detections, dtype=float).reshape(len(detections), -1)
    T = len(predicted_measurements)
    D = detections.shape[0]
    dist = np.zeros((T, D))
    for i in range(T):
        z_pred = np.asarray(predicted_measurements[i], dtype=float).reshape(-1)
        S_inv = safe_inv(innovation_covariances[i])
        diff = detections - z_pred  # (D, m)
        # d^2 = diff S^-1 diff^T for each row
        dist[i] = np.einsum("dj,jk,dk->d", diff, S_inv, diff)
    return np.sqrt(np.clip(dist, 0.0, None))


def associate(
    cost_matrix: np.ndarray,
    max_cost: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Optimal one-to-one assignment via the Hungarian algorithm, gated by cost.

    Args:
        cost_matrix: Shape (T, D); lower cost = better match. Use e.g.
            ``1 - IoU`` or a Mahalanobis distance.
        max_cost: Pairings with cost strictly greater than this are rejected
            (the gate), leaving both parties unmatched.

    Returns:
        Tuple ``(matches, unmatched_tracks, unmatched_detections)`` where
        ``matches`` is a list of ``(track_index, detection_index)`` pairs.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    T, D = cost_matrix.shape

    if T == 0 or D == 0:
        return [], list(range(T)), list(range(D))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    matched_tracks = set()
    matched_dets = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

    unmatched_tracks = [t for t in range(T) if t not in matched_tracks]
    unmatched_detections = [d for d in range(D) if d not in matched_dets]
    return matches, unmatched_tracks, unmatched_detections
