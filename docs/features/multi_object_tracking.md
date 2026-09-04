# Multi-Object Tracking

`MultiObjectTracker` turns any linear kalbee filter into a **SORT-style online tracker**. It owns only the two concerns a filter does not — data association and track lifecycle — and delegates all state estimation to the filter core. This is exactly the pattern used by SORT, ByteTrack, and StrongSORT, where a Kalman filter is the motion model.

## Pipeline

Each call to `update(detections)` performs:

1. **Predict** every existing track forward one step.
2. **Associate** predicted tracks with detections using the Hungarian algorithm on a cost matrix, gated so implausible pairings are rejected.
3. **Update** matched tracks with their detection.
4. **Age out** unmatched tracks (miss counter).
5. **Spawn** tentative tracks for unmatched detections.
6. **Delete** dead tracks.

## Track Lifecycle

Each `Track` runs a small state machine:

```
tentative ──(n_init consecutive hits)──▶ confirmed ──(max_age misses)──▶ deleted
    │
    └──(a miss before confirming)──▶ deleted
```

Only **confirmed** tracks are returned from `update()`, which suppresses spurious one-frame detections.

## Data Association

The cost matrix is built with a **Mahalanobis distance** that reuses each filter's innovation covariance $S = H P H^\top + R$:

$$
d^2(\text{track}, \text{det}) = (z - H\hat{x})^\top S^{-1} (z - H\hat{x})
$$

Assignment is solved optimally with `scipy.optimize.linear_sum_assignment`, and any pairing whose cost exceeds `gate` is rejected. The primitives are also exposed directly:

```python
from kalbee.tracking import iou_matrix, mahalanobis_matrix, associate

# IoU cost for bounding-box tracking
cost = 1.0 - iou_matrix(track_boxes, detection_boxes)
matches, unmatched_tracks, unmatched_dets = associate(cost, max_cost=0.7)
```

## Example

```python
import numpy as np
from kalbee import KalmanFilter, MultiObjectTracker
from kalbee.models import constant_velocity, position_measurement_model

F, Q = constant_velocity(dt=1.0, process_var=0.05, n_dims=2)
H, R = position_measurement_model(order=1, n_dims=2, measurement_var=0.5)

def new_track(z):
    # Seed state [x, vx, y, vy] at the detection with zero initial velocity.
    x0 = np.array([[z[0]], [0.0], [z[1]], [0.0]])
    return KalmanFilter(x0, np.eye(4) * 10.0, F, Q, H, R)

tracker = MultiObjectTracker(new_track, n_init=3, max_age=5, gate=6.0)

# detection_stream yields (D, 2) arrays of measured positions per frame
for detections in detection_stream:
    confirmed = tracker.update(detections)
    for t in confirmed:
        print(f"id={t.id}  pos=({t.state[0, 0]:.2f}, {t.state[2, 0]:.2f})")
```

!!! note "Full runnable demo"
    See [`examples/multi_object_tracking.py`](https://github.com/LakoreAI/kalbee/blob/main/examples/multi_object_tracking.py) for a three-target scene (with a crossing) that maintains stable identities.

## Parameters

| Parameter | Meaning |
|---|---|
| `filter_factory` | `factory(measurement) -> BaseFilter`, builds a new track's filter seeded on a detection |
| `n_init` | Consecutive hits required to confirm a track |
| `max_age` | Consecutive misses tolerated before a confirmed track is deleted |
| `gate` | Maximum Mahalanobis distance for a valid track/detection match |

!!! tip "Pairing with YOLO"
    Feed the box centers from an object detector (see [YOLO Object Tracking](yolo_tracking.md)) as the per-frame detections to get a complete detection-based tracker.
