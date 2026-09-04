# YOLO + kalbee: Bounding-Box Tracking (Vehicles, People, ...)

YOLO gives you per-frame detections — a list of boxes like
`[x1, y1, x2, y2]` with a class and a confidence. Detections alone are not
*tracks*: they jitter, flicker during occlusion, and carry no identity across
frames. kalbee's filters and SORT-style
[`MultiObjectTracker`](../examples.md) turn raw YOLO boxes into stable,
ID-carrying tracks.

This page covers three increasingly realistic setups:

| Setup | What it teaches | Files |
|---|---|---|
| Single target, simulated detections | filter comparison on box noise / occlusion | `examples/yolo_tracking.py` |
| Real video, one tracked target | nearest-neighbour association + filters | `examples/yolo_real_video.py` |
| **Real video, many boxes (vehicles / people)** | box-corner state + gated multi-target tracking | `examples/yolo_mot.py`, `yolo_vehicles.py`, `yolo_people.py` |

All three run with `pip install "kalbee[yolo]"` (ultralytics + opencv).

---

## Tracking a bounding box

kalbee works on numbers, so a box becomes a 4-dimensional measurement:

$$\text{box} = [x_1,\ y_1,\ x_2,\ y_2]^\top$$

The simplest faithful motion model treats each corner as an independent
constant-velocity axis. kalbee's model builders do this for *any* number of
axes (`n_dims`), so tracking box corners is exactly the same code as tracking
`(x, y)` positions — only `n_dims=4` differs:

```python
from kalbee import KalmanFilter, MultiObjectTracker
from kalbee.models import constant_velocity, position_measurement_model

# State layout (block-per-axis): [x1, vx1, y1, vy1, x2, vx2, y2, vy2]
F, Q = constant_velocity(dt=dt, process_var=1.0, n_dims=4)
H, R = position_measurement_model(order=1, n_dims=4, measurement_var=6.0**2)

def new_track(box):                       # called when a new object appears
    x0 = np.concatenate([np.array([[zi], [0.0]]) for zi in box])
    return KalmanFilter(x0, np.eye(8) * 50.0, F, Q, H, R)

tracker = MultiObjectTracker(new_track, n_init=2, max_age=30, gate=8.0)

for boxes in yolo_boxes_per_frame:        # (N, 4) or None
    confirmed = tracker.update(boxes)
    for t in confirmed:
        t.id                                   # stable object identity
        x1, y1, x2, y2 = t.state[[0, 2, 4, 6], 0]
```

`MultiObjectTracker` adds the parts a lone Kalman filter cannot:
Mahalanobis-gated Hungarian association, tentative→confirmed track states
(`n_init`), and deletion after `max_age` missed frames. Because each corner
*velocity* is part of the state, a confirmed track keeps predicting a
sensible box through short occlusions.

---

## Real-time multi-class tracking (vehicles + people)

`examples/yolo_mot.py` is a complete detector→tracker→annotator pipeline for
any source (video file or webcam):

```bash
# default classes: person, bicycle, car, motorcycle, bus, truck
python examples/yolo_mot.py --source path/to/traffic.mp4 --view

# save an annotated video instead of showing a window
python examples/yolo_mot.py --source path/to/traffic.mp4 --save out.mp4
```

### Vehicle-focused preset

`examples/yolo_vehicles.py` restricts tracking to the vehicle classes of the
COCO dataset — useful for traffic-flow and collision-avoidance studies:

```bash
python examples/yolo_vehicles.py --source highway.mp4 --save vehicles.mp4
```

### Person-focused preset

`examples/yolo_people.py` tracks only COCO class `0` (person) — the crowd /
pedestrian case:

```bash
python examples/yolo_people.py --source walkway.mp4 --view
```

The presets are thin wrappers that reuse `yolo_mot.py`, so every `--model`,
`--conf`, `--max-frames` and `--save` flag still works.

### Relevant COCO class ids

| id | class | | id | class |
|---|---|---|---|---|
| 0 | person | | 5 | bus |
| 1 | bicycle | | 6 | train |
| 2 | car | | 7 | truck |
| 3 | motorcycle | | 8 | boat |

```bash
python examples/yolo_mot.py --source video.mp4 --classes 2,5,7   # cars/buses/trucks only
python examples/yolo_mot.py --source video.mp4 --classes 0       # people only
```

---

## Ground truth from a real dataset: MOT16

To validate a tracker you want *ground-truth* boxes, not a detector's noisy
output. The [MOT Challenge](https://motchallenge.net) publishes real
pedestrian footage with hand-annotated boxes. kalbee ships a small fetcher
that pulls the MOT16-02 sequence (~15 MB) without downloading the multi-GB
archive:

```bash
uv run python scripts/fetch_mot16_02.py --frames 90   # -> data/MOT16-02
uv run python examples/mot16_pedestrian_tracking.py   # prints tracking metrics
uv run python scripts/mot16_demo.py --gif             # renders the animated demo
```

The animation below shows confirmed kalbee tracks (colored, ID-tagged)
following real pedestrians, with white boxes = per-frame detections:

<figure>
  <img src="../assets/gif/mot16_tracking.gif" alt="kalbee tracking real pedestrians in MOT16-02" width="640"/>
  <figcaption>kalbee MultiObjectTracker on MOT16-02 (real footage, ground-truth detections).</figcaption>
</figure>

---

## Choosing noise levels

Boxes are measured in pixels, so `Q` and `R` are *pixel* quantities:

| Parameter | Meaning | Typical value |
|---|---|---|
| `process_var` | how fast a corner is allowed to accelerate | 0.5–5 (px² per frame²) |
| `measurement_var` | detector jitter on a box corner | (2–10 px)² |
| `gate` | max Mahalanobis distance to accept a match | 5–10 |
| `n_init` | detections needed before a track is "confirmed" | 2–5 |
| `max_age` | missed frames before a track is deleted | 10–30 |

If you do not know your detector's jitter, run a few seconds of footage and
use the innovation statistics: `FilterDiagnostics` / `nis` report whether the
filter is over- or under-confident, and `quick_tune` estimates `Q`, `R` from
a measurement sequence automatically.

---

## Related

- [Multi-Object Tracking](multi_object_tracking.md) — tracker API in depth
- [Innovation Gating](gating.md) — how association gates reject false matches
- [Outlier Detection](outlier_detection.md) — robust updates in clutter
- [Examples & Gallery](../examples.md) — animated demos and MOT16 real footage
