"""
Track real pedestrians from the public MOT Challenge MOT16-02 sequence.

kalbee never needs to "see" the pixels: the classic MOT formulation feeds a
detector's per-frame bounding boxes to the tracker, which is exactly what
this example does with the sequence's ground-truth boxes. Watch confirmed
tracks keep stable IDs across the clip and bridge the frames where a person
is briefly occluded.

Setup (downloads only the needed frames + gt, ~15 MB)::

    uv run python scripts/fetch_mot16_02.py --frames 90   # -> data/MOT16-02
    uv run python examples/mot16_pedestrian_tracking.py

To render an animated GIF of the tracked boxes over the real footage:

    uv run python scripts/mot16_demo.py --gif

Data: Dendorfer et al., "MOT16: A Benchmark for Multi-Object Tracking",
arXiv:1603.00831. https://motchallenge.net
"""

import argparse
import os
import sys

import numpy as np

from kalbee import KalmanFilter, MultiObjectTracker
from kalbee.models import constant_velocity, position_measurement_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "MOT16-02")


def _load_boxes(gt_path, max_frame):
    boxes = {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7 or int(p[0]) > max_frame or float(p[6]) <= 0:
                continue
            x1 = float(p[2])
            y1 = float(p[3])
            boxes.setdefault(int(p[0]), []).append(
                [x1, y1, x1 + float(p[4]), y1 + float(p[5])]
            )
    return {k: np.asarray(v) for k, v in boxes.items()}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def make_tracker():
    F, Q = constant_velocity(dt=1.0, process_var=1.0, n_dims=4)
    H, R = position_measurement_model(order=1, n_dims=4, measurement_var=6.0**2)

    def factory(box):
        x0 = np.concatenate([np.array([[zi], [0.0]]) for zi in box[:4]])
        return KalmanFilter(x0, np.eye(8) * 50.0, F, Q, H, R)

    return MultiObjectTracker(factory, n_init=2, max_age=30, gate=8.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args(argv)

    gt = os.path.join(args.data, "gt", "gt.txt")
    if not os.path.exists(gt):
        print(
            f"ground truth not found at {gt}.\n"
            "Run first:  uv run python scripts/fetch_mot16_02.py",
            file=sys.stderr,
        )
        return 1

    detections = _load_boxes(gt, args.frames)
    tracker = make_tracker()

    track_ids = set()
    iou_hits = []
    for fr in range(1, args.frames + 1):
        z = detections.get(fr)
        confirmed = tracker.update(z if z is not None else None, dt=1.0)
        track_ids.update(t.id for t in confirmed)

        # How well does each filtered box overlap the detection it should match?
        unmatched = list(z) if z is not None else []
        for t in confirmed:
            box = t.state[[0, 2, 4, 6], 0]
            if unmatched:
                # greedy best match by IoU
                best = max(range(len(unmatched)), key=lambda k: _iou(box, unmatched[k]))
                iou_hits.append(_iou(box, unmatched[best]))
                del unmatched[best]

    print(f"frames:        {args.frames}")
    print(f"stable IDs:    {len(track_ids)} tracks were confirmed over the clip")
    if iou_hits:
        print(f"mean box IoU (filtered vs detections): {np.mean(iou_hits):.3f}")
        print(f"min  box IoU (filtered vs detections): {np.min(iou_hits):.3f}")
    print("\nVisualize:  uv run python scripts/mot16_demo.py --gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
