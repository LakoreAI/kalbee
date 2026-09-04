"""
Track detection *bounding boxes* (vehicles, people, ...) from a real video
with kalbee's SORT-style MultiObjectTracker.

This is the multi-object extension of the "smooth a detector's boxes" idea.
Each detection is a box ``[x1, y1, x2, y2]``. kalbee's tracker treats the
four corners as independent constant-velocity axes (state layout
``[x1, v1, y1, v2, x2, v3, y2, v4]``), so each track keeps one stable ID and
keeps predicting through detector dropouts/occlusions.

Usage (needs ``pip install "kalbee[yolo]"`` + a video/webcam)::

    python examples/yolo_mot.py --source path/to/traffic.mp4 --view
    python examples/yolo_mot.py --source 0 --classes 2,5,7     # cars/buses/trucks
    python examples/yolo_vehicles.py --source video.mp4 --save out.mp4
    python examples/yolo_people.py    --source video.mp4 --view

COCO class ids (subset): 0 person, 1 bicycle, 2 car, 3 motorcycle,
4 aeroplane, 5 bus, 6 train, 7 truck, 8 boat.
"""

import argparse
import os

import cv2
import numpy as np

from kalbee import KalmanFilter, MultiObjectTracker
from kalbee.models import constant_velocity, position_measurement_model

COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "aeroplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
}
DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7]  # person + common vehicles

_ID_COLORS = [
    (214, 39, 40),
    (31, 119, 180),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
    (23, 190, 207),
    (227, 119, 194),
    (188, 189, 34),
]


def make_tracker(dt=1.0, pos_var=2.0, meas_var=36.0, n_init=2, max_age=30):
    """SORT-style tracker on box corners (state [x1,y1,x2,y2] + velocities)."""
    F, Q = constant_velocity(dt=dt, process_var=pos_var, n_dims=4)
    H, R = position_measurement_model(order=1, n_dims=4, measurement_var=meas_var)

    def factory(box):
        x0 = np.concatenate([np.array([[zi], [0.0]]) for zi in box[:4]])
        return KalmanFilter(x0, np.eye(8) * 50.0, F, Q, H, R)

    return MultiObjectTracker(factory, n_init=n_init, max_age=max_age, gate=8.0)


def draw(frame, tracks, classes, boxes):
    """Overlay YOLO detections (thin) and confirmed kalbee boxes (colored)."""
    out = frame.copy()
    for (x1, y1, x2, y2), cls in zip(boxes, classes):
        cv2.rectangle(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 255, 255),
            1,
        )
    for t in tracks:
        color = _ID_COLORS[t.id % len(_ID_COLORS)]
        x1, y1, x2, y2 = t.state[[0, 2, 4, 6], 0]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 1.5 or y2 - y1 < 1.5:
            continue
        cv2.rectangle(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )
        label = f"#{t.id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            out,
            (int(x1), max(int(y1) - 16, 0)),
            (int(x1) + tw + 6, max(int(y1) - 16, 0) + th + 6),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (int(x1) + 3, max(int(y1) - 16, 0) + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return out


def run_video(
    source,
    class_ids,
    model_name="yolov8n.pt",
    conf=0.3,
    max_frames=0,
    view=False,
    save=None,
):
    """Run YOLO detection + kalbee tracking over ``source``."""
    from ultralytics import YOLO

    source = str(source)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(
                f"video not found: {source} (use --source 0 for the webcam)"
            )
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    tracker = make_tracker(dt=dt)
    model = YOLO(model_name)

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save, fourcc, fps, (w, h))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if max_frames and n > max_frames:
            break

        result = model(frame, verbose=False, conf=conf)[0]
        boxes, det_classes = [], []
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls in class_ids:
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append(xyxy)
                det_classes.append(cls)
        boxes = (
            np.asarray(boxes, dtype=float).reshape(-1, 4) if boxes else np.zeros((0, 4))
        )

        confirmed = tracker.update(boxes if len(boxes) else None, dt=dt)
        frame = draw(frame, confirmed, det_classes, boxes)
        for t in confirmed:
            print(
                f"frame {n}: track #{t.id} box="
                + np.round(t.state[[0, 2, 4, 6], 0], 1).__str__()
            )

        if view:
            cv2.imshow("kalbee yolo_mot", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"processed {n} frames")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="0", help="video path or camera index")
    parser.add_argument(
        "--classes",
        default=",".join(map(str, DEFAULT_CLASSES)),
        help="comma-separated COCO class ids to track",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--save")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    class_ids = {int(c) for c in args.classes.split(",") if c}
    run_video(
        args.source,
        class_ids,
        model_name=args.model,
        conf=args.conf,
        max_frames=args.max_frames,
        view=args.view,
        save=args.save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
