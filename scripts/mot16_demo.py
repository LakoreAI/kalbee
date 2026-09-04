"""
Multi-object tracking demo on *real* MOT Challenge footage.

Loads a public MOT16/17-style sequence (frames + ``gt/det.txt``), runs the
detections through kalbee's SORT-style :class:`~kalbee.MultiObjectTracker`,
and renders the confirmed track boxes (with stable track IDs) over the real
frames as an animated GIF.

The default sequence is ``MOT16-02`` (a pedestrian street-crossing scene).
Obtain it with ``scripts/fetch_mot16_02.py`` (downloads only the needed
frames + ground truth by HTTP range):

    uv run python scripts/fetch_mot16_02.py            # -> data/MOT16-02
    uv run python scripts/mot16_demo.py --gif          # -> docs/assets/gif/mot16_tracking.gif
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from kalbee import KalmanFilter, MultiObjectTracker
from kalbee.models import constant_velocity, position_measurement_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "MOT16-02")
DEFAULT_GIF = os.path.join(REPO_ROOT, "docs", "assets", "gif", "mot16_tracking.gif")

ID_COLORS = [
    (214, 39, 40),
    (31, 119, 180),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
    (23, 190, 207),
    (227, 119, 194),
    (188, 189, 34),
    (140, 86, 75),
    (250, 187, 108),
]


def load_detections(gt_path, max_frame):
    """Parse a MOT ``gt.txt`` into {frame: (N, 4) box array}."""
    frames = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            fr = int(parts[0])
            conf = float(parts[6])
            w, h = float(parts[4]), float(parts[5])
            if fr > max_frame or conf <= 0.0 or w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = (
                float(parts[2]),
                float(parts[3]),
                float(parts[2]) + w,
                float(parts[3]) + h,
            )
            frames.setdefault(fr, []).append([x1, y1, x2, y2])
    return {fr: np.asarray(b, dtype=float) for fr, b in frames.items()}


def make_tracker():
    """SORT-style tracker whose state holds four independent CV corners.

    State layout (block-per-axis, kalbee convention):
    [x1, vx1, y1, vy1, x2, vx2, y2, vy2]; measurements are [x1, y1, x2, y2].
    """
    F, Q = constant_velocity(dt=1.0, process_var=1.0, n_dims=4)
    H, R = position_measurement_model(order=1, n_dims=4, measurement_var=6.0**2)

    def factory(z):
        x0 = np.concatenate([np.array([[zi], [0.0]]) for zi in z[:4]])
        return KalmanFilter(x0, np.eye(8) * 50.0, F, Q, H, R)

    return MultiObjectTracker(factory, n_init=2, max_age=30, gate=8.0)


def run_tracking(detections, n_frames):
    """Run the tracker over the detection sequence."""
    tracker = make_tracker()
    per_frame = []
    for fr in range(1, n_frames + 1):
        z = detections.get(fr)
        confirmed = tracker.update(z if z is not None else None, dt=1.0)
        per_frame.append([(t.id, t.state[[0, 2, 4, 6], 0].copy()) for t in confirmed])
    return per_frame


def build_gif(data_dir, out_path, n_frames, width=560, step=4, colors=160):
    detections = load_detections(
        os.path.join(data_dir, "gt", "gt.txt"), max_frame=n_frames
    )
    frame_ids = list(range(1, n_frames + 1, step))
    tracks = run_tracking(detections, n_frames)

    images = []
    for fr in frame_ids:
        img = Image.open(os.path.join(data_dir, "img1", f"{fr:06d}.jpg"))
        scale = width / img.width
        img = img.resize((width, int(img.height * scale)), Image.LANCZOS)
        draw = ImageDraw.Draw(img, "RGBA")

        # raw detections: thin translucent boxes
        raw = detections.get(fr)
        if raw is not None:
            for x1, y1, x2, y2 in raw:
                draw.rectangle(
                    [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                    outline=(255, 255, 255, 90),
                    width=1,
                )

        # confirmed kalbee tracks: colored boxes + stable IDs
        for tid, box in tracks[fr - 1]:
            color = ID_COLORS[tid % len(ID_COLORS)]
            box = np.asarray(box, dtype=float)
            if not np.all(np.isfinite(box)):
                continue
            x1, y1, x2, y2 = box
            # Corner states drift independently during occlusions; order the
            # corners for drawing and skip degenerate boxes.
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < 1.5 or y2 - y1 < 1.5:
                continue
            x1, y1, x2, y2 = (int(v * scale) for v in (x1, y1, x2, y2))
            label_y = max(y1, 2)
            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
            draw.rectangle([x1, label_y - 16, x1 + 52, label_y], fill=color + (255,))
            draw.text((x1 + 3, label_y - 15), f"#{tid}", fill=(255, 255, 255, 255))

        images.append(img.convert("P", palette=Image.ADAPTIVE, colors=colors))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out_path} ({len(images)} frames, step {step})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", default=DEFAULT_DATA, help="sequence dir with img1/ and gt/gt.txt"
    )
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument(
        "--gif", action="store_true", help="write docs/assets/gif/mot16_tracking.gif"
    )
    parser.add_argument("--out", default=DEFAULT_GIF)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args(argv)

    if not os.path.isdir(os.path.join(args.data, "img1")):
        print(
            f"Sequence not found at {args.data}.\n"
            "Run first:  uv run python scripts/fetch_mot16_02.py",
            file=sys.stderr,
        )
        return 1

    if args.gif:
        build_gif(
            args.data,
            args.out,
            n_frames=args.frames,
            width=args.width,
            step=args.step,
        )
    else:
        detections = load_detections(
            os.path.join(args.data, "gt", "gt.txt"), max_frame=args.frames
        )
        tracks = run_tracking(detections, args.frames)
        n_confirmed = [len(t) for t in tracks]
        print(f"frames processed: {args.frames}")
        print(f"confirmed tracks at end of clip: {n_confirmed[-1]}")
        print(f"avg confirmed tracks / frame: {np.mean(n_confirmed):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
