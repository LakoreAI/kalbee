"""
Track *vehicles* (bicycle / car / motorcycle / bus / truck) in a real video,
bounding boxes handled by kalbee's MultiObjectTracker.

Thin preset over ``examples/yolo_mot.py`` restricted to vehicle COCO classes::

    python examples/yolo_vehicles.py --source path/to/traffic.mp4 --view
    python examples/yolo_vehicles.py --source 0 --save out_vehicles.mp4

Requires ``pip install "kalbee[yolo]"``.
"""

from yolo_mot import build_parser, run_video

VEHICLE_CLASSES = "1,2,3,5,7"  # bicycle, car, motorcycle, bus, truck


def main(argv=None):
    parser = build_parser()
    parser.set_defaults(classes=VEHICLE_CLASSES)
    args = parser.parse_args(argv)

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
