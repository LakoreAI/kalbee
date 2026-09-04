"""
Track *people* in a real video, bounding boxes handled by kalbee's
MultiObjectTracker (SORT-style, gated association keeps each person's ID).

Thin preset over ``examples/yolo_mot.py`` restricted to the person class::

    python examples/yolo_people.py --source path/to/walkway.mp4 --view
    python examples/yolo_people.py --source 0 --max-frames 300 --save out_people.mp4

Requires ``pip install "kalbee[yolo]"``.
"""

from yolo_mot import build_parser, run_video


def main(argv=None):
    parser = build_parser()
    parser.set_defaults(classes="0")  # COCO class 0 = person
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
