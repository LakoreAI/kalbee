import numpy as np

from kalbee import KalmanFilter
from kalbee.models import constant_velocity, position_measurement_model
from kalbee.tracking import iou_matrix, associate, MultiObjectTracker


# --- Association primitives ---


def test_iou_identical_and_disjoint():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    same = np.array([[0.0, 0.0, 10.0, 10.0]])
    disjoint = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert np.isclose(iou_matrix(boxes, same)[0, 0], 1.0)
    assert np.isclose(iou_matrix(boxes, disjoint)[0, 0], 0.0)


def test_iou_half_overlap():
    a = np.array([[0.0, 0.0, 2.0, 2.0]])  # area 4
    b = np.array([[1.0, 0.0, 3.0, 2.0]])  # area 4, overlap area 2
    # IoU = 2 / (4 + 4 - 2) = 1/3
    assert np.isclose(iou_matrix(a, b)[0, 0], 1.0 / 3.0)


def test_associate_matches_and_gates():
    # Track 0 clearly matches det 1, track 1 matches det 0; a third det is spare.
    cost = np.array(
        [
            [9.0, 0.1, 9.0],
            [0.2, 9.0, 9.0],
        ]
    )
    matches, un_tracks, un_dets = associate(cost, max_cost=1.0)
    assert set(matches) == {(0, 1), (1, 0)}
    assert un_tracks == []
    assert un_dets == [2]


def test_associate_empty():
    matches, un_tracks, un_dets = associate(np.zeros((0, 3)), max_cost=1.0)
    assert matches == []
    assert un_tracks == []
    assert un_dets == [0, 1, 2]


# --- End-to-end multi-object tracking ---


def _make_factory():
    F, Q = constant_velocity(dt=1.0, process_var=0.1, n_dims=2)
    H, R = position_measurement_model(order=1, n_dims=2, measurement_var=0.25)

    def factory(z):
        x0 = np.array([[z[0]], [0.0], [z[1]], [0.0]])
        return KalmanFilter(x0, np.eye(4) * 10.0, F, Q, H, R)

    return factory


def test_tracker_confirms_two_targets():
    rng = np.random.default_rng(0)
    tracker = MultiObjectTracker(_make_factory(), n_init=3, max_age=5, gate=6.0)

    # Two targets moving with constant velocity.
    p1 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.5])
    p2 = np.array([20.0, 10.0])
    v2 = np.array([-0.5, 1.0])

    confirmed = []
    for k in range(8):
        p1 = p1 + v1
        p2 = p2 + v2
        dets = np.stack(
            [
                p1 + rng.standard_normal(2) * 0.1,
                p2 + rng.standard_normal(2) * 0.1,
            ]
        )
        confirmed = tracker.update(dets)

    assert len(confirmed) == 2
    # Estimated positions should be close to the two ground-truth targets.
    est = sorted(
        [(t.state[0, 0], t.state[2, 0]) for t in confirmed], key=lambda e: e[0]
    )
    truth = sorted([tuple(p1), tuple(p2)], key=lambda e: e[0])
    for e, tr in zip(est, truth):
        assert np.allclose(e, tr, atol=1.5)


def test_tracker_deletes_disappeared_target():
    tracker = MultiObjectTracker(_make_factory(), n_init=2, max_age=3, gate=6.0)

    p = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    # Confirm one target.
    for _ in range(3):
        p = p + v
        tracker.update(p.reshape(1, 2))
    assert len(tracker.confirmed_tracks()) == 1

    # Now feed empty detections; after max_age misses the track is deleted.
    for _ in range(5):
        tracker.update(np.zeros((0, 2)))
    assert len(tracker.tracks) == 0


def test_tracker_ids_are_stable():
    tracker = MultiObjectTracker(_make_factory(), n_init=2, max_age=5, gate=6.0)
    p = np.array([0.0, 0.0])
    v = np.array([0.7, 0.7])
    ids = set()
    for _ in range(6):
        p = p + v
        confirmed = tracker.update(p.reshape(1, 2))
        ids.update(t.id for t in confirmed)
    # A single continuously-tracked target must keep exactly one id.
    assert len(ids) == 1
