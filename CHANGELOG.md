# Changelog

All notable changes to kalbee are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-04

Stable public API release. The `predict`/`update` contract on every filter,
the top-level exports in `kalbee/__init__.py`, and the model builders in
`kalbee.models` are considered stable from this version on.

### Added

- Central `kalbee/constants.py` registry — single source of truth for the
  package version and shared numerical/engineering constants
  (`COND_LIMIT`, regularization scales, default initial covariance).
- Animated documentation gallery: `docs/examples.md` with GIFs generated from
  the public API (`scripts/generate_demo_gif.py`).
- Real-video multi-object tracking on the public MOT Challenge sequence
  MOT16-02: `scripts/fetch_mot16_02.py` (HTTP-range downloader),
  `scripts/mot16_demo.py` (GIF renderer), and
  `examples/mot16_pedestrian_tracking.py` (console metrics).
- YOLO bounding-box tracking examples for vehicles and people:
  `examples/yolo_mot.py`, `examples/yolo_vehicles.py`, `examples/yolo_people.py`.
- Intuition-first tutorial `docs/learn.md` (worked 1-D example, math↔API
  mapping, Q/R tuning via NIS/NEES) plus design notes in the docs nav.
- `CHANGELOG.md` and `RELEASING.md` release process.
- Academic paper (`.github/docs/kalbee.tex` → `kalbee.pdf`) presenting the
  toolkit and its design; now part of the repository.

### Changed

- Test suite reorganized to mirror the package (`tests/filters`,
  `tests/smoothers`, `tests/models`, `tests/tracking`, `tests/fusion`,
  `tests/learning`, `tests/integration`, `tests/utils`, `tests/experiments`,
  `tests/cli`); version/feature-named kitchen-sink files split into topical
  tests and duplicate coverage removed.
- Repository identity and metadata unified on `LakoreAI/kalbee`
  (README, pyproject URLs, docs, paper).
- Documentation count/claims reconciled (18 filters), landing-page wording
  updated to the "filtering and tracking toolkit" framing.

### Fixed

- `ruff format --check .` now passes across the whole repository.

## [0.6.0] - 2026

Historical release: 18 filter implementations, smoothers, tracking, learning,
pandas/Polars/scikit-learn integration, CLI, and sensor-fusion cookbook as
described in the project README. See the git history for granular changes.
