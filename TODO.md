# TODOs and Roadmap

## Completed Work
- [x] **Unscented Kalman Filter (UKF)**: Implemented - handles highly non-linear systems without Jacobians.
- [x] **Particle Filter**: Implemented - Sequential Monte Carlo for non-Gaussian distributions.
- [x] **Ensemble Kalman Filter (EnKF)**: Implemented - ensemble-based for high-dimensional systems.
- [x] **Information Filter**: Implemented - dual of KF, ideal for multi-sensor fusion.
- [x] **Smoothing**: Implemented RTS (Rauch-Tung-Striebel) smoothing for post-processing.
- [x] **Adaptive Filtering**: Implemented adaptive noise estimation (estimating Q and R on the fly).
- [x] **Joseph Form**: KalmanFilter and EKF use Joseph form for numerically stable covariance updates.
- [x] **Filter Metrics**: RMSE, NEES, NIS, and log-likelihood for filter diagnostics.
- [x] **Square-Root Kalman Filter (SRKF)**: Implemented - Cholesky factor-based covariance updates directly on the square root.
- [x] **Interacting Multiple Model (IMM)**: Implemented - switching/blending estimator combining multiple model hypotheses.
- [x] **NumPy 2.x Compatibility**: Fixed scalar conversion TypeError in metrics.py using `.item()`.
- [x] **Singular Matrix Fallbacks**: Implemented robust fallbacks (using regularization and pseudoinverse) for matrix inversions across all filters.
- [x] **Benchmarks**: Developed a performance benchmarking suite to compare all filter implementations.
- [x] **Documentation & API Reference**: Configured navigation and added detailed documentation for SRKF and IMM filters.
- [x] **Fading Memory Kalman Filter**: Implemented - discounted covariance inflation for maneuvering targets.
- [x] **Innovation Gating**: Chi-squared and Mahalanobis gating for outlier rejection.
- [x] **Control Input Support**: Added B matrix for control inputs in KF predict step.
- [x] **Batch Processing**: Added `filter_sequence()` for processing measurement sequences with missing data handling.
- [x] **State Serialization**: Added `save_state()` / `load_state()` for filter persistence.
- [x] **AutoFilter Expansion**: Added Fading Memory KF to factory with fmkf/fading aliases.
- [x] **H-Infinity Filter**: Implemented - robust filter for worst-case estimation scenarios.
- [x] **Sigma Point Options**: Added SimplexSigmaPoints, MerweScaledSigmaPoints, JulierSigmaPoints for UKF customization.
- [x] **Consistency Tests**: NIS test, NEES test, and innovation whiteness test for filter validation.
- [x] **Auto-Tuning**: NIS-based automatic Q/R tuning for Kalman filters.
- [x] **Quick Tune**: Fast single-pass tuning based on innovation statistics.

## Planned Enhancements

### Robustness and Numerical Stability
- [x] **Cholesky Factor Stabilization**:
  - Implement additional checks to keep Cholesky factors symmetric and positive-definite under highly chaotic measurements.

### Advanced Estimation Algorithms
- [x] **Other Models**:
  - Extend models supported by default in experiments.

### Engineering and Quality
- [x] **Type Hints**:
  - Add explicit types across all function signatures, specifically targeting matrix/array dimensions.
- [x] **Test Coverage**:
  - Achieve 90%+ test coverage. Add specific edge cases for singular/badly-conditioned inputs.
- [x] **Vectorization**:
  - Optimize filters to process batches of states and measurements.

### Documentation
- [x] **Tutorials**:
  - Create Jupyter notebooks for tracking simulations (e.g. 2D target tracking, robot localization).

## New Features
- [x] **SigmaPointUKF**: UKF class that accepts sigma point strategy objects (MerweScaled, Julier, etc.)
- [x] **Filter Diagnostics**: FilterDiagnostics class for real-time monitoring and reporting
- [x] **Chi-squared Outlier Detector**: Real-time outlier detection using NIS with configurable thresholds
- [x] **Predict-only step**: `predict_only()` method for pure prediction without measurement
- [x] **Filter reset**: `reset()` method to reinitialize filter state without recreating object

## Feature Suggestions for v0.6.0

Based on comparison with filterpy, pykalman, and recent research papers:

### New Filters
- [x] **Cubature Kalman Filter (CKF)**: Third-order cubature rule for high-dimensional nonlinear systems. Better scaling than UKF for n > 5.
- [x] **Extended Kalman Smoother**: RTS smoother for EKF (currently only works with linear KF).
- [x] **Fixed-Lag Smoother**: Real-time smoothing with bounded delay, suitable for online applications.
- [x] **Covariance Intersection (CI)**: Fusing estimates with unknown correlations — critical for distributed multi-sensor systems.
- [x] **Federated Kalman Filter**: Hierarchical fusion for large-scale distributed systems (GPS/INS integration).
- [x] **Rao-Blackwellized Particle Filter**: Marginalize out linear states for reduced variance in partially linear systems.

### Numerical Improvements
- [x] **Cholesky-Based KF**: Work directly with Cholesky factors of P instead of P itself (20% slower but far more stable).
- [x] **Square-Root UKF**: Cholesky-based UKF for better numerical properties.
- [x] **Square-Root EKF**: Cholesky-based EKF for improved stability.

### Functional Interface
- [x] **Procedural API**: Standalone `predict(x, P, F, Q)` and `update(x, P, z, R, H)` functions alongside OOP interface (like filterpy).
- [x] **Typed State Classes**: TypedDict or dataclass for filter state snapshots.

### Tracking & Fusion
- [x] **Track-to-Track Fusion**: Fusion of pre-existing tracks from multiple sensors (covariance intersection).

### Learning & Adaptation
- [x] **Online EM**: Streaming EM for continuous Q/R adaptation without storing full history.

### Engineering
- [x] **Async/Streaming Interface**: `async def predict()` / `async def update()` for real-time sensor streams.
- [x] **JSON Schema for Filter Config**: Export/import filter configuration as JSON for reproducibility.
- [x] **Matplotlib Plotting Helpers**: Built-in `plot_trajectory()`, `plot_covariance()`, `plot_innovations()` utilities.
- [x] **Integration with Polars**: `filter_dataframe()` and `filter_series()` for processing DataFrames.
- [x] **Integration with pandas**: `filter_dataframe()` and `filter_series()` for processing DataFrames.

## Adoption & DX (2026-09)

- [x] **Reproducible benchmarks vs. FilterPy/pykalman/simdkalman**: `scripts/compare_benchmarks.py` + `docs/benchmarks.md`. Honest result: matches FilterPy/pykalman on RMSE exactly, `VectorizedKalmanFilter` beats simdkalman ~4x on batched series, FilterPy still has less per-call overhead for a single bare filter loop.
- [x] **Sensor-fusion cookbook**: quaternion attitude EKF (`kalbee.models.attitude`) and GPS+IMU loosely-coupled fusion (`imu_velocity_control`), with runnable examples and docs.
- [x] **Numerical Jacobians**: `numerical_jacobian()`/`numerical_transition_jacobian()`/`numerical_measurement_jacobian()` — build EKF Jacobians from plain functions via finite differences.
- [x] **scikit-learn integration**: `KalmanEstimator` (`fit`/`transform`/`predict`, `Pipeline`-compatible). Optional `kalbee[sklearn]`.
- [x] **`kalbee` CLI**: `demo` (with an animated `--live` terminal chart via `rich`), `bench`, `new` (scaffolding). Optional `kalbee[cli]`.
- [x] **`py.typed` + mypy config**: package ships a `py.typed` marker; `[tool.mypy]` configured in `pyproject.toml`. New modules (`cli.py`, `models/attitude.py`, `modules/integration/sklearn_api.py`, `modules/utils/jacobian.py`) type-check cleanly. Scope note: this does **not** retrofit strict typing across the ~30 pre-existing filter modules (mostly `Optional[np.ndarray]` matrix attributes on `BaseFilter` not being narrowed) — that's a larger, separate effort tracked below.

### Not done / explicitly deferred
- [ ] **Full strict-typing retrofit** of pre-existing filter/smoother modules (`mypy --strict` currently reports ~85 pre-existing errors, none in new code).
- [ ] **JAX autodiff/GPU backend**: gradient-based Q/R tuning and `vmap`-parallel filtering across many tracks. Bigger lift than the items above (a parallel backend, not a single feature) — deferred pending demand.
- [ ] **Magnetometer update** for the quaternion attitude EKF (would make yaw observable too; noted as a known gap in `kalbee.models.attitude`).
