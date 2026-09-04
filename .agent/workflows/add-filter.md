---
description: Steps to implement a new filter type in the kalbee library.
---

# Add New Filter Workflow

Follow these steps to add a new filter (e.g., EKF, UKF, Particle Filter) to the `kalbee` project.

1. **Create the filter file**:
   - Create a new file in `kalbee/modules/filters/` named after the filter (e.g., `ekf_filter.py`).
   - Import `BaseFilter` and `numpy`.

2. **Implement the class**:
   - Inherit from `BaseFilter`.
   - Implement `__init__`, `predict(self, dt, **kwargs)`, and `update(self, measurement, **kwargs)`.

3. **Register the filter**:
   - Add the filter class to `kalbee/modules/filters/__init__.py` and include it in `__all__`.

4. **Add unit tests**:
   - Create a corresponding test file in `tests/test_<filter_name>.py`.
   - Include tests for:
     - Initialization.
     - Prediction step (verify math).
     - Update step (verify convergence).

5. **Verify everything**:
// turbo
   - Run the tests:
     ```bash
     uv run python -m pytest tests/
     ```
   - Check linting:
     ```bash
     uv run ruff check .
     ```
