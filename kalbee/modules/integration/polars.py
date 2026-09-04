from typing import List, Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter


def filter_dataframe(
    df,
    filter_obj: BaseFilter,
    measurement_columns: List[str],
    timestamp_column: Optional[str] = None,
    state_columns: Optional[List[str]] = None,
    covariance: bool = False,
    dt: float = 1.0,
    missing: Optional[float] = None,
):
    """
    Run a Kalman filter on a Polars DataFrame.

    Reads measurement columns, runs predict/update, and returns
    a DataFrame with filtered state estimates.

    Args:
        df: Input Polars DataFrame with measurement data.
        filter_obj: Any BaseFilter instance (KF, EKF, UKF, etc.).
        measurement_columns: Column names to use as measurement vector.
        timestamp_column: Optional column to preserve as identifier.
        state_columns: Names for output state columns. Auto-generated if None.
        covariance: If True, also return covariance trace column.
        dt: Time step for filter predict.
        missing: Value to treat as missing (triggers predict-only). NaN is always treated as missing.

    Returns:
        Polars DataFrame with filtered state estimates.

    Example:
        >>> import polars as pl
        >>> from kalbee import KalmanFilter, filter_dataframe
        >>> df = pl.DataFrame({"t": [0, 1, 2, 3], "x": [1.0, 2.1, 2.9, 4.0]})
        >>> kf = KalmanFilter(state, cov, F, Q, H, R)
        >>> result = filter_dataframe(df, kf, measurement_columns=["x"])
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "Polars is required for filter_dataframe. "
            "Install it with: pip install kalbee[polars]"
        )

    n_state = len(filter_obj.state)

    # Auto-generate state column names
    if state_columns is None:
        state_columns = [f"state_{i}" for i in range(n_state)]

    if len(state_columns) != n_state:
        raise ValueError(
            f"state_columns length ({len(state_columns)}) "
            f"must match state dimension ({n_state})"
        )

    # Convert measurement columns to numpy
    measurements = df.select(measurement_columns).to_numpy()

    # Run filter
    state_history = []
    cov_history = []

    for i in range(len(measurements)):
        z = measurements[i].reshape(-1, 1)

        # Check for missing data
        is_missing = missing is not None and np.any(z == missing)
        is_nan = np.any(np.isnan(z))

        filter_obj.predict(dt=dt)

        if is_missing or is_nan:
            # Predict-only for missing measurements
            pass
        else:
            filter_obj.update(z)

        state_history.append(filter_obj.x.flatten().copy())
        cov_history.append(np.trace(filter_obj.P))

    # Build output DataFrame
    result_data = {}

    # Preserve timestamp column if provided
    if timestamp_column is not None and timestamp_column in df.columns:
        result_data[timestamp_column] = df[timestamp_column]

    # Add state columns
    state_array = np.array(state_history)
    for j, col_name in enumerate(state_columns):
        result_data[col_name] = state_array[:, j]

    # Add covariance trace if requested
    if covariance:
        result_data["cov_trace"] = cov_history

    return pl.DataFrame(result_data)


def filter_series(
    series,
    filter_obj: BaseFilter,
    dt: float = 1.0,
):
    """
    Run a Kalman filter on a single Polars Series (1D measurement).

    Convenience wrapper for single-measurement filtering.

    Args:
        series: Polars Series with measurement values.
        filter_obj: Any BaseFilter instance.
        dt: Time step for filter predict.

    Returns:
        Polars Series with filtered state estimates (first state dimension).

    Example:
        >>> import polars as pl
        >>> from kalbee import KalmanFilter, filter_series
        >>> s = pl.Series("x", [1.0, 2.1, 2.9, 4.0, 5.1])
        >>> kf = KalmanFilter(state, cov, F, Q, H, R)
        >>> result = filter_series(s, kf)
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "Polars is required for filter_series. "
            "Install it with: pip install kalbee[polars]"
        )

    measurements = series.to_numpy()
    state_history = []

    for z in measurements:
        z_vec = np.array([[z]])
        filter_obj.predict(dt=dt)
        filter_obj.update(z_vec)
        state_history.append(filter_obj.x[0, 0])

    return pl.Series(name=f"filtered_{series.name}", values=state_history)
