import numpy as np
import pytest
import polars as pl

from kalbee import KalmanFilter, filter_dataframe, filter_series
from kalbee.models import constant_velocity, position_measurement_model


@pytest.fixture
def setup_filter():
    """Setup a simple Kalman filter for testing."""
    F, Q = constant_velocity(dt=1.0, process_var=0.01, n_dims=1)
    H, R = position_measurement_model(order=1, n_dims=1, measurement_var=0.5)

    state = np.zeros((2, 1))
    cov = np.eye(2) * 10.0

    return KalmanFilter(state, cov, F, Q, H, R)


class TestFilterDataframe:
    """Tests for filter_dataframe."""

    def test_basic_filtering(self, setup_filter):
        df = pl.DataFrame({
            "t": [0, 1, 2, 3, 4],
            "x": [1.0, 2.1, 2.9, 4.0, 5.1],
        })

        result = filter_dataframe(
            df, setup_filter,
            measurement_columns=["x"],
            timestamp_column="t",
        )

        assert isinstance(result, pl.DataFrame)
        assert "t" in result.columns
        assert "state_0" in result.columns
        assert "state_1" in result.columns
        assert len(result) == 5

    def test_without_timestamp(self, setup_filter):
        df = pl.DataFrame({
            "x": [1.0, 2.1, 2.9, 4.0, 5.1],
        })

        result = filter_dataframe(
            df, setup_filter,
            measurement_columns=["x"],
        )

        assert "t" not in result.columns
        assert "state_0" in result.columns
        assert len(result) == 5

    def test_custom_state_columns(self, setup_filter):
        df = pl.DataFrame({
            "x": [1.0, 2.1, 2.9, 4.0, 5.1],
        })

        result = filter_dataframe(
            df, setup_filter,
            measurement_columns=["x"],
            state_columns=["position", "velocity"],
        )

        assert "position" in result.columns
        assert "velocity" in result.columns
        assert "state_0" not in result.columns

    def test_with_covariance(self, setup_filter):
        df = pl.DataFrame({
            "x": [1.0, 2.1, 2.9, 4.0, 5.1],
        })

        result = filter_dataframe(
            df, setup_filter,
            measurement_columns=["x"],
            covariance=True,
        )

        assert "cov_trace" in result.columns

    def test_missing_data(self, setup_filter):
        df = pl.DataFrame({
            "t": [0, 1, 2, 3, 4],
            "x": [1.0, float("nan"), 2.9, 4.0, 5.1],
        })

        result = filter_dataframe(
            df, setup_filter,
            measurement_columns=["x"],
            timestamp_column="t",
            missing=float("nan"),
        )

        assert len(result) == 5

    def test_multi_measurement(self):
        """Test with multiple measurement columns."""
        # Create a filter with 2D measurements
        F = np.array([[1, 1, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 1],
                       [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01
        H = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0]])
        R = np.eye(2) * 0.5

        state = np.zeros((4, 1))
        cov = np.eye(4) * 10.0
        kf = KalmanFilter(state, cov, F, Q, H, R)

        df = pl.DataFrame({
            "x": [1.0, 2.1, 2.9],
            "y": [0.5, 1.2, 1.8],
        })

        result = filter_dataframe(
            df, kf,
            measurement_columns=["x", "y"],
            state_columns=["x", "vx", "y", "vy"],
        )

        assert len(result) == 3
        assert "x" in result.columns
        assert "y" in result.columns

    def test_state_columns_length_mismatch(self, setup_filter):
        df = pl.DataFrame({
            "x": [1.0, 2.1, 2.9],
        })

        with pytest.raises(ValueError, match="state_columns length"):
            filter_dataframe(
                df, setup_filter,
                measurement_columns=["x"],
                state_columns=["pos"],  # Wrong: need 2 for 2D state
            )


class TestFilterSeries:
    """Tests for filter_series."""

    def test_basic_filtering(self, setup_filter):
        s = pl.Series("x", [1.0, 2.1, 2.9, 4.0, 5.1])

        result = filter_series(s, setup_filter)

        assert isinstance(result, pl.Series)
        assert result.name == "filtered_x"
        assert len(result) == 5

    def test_values_reasonable(self, setup_filter):
        s = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])

        result = filter_series(s, setup_filter)

        # Filtered values should be close to input for linear motion
        values = result.to_list()
        assert all(abs(v - i - 1) < 1.0 for i, v in enumerate(values))


class TestIntegration:
    """Integration tests with realistic data."""

    def test_tracking_workflow(self):
        """Full workflow: generate data, filter, check results."""
        # Setup filter
        F, Q = constant_velocity(dt=1.0, process_var=0.01, n_dims=1)
        H, R = position_measurement_model(order=1, n_dims=1, measurement_var=0.25)

        state = np.zeros((2, 1))
        cov = np.eye(2) * 10.0
        kf = KalmanFilter(state, cov, F, Q, H, R)

        # Generate synthetic data
        np.random.seed(42)
        true_positions = np.arange(0, 10, 0.5)
        noisy_measurements = true_positions + np.random.randn(len(true_positions)) * 0.5

        df = pl.DataFrame({
            "time": range(len(noisy_measurements)),
            "position": noisy_measurements,
        })

        # Filter
        result = filter_dataframe(
            df, kf,
            measurement_columns=["position"],
            timestamp_column="time",
            state_columns=["est_position", "est_velocity"],
            covariance=True,
        )

        # Check results
        assert len(result) == len(noisy_measurements)
        assert result["est_position"][0] is not None
        assert result["est_velocity"][0] is not None
        assert result["cov_trace"][0] > 0
