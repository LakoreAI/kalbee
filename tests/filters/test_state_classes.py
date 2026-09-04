import numpy as np

from kalbee import FilterConfig, FilterState


class TestFilterState:
    """Tests for FilterState dataclass."""

    def test_basic_creation(self):
        state = FilterState(
            state_mean=np.array([[1.0], [2.0]]),
            state_covariance=np.eye(2),
        )

        assert state.state_dim == 2
        assert state.covariance_trace == 2.0
        assert state.state_std.shape == (2,)

    def test_serialization(self):
        state = FilterState(
            state_mean=np.array([[1.0], [2.0]]),
            state_covariance=np.eye(2) * 0.5,
            timestamp=100,
        )

        json_str = state.to_json()
        restored = FilterState.from_json(json_str)

        np.testing.assert_array_equal(state.state_mean, restored.state_mean)
        assert state.timestamp == restored.timestamp


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_basic_creation(self):
        config = FilterConfig(
            filter_type="kf",
            state_dim=2,
            measurement_dim=1,
            transition_matrix=np.array([[1, 1], [0, 1]]),
        )

        assert config.filter_type == "kf"
        assert config.state_dim == 2

    def test_serialization(self):
        config = FilterConfig(
            filter_type="kf",
            state_dim=2,
            measurement_dim=1,
            transition_matrix=np.array([[1, 1], [0, 1]]),
            transition_covariance=np.eye(2) * 0.01,
        )

        json_str = config.to_json()
        restored = FilterConfig.from_json(json_str)

        assert config.filter_type == restored.filter_type
        np.testing.assert_array_equal(
            config.transition_matrix, restored.transition_matrix
        )
