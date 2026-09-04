import numpy as np

from kalbee import track_to_track_fusion


class TestTrackToTrackFusion:
    """Tests for track-to-track fusion."""

    def test_ci_fusion(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2)
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2

        fused_mean, fused_cov = track_to_track_fusion(
            mean_a, cov_a, mean_b, cov_b, method="ci"
        )

        assert fused_mean.shape == (2, 1)
        assert fused_cov.shape == (2, 2)

    def test_simple_fusion(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2)
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2

        fused_mean, fused_cov = track_to_track_fusion(
            mean_a, cov_a, mean_b, cov_b, method="simple"
        )

        assert fused_mean.shape == (2, 1)

    def test_correlated_fusion(self):
        """Correlated track fusion must execute Bar-Shalom / Campo formula cleanly."""
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2) * 2.0
        mean_b = np.array([[1.5], [2.2]])
        cov_b = np.eye(2) * 3.0
        cross_cov = np.array([[0.5, 0.1], [0.1, 0.4]])

        fused_mean, fused_cov = track_to_track_fusion(
            mean_a, cov_a, mean_b, cov_b, cross_covariance=cross_cov, method="klf"
        )

        assert fused_mean.shape == (2, 1)
        assert fused_cov.shape == (2, 2)
        # Fused covariance trace should be less than or equal to individual trace
        assert np.trace(fused_cov) < np.trace(cov_a)
