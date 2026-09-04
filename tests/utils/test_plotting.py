"""Tests for the matplotlib plotting helpers."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from kalbee.modules.utils.plotting import plot_covariance  # noqa: E402


def test_plot_covariance_individual_states():
    """plot_covariance should return figure and axes without error."""
    covs = [np.eye(2) * 1.0, np.eye(2) * 0.5]
    fig, ax = plot_covariance(covs, save_path=None)
    assert fig is not None
    assert ax is not None
