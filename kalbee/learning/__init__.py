"""Offline parameter learning for state-space models."""

from kalbee.learning.em import em_kalman, EMResult

__all__ = [
    "em_kalman",
    "EMResult",
]
