"""
Core data structures and algorithms for autostereogram processing.
"""

from .datastructures import StereogramImage
from .period import (
    PeriodEstimator,
    PeriodEstimationResult,
    estimate_base_period,
)

__all__ = [
    "StereogramImage",
    "PeriodEstimator",
    "PeriodEstimationResult",
    "estimate_base_period",
]
