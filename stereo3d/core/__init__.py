"""
Core data structures and algorithms for autostereogram processing.
"""

from .datastructures import StereogramImage
from .period import (
    PeriodEstimator,
    PeriodEstimationResult,
    estimate_base_period,
)
from .depth import (
    DepthExtractor,
    DepthMapResult,
    extract_depth_map,
)
from .mesh import (
    MeshGenerator,
    MeshResult,
)

__all__ = [
    "StereogramImage",
    "PeriodEstimator",
    "PeriodEstimationResult",
    "estimate_base_period",
    "DepthExtractor",
    "DepthMapResult",
    "extract_depth_map",
    "MeshGenerator",
    "MeshResult",
]
