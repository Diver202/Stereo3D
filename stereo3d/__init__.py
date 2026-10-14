"""
Stereo3D: A Python framework for autostereogram depth extraction and 3D reconstruction.
"""

from .core.datastructures import StereogramImage
from .core.period import PeriodEstimator, PeriodEstimationResult, estimate_base_period
from .core.depth import DepthExtractor, DepthMapResult, extract_depth_map
from .core.mesh import MeshGenerator, MeshResult

__version__ = "0.1.0"

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
