import numpy as np
import pytest

from stereo3d.core.datastructures import StereogramImage
from stereo3d.core.depth import DepthExtractor, extract_depth_map
from stereo3d.utils.synthetic import generate_synthetic_stereogram


def test_depth_extractor_circle():
    base_period = 100
    max_d = 20
    stereo_arr, gt_depth = generate_synthetic_stereogram(
        width=400,
        height=300,
        base_period=base_period,
        shape_type="circle",
        max_disparity=max_d,
    )
    img = StereogramImage.from_numpy(stereo_arr)

    result = extract_depth_map(img, base_period=base_period, max_disparity=max_d)

    assert result.depth.shape == (300, 400)
    assert result.disparity.shape == (300, 400)
    assert 0.0 <= result.depth.min() <= result.depth.max() <= 1.0

    # Ensure no NaN or infinite values
    assert not np.isnan(result.depth).any()
    assert not np.isinf(result.depth).any()

    # Verify high correlation with ground truth
    corr = np.corrcoef(result.depth.ravel(), gt_depth.ravel())[0, 1]
    assert corr > 0.95


def test_depth_extractor_box():
    base_period = 90
    max_d = 18
    stereo_arr, gt_depth = generate_synthetic_stereogram(
        width=360,
        height=240,
        base_period=base_period,
        shape_type="box",
        max_disparity=max_d,
    )
    img = StereogramImage.from_numpy(stereo_arr)

    result = extract_depth_map(img, base_period=base_period, max_disparity=max_d)

    assert result.depth.shape == (240, 360)
    corr = np.corrcoef(result.depth.ravel(), gt_depth.ravel())[0, 1]
    assert corr > 0.95
