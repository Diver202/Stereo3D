import numpy as np
import pytest

from stereo3d.core.datastructures import StereogramImage
from stereo3d.core.period import PeriodEstimator, estimate_base_period
from stereo3d.utils.synthetic import generate_synthetic_stereogram


def test_stereogram_image_structure():
    arr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    img = StereogramImage.from_numpy(arr)

    assert img.height == 200
    assert img.width == 300
    assert img.shape == (200, 300)
    assert img.gray.shape == (200, 300)
    assert 0.0 <= img.gray.min() <= img.gray.max() <= 1.0

    # Test difference map
    diff = img.compute_difference_map(lag=50)
    assert diff.shape == (200, 250)


def test_synthetic_autocorrelation_period_estimation():
    # Ground truth period = 100 px
    gt_period = 100
    stereo_arr, _ = generate_synthetic_stereogram(
        width=500,
        height=300,
        base_period=gt_period,
        shape_type="circle",
    )
    img = StereogramImage.from_numpy(stereo_arr)

    result = estimate_base_period(
        img,
        method="autocorrelation",
        sampling="distributed",
        min_period=30,
        max_period=200,
    )

    assert result.period == gt_period
    assert result.confidence > 0.3


def test_synthetic_difference_period_estimation():
    # Ground truth period = 80 px (Photoshop difference method)
    gt_period = 80
    stereo_arr, _ = generate_synthetic_stereogram(
        width=480,
        height=320,
        base_period=gt_period,
        shape_type="box",
    )
    img = StereogramImage.from_numpy(stereo_arr)

    result = estimate_base_period(
        img,
        method="difference",
        sampling="distributed",
        min_period=30,
        max_period=180,
    )

    assert result.period == gt_period
