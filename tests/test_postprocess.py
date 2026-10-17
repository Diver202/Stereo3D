import numpy as np
import pytest

from stereo3d.core.datastructures import StereogramImage
from stereo3d.core.postprocess import DepthPostprocessor, postprocess_depth_map


def test_postprocess_side_margin_drop():
    H, W = 100, 200
    base_period = 40
    # Dummy depth map
    depth = np.full((H, W), 0.5, dtype=np.float32)
    # Put artificial border bars on columns 0..20 and 180..200
    depth[:, :20] = 0.9
    depth[:, 180:] = 0.9

    postprocessor = DepthPostprocessor(
        drop_side_margins=True,
        margin_factor=0.5,
    )
    res, _ = postprocessor.postprocess(depth, base_period=base_period)

    # Margin left = 20, margin right = 20
    assert res.cropped_margins == (20, 20)
    assert res.width == 160
    assert res.height == 100
    # Ensure values in cropped region are valid
    assert 0.0 <= res.depth.min() <= res.depth.max() <= 1.0


def test_postprocess_spike_removal():
    H, W = 120, 160
    base_period = 30
    depth = np.zeros((H, W), dtype=np.float32)
    # Add a flat elevated circle in the center
    yy, xx = np.ogrid[:H, :W]
    mask = ((xx - 80) ** 2 + (yy - 60) ** 2) < 30 ** 2
    depth[mask] = 0.7

    # Inject isolated salt-and-pepper noise spikes
    rng = np.random.default_rng(42)
    noise_coords_y = rng.integers(10, H - 10, size=25)
    noise_coords_x = rng.integers(30, W - 30, size=25)
    depth[noise_coords_y, noise_coords_x] = 1.0

    postprocessor = DepthPostprocessor(
        drop_side_margins=False,
        diff_thresholds=(0.10, 0.15, 0.20),
    )
    res, _ = postprocessor.postprocess(depth, base_period=base_period)

    # Check that spikes were flagged and inpainted
    assert res.spike_percentage > 0.0
    # Center of spikes should have been smoothed/inpainted down towards background or circle
    for y, x in zip(noise_coords_y, noise_coords_x):
        if not mask[y, x]:
            # Should have been smoothed back close to background (0.0)
            assert res.depth[y, x] < 0.25


def test_postprocess_isolated_splotch_filtering():
    H, W = 100, 150
    base_period = 20
    depth = np.zeros((H, W), dtype=np.float32)

    # Main object: area 1200 pixels
    depth[20:50, 40:80] = 0.8

    # Small isolated splotch: area 40 pixels
    depth[70:75, 20:28] = 0.9

    postprocessor = DepthPostprocessor(
        drop_side_margins=False,
        min_island_area=100,
    )
    res, _ = postprocessor.postprocess(depth, base_period=base_period)

    # The small splotch should have been marked in the mask and inpainted away
    assert res.spike_mask[72, 24] > 0
    assert res.depth[72, 24] < 0.2
    # The main object should remain elevated
    assert res.depth[35, 60] > 0.6


def test_postprocess_file_exports(tmp_path):
    H, W = 60, 80
    depth = np.full((H, W), 0.5, dtype=np.float32)
    postprocessor = DepthPostprocessor(drop_side_margins=False)
    res, _ = postprocessor.postprocess(depth, base_period=20)

    p_png = tmp_path / "test_depth.png"
    p_16bit = tmp_path / "test_16bit.png"
    p_color = tmp_path / "test_color.png"
    p_npy = tmp_path / "test.npy"
    p_mask = tmp_path / "test_mask.png"

    res.save_png(p_png)
    res.save_16bit(p_16bit)
    res.save_color(p_color)
    res.save_numpy(p_npy)
    res.save_mask(p_mask)

    assert p_png.is_file() and p_png.stat().st_size > 0
    assert p_16bit.is_file() and p_16bit.stat().st_size > 0
    assert p_color.is_file() and p_color.stat().st_size > 0
    assert p_npy.is_file() and p_npy.stat().st_size > 0
    assert p_mask.is_file() and p_mask.stat().st_size > 0
