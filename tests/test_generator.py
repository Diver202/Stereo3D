import numpy as np
import pytest

from stereo3d.generators.stereogram_generator import (
    COLOR_PALETTES,
    AutostereogramGenerator,
    generate_autostereogram,
    parse_color_spec,
)


def test_parse_color_spec_presets():
    for name in ["blue", "emerald", "crimson", "amber", "purple", "cyan", "cyberpunk", "sunset", "matrix", "monochrome", "rainbow"]:
        spec = parse_color_spec(name)
        assert "sat_range" in spec
        assert "val_range" in spec


def test_parse_color_spec_hex():
    spec_hex = parse_color_spec("#3498db")
    assert "sat_range" in spec_hex
    assert "val_range" in spec_hex
    assert spec_hex["hue"] is not None


def test_generator_stochastic_vs_seed():
    H, W = 60, 100
    depth = np.zeros((H, W), dtype=np.float32)
    depth[20:40, 30:70] = 0.8

    gen_det1 = AutostereogramGenerator(base_period=25, seed=42, pattern_style="organic")
    res_det1 = gen_det1.generate(depth)

    gen_det2 = AutostereogramGenerator(base_period=25, seed=42, pattern_style="organic")
    res_det2 = gen_det2.generate(depth)

    # Identical seeds must produce identical stereograms
    np.testing.assert_array_equal(res_det1.stereogram, res_det2.stereogram)

    # Stochastic / different seeds should produce different stereograms
    gen_diff = AutostereogramGenerator(base_period=25, seed=999, pattern_style="organic")
    res_diff = gen_diff.generate(depth)
    assert not np.array_equal(res_det1.stereogram, res_diff.stereogram)


def test_generator_patterns():
    H, W = 50, 80
    depth = np.zeros((H, W), dtype=np.float32)
    depth[15:35, 20:60] = 0.7

    for pattern in ["organic", "speckle", "cellular", "plasma"]:
        res = generate_autostereogram(depth, base_period=20, pattern=pattern, seed=123)
        assert res.stereogram.shape == (H, W, 3)
        assert res.stereogram.dtype == np.uint8


def test_generator_file_exports(tmp_path):
    H, W = 40, 60
    depth = np.full((H, W), 0.5, dtype=np.float32)
    gen = AutostereogramGenerator(base_period=15, seed=7)
    res = gen.generate(depth)

    p_img = tmp_path / "test_stereo.png"
    p_analysis = tmp_path / "test_analysis.png"

    res.save_png(p_img)
    res.save_analysis(p_analysis, fusion_dots=True)

    assert p_img.is_file() and p_img.stat().st_size > 0
    assert p_analysis.is_file() and p_analysis.stat().st_size > 0
