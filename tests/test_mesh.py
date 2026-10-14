import numpy as np
import pytest

from stereo3d.core.datastructures import StereogramImage
from stereo3d.core.mesh import MeshGenerator, MeshResult
from stereo3d.utils.synthetic import generate_synthetic_stereogram
from stereo3d.visualization.viewer import find_3d_model


def test_mesh_generator_basic():
    width, height = 200, 150
    stereo_arr, gt_depth = generate_synthetic_stereogram(
        width=width,
        height=height,
        base_period=50,
        shape_type="circle",
        max_disparity=10,
    )
    img = StereogramImage.from_numpy(stereo_arr)

    gen = MeshGenerator(step=2, z_scale=50.0, color_mode="rgb")
    result = gen.generate(gt_depth, stereogram=img)

    expected_hs = 150 // 2
    expected_ws = 200 // 2
    expected_verts = expected_hs * expected_ws
    expected_tris = (expected_hs - 1) * (expected_ws - 1) * 2

    assert result.num_vertices == expected_verts
    assert result.num_triangles == expected_tris
    assert result.vertices.shape == (expected_verts, 3)
    assert result.triangles.shape == (expected_tris, 3)
    assert result.normals is not None
    assert result.normals.shape == (expected_verts, 3)
    assert result.vertex_colors is not None
    assert result.vertex_colors.shape == (expected_verts, 3)
    assert result.uvs is not None
    assert result.uvs.shape == (expected_verts, 2)

    # Validate bounds and sanity
    assert not np.isnan(result.vertices).any()
    assert not np.isnan(result.normals).any()
    min_b, max_b = result.bounds
    assert min_b[2] >= 0.0
    assert max_b[2] <= 50.0 + 1e-4

    # Open3D conversion
    o3d_mesh = result.to_open3d()
    assert len(o3d_mesh.vertices) == expected_verts
    assert len(o3d_mesh.triangles) == expected_tris

    # Trimesh conversion
    tm_mesh = result.to_trimesh()
    assert len(tm_mesh.vertices) == expected_verts
    assert len(tm_mesh.faces) == expected_tris


def test_mesh_file_exports(tmp_path):
    width, height = 100, 80
    _, gt_depth = generate_synthetic_stereogram(
        width=width,
        height=height,
        base_period=30,
        shape_type="box",
        max_disparity=6,
    )

    gen = MeshGenerator(step=2, z_scale=30.0, color_mode="depth")
    result = gen.generate(gt_depth)

    obj_file = tmp_path / "test_model.obj"
    ply_file = tmp_path / "test_model.ply"

    result.save_obj(obj_file)
    result.save_ply(ply_file)

    assert obj_file.is_file()
    assert obj_file.stat().st_size > 0
    assert ply_file.is_file()
    assert ply_file.stat().st_size > 0

    # Test file discovery
    found = find_3d_model("test_model", output_dir=tmp_path)
    assert found == obj_file.resolve()

    found_latest = find_3d_model(None, output_dir=tmp_path)
    assert found_latest.is_file()


def test_mesh_color_modes():
    depth = np.ones((60, 80), dtype=np.float32) * 0.5

    # Depth colormap mode
    gen_depth = MeshGenerator(color_mode="depth", colormap="viridis")
    res_depth = gen_depth.generate(depth)
    assert res_depth.vertex_colors is not None
    assert 0.0 <= res_depth.vertex_colors.min() <= res_depth.vertex_colors.max() <= 1.0

    # Uniform color mode
    gen_uni = MeshGenerator(color_mode="uniform")
    res_uni = gen_uni.generate(depth)
    assert res_uni.vertex_colors is not None
    assert np.allclose(res_uni.vertex_colors, 0.85)
