from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def find_3d_model(
    model_arg: Optional[str] = None,
    output_dir: Union[str, Path] = "output",
) -> Path:
    """
    Locate a 3D model file from an explicit path, stem name, or latest output file.
    """
    out_path = Path(output_dir)

    if model_arg:
        p = Path(model_arg)
        # 1. Direct path
        if p.is_file():
            return p.resolve()

        # 2. Inside output directory
        cand = out_path / model_arg
        if cand.is_file():
            return cand.resolve()

        # 3. Try with suffixes
        for suffix in [".obj", ".ply", "_mesh.obj", "_mesh.ply"]:
            cand = out_path / f"{model_arg}{suffix}"
            if cand.is_file():
                return cand.resolve()
            cand_direct = Path(f"{model_arg}{suffix}")
            if cand_direct.is_file():
                return cand_direct.resolve()

        raise FileNotFoundError(
            f"3D model '{model_arg}' not found. Searched at '{p.resolve()}' "
            f"and inside '{out_path.resolve()}'."
        )

    # If no argument provided: search output/ for newest .obj or .ply
    if not out_path.is_dir():
        raise FileNotFoundError(f"Output directory '{out_path.resolve()}' does not exist.")

    candidates: List[Path] = []
    for ext in ["*.obj", "*.ply"]:
        candidates.extend(out_path.glob(ext))

    if not candidates:
        raise FileNotFoundError(
            f"No 3D models (*.obj, *.ply) found in '{out_path.resolve()}'. "
            "Run 'python main.py <image>' first to reconstruct a 3D object."
        )

    # Sort by modification time, newest first
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def launch_viewer(
    filepath: Union[str, Path],
    window_name: Optional[str] = None,
    width: int = 1280,
    height: int = 800,
) -> int:
    """
    Launch native interactive 3D visualizer for the specified mesh file.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    title = window_name or f"Stereo3D Mesh Viewer - {path.name}"

    if HAS_OPEN3D:
        try:
            return _launch_open3d_viewer(path, window_name=title, width=width, height=height)
        except Exception as e:
            print(f"Open3D window initialization failed ({e}). Attempting fallback...", file=sys.stderr)

    if HAS_TRIMESH:
        print(f"Opening with Trimesh fallback viewer: {path.name}")
        tm = trimesh.load(str(path))
        tm.show()
        return 0

    print("Error: Neither Open3D nor Trimesh visualizer could be launched.", file=sys.stderr)
    return 1


def _launch_open3d_viewer(
    filepath: Path,
    window_name: str,
    width: int,
    height: int,
) -> int:
    """Internal Open3D visualization implementation with custom key controls."""
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh '{filepath.name}' contains 0 vertices.")

    # Ensure smooth vertex normals exist
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    num_v = len(verts)
    num_f = len(mesh.triangles)

    min_b = verts.min(axis=0)
    max_b = verts.max(axis=0)

    # Print model diagnostics & controls
    print("==================================================")
    print("STEREO3D: INTERACTIVE 3D OBJECT VIEWER")
    print("==================================================")
    print(f"Model File      : {filepath}")
    print(f"Vertices        : {num_v:,}")
    print(f"Triangles       : {num_f:,}")
    print(f"Extent (X)      : [{min_b[0]:.2f}, {max_b[0]:.2f}] px")
    print(f"Extent (Y)      : [{min_b[1]:.2f}, {max_b[1]:.2f}] px")
    print(f"Displacement (Z): [{min_b[2]:.2f}, {max_b[2]:.2f}] px")
    print("--------------------------------------------------")
    print("INTERACTIVE CONTROLS:")
    print("  Left Click + Drag   : Rotate / Orbit mesh")
    print("  Right Click + Drag  : Pan / Translate (or Shift + Left Drag)")
    print("  Mouse Wheel         : Zoom in / out (or Ctrl + Left Drag)")
    print("  [W] Key             : Toggle Wireframe overlay")
    print("  [C] Key             : Cycle Color mode (Texture -> Elevation -> Normals -> Clay)")
    print("  [B] Key             : Cycle Background (Dark Slate -> Pitch Black -> Studio Grey)")
    print("  [R] Key             : Reset Camera View")
    print("  [Q] or [ESC]        : Close Viewer")
    print("==================================================")

    # Prepare color modes
    has_orig_colors = mesh.has_vertex_colors()
    if has_orig_colors:
        orig_colors = np.asarray(mesh.vertex_colors).copy()
    else:
        orig_colors = np.full((num_v, 3), 0.85, dtype=np.float64)

    # Depth elevation colormap
    z_vals = verts[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = max(1e-5, z_max - z_min)
    z_norm = (z_vals - z_min) / z_range
    cmap_turbo = plt.get_cmap("turbo")
    elevation_colors = cmap_turbo(z_norm)[:, :3].astype(np.float64)

    # Surface normals as colors
    normals = np.asarray(mesh.vertex_normals)
    normal_colors = ((normals + 1.0) * 0.5).clip(0.0, 1.0)

    # Studio Clay
    clay_colors = np.full((num_v, 3), 0.85, dtype=np.float64)

    color_modes = [
        ("RGB Texture / Vertex Colors" if has_orig_colors else "Neutral Surface", orig_colors),
        ("Depth Elevation (Turbo Heatmap)", elevation_colors),
        ("Surface Normal Orientations", normal_colors),
        ("Studio Neutral Clay", clay_colors),
    ]
    current_color_idx = 0

    # Background themes
    bg_modes = [
        ("Dark Slate", np.array([0.08, 0.08, 0.10])),
        ("Pitch Black", np.array([0.00, 0.00, 0.00])),
        ("Studio Grey", np.array([0.80, 0.80, 0.82])),
    ]
    current_bg_idx = 0

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=width, height=height)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = bg_modes[current_bg_idx][1]
    opt.light_on = True
    opt.mesh_show_back_face = True

    # Callbacks
    def toggle_wireframe(v):
        ro = v.get_render_option()
        ro.mesh_show_wireframe = not ro.mesh_show_wireframe
        state = "ON" if ro.mesh_show_wireframe else "OFF"
        print(f"Wireframe: {state}")
        return False

    def cycle_color(v):
        nonlocal current_color_idx
        current_color_idx = (current_color_idx + 1) % len(color_modes)
        mode_name, c_arr = color_modes[current_color_idx]
        mesh.vertex_colors = o3d.utility.Vector3dVector(c_arr)
        v.update_geometry(mesh)
        print(f"Color Mode: {mode_name}")
        return False

    def cycle_background(v):
        nonlocal current_bg_idx
        current_bg_idx = (current_bg_idx + 1) % len(bg_modes)
        bg_name, bg_val = bg_modes[current_bg_idx]
        ro = v.get_render_option()
        ro.background_color = bg_val
        print(f"Background: {bg_name}")
        return False

    def reset_view(v):
        v.reset_view_point(True)
        print("Camera view reset.")
        return False

    vis.register_key_callback(ord("W"), toggle_wireframe)
    vis.register_key_callback(ord("w"), toggle_wireframe)
    vis.register_key_callback(ord("C"), cycle_color)
    vis.register_key_callback(ord("c"), cycle_color)
    vis.register_key_callback(ord("B"), cycle_background)
    vis.register_key_callback(ord("b"), cycle_background)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("r"), reset_view)

    vis.run()
    vis.destroy_window()
    print("Viewer closed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer for reconstructed autostereogram meshes."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        type=str,
        help="Path or name of 3D model file (.obj, .ply). If omitted, opens the newest model in output/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to search for models if a relative name is given (default: output).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Initial window width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Initial window height in pixels (default: 800).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        model_path = find_3d_model(args.model, output_dir=args.output_dir)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    return launch_viewer(model_path, width=args.width, height=args.height)


if __name__ == "__main__":
    sys.exit(main())
