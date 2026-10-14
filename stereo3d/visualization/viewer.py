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


def _get_ibl_path() -> Optional[str]:
    """Retrieve bundled Open3D HDR environment map for global illumination."""
    if not HAS_OPEN3D:
        return None
    o3d_dir = Path(o3d.__file__).parent
    for name in ["default_ibl.ktx", "brightday_ibl.ktx", "crossroads_ibl.ktx"]:
        cand = o3d_dir / "resources" / name
        if cand.is_file():
            return str(cand)
    return None


def launch_viewer(
    filepath: Union[str, Path],
    window_name: Optional[str] = None,
    width: int = 1280,
    height: int = 800,
    engine: str = "pbr",
    default_mode: str = "untextured",
) -> int:
    """
    Launch interactive 3D visualizer for the specified mesh file.

    Args:
        filepath: Path to the 3D mesh (.obj, .ply).
        window_name: Window title string.
        width: Initial window width in pixels.
        height: Initial window height in pixels.
        engine: Rendering engine ('pbr' for Filament Global Illumination or 'classic' for OpenGL).
        default_mode: Starting display mode ('untextured' for shaded clay or 'material' for textured).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    title = window_name or f"Stereo3D Mesh Viewer - {path.name}"

    if HAS_OPEN3D:
        if engine.lower() == "pbr":
            try:
                return _launch_pbr_viewer(
                    path,
                    window_name=title,
                    width=width,
                    height=height,
                    default_mode=default_mode,
                )
            except Exception as e:
                print(f"PBR Visualizer initialization failed ({e}). Falling back to classic OpenGL viewer...", file=sys.stderr)

        # Fallback or explicit classic visualizer
        try:
            return _launch_classic_viewer(
                path,
                window_name=title,
                width=width,
                height=height,
                default_mode=default_mode,
            )
        except Exception as e:
            print(f"Classic visualizer failed ({e}). Attempting Trimesh fallback...", file=sys.stderr)

    if HAS_TRIMESH:
        print(f"Opening with Trimesh fallback viewer: {path.name}")
        tm = trimesh.load(str(path))
        tm.show()
        return 0

    print("Error: No compatible 3D visualizer could be launched.", file=sys.stderr)
    return 1


def _launch_pbr_viewer(
    filepath: Path,
    window_name: str,
    width: int,
    height: int,
    default_mode: str = "untextured",
) -> int:
    """
    Filament PBR rendering with Image-Based Lighting (Global Illumination)
    and interactive UI material / untextured swapping.
    """
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh '{filepath.name}' contains 0 vertices.")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    num_v = len(verts)
    num_f = len(mesh.triangles)

    min_b = verts.min(axis=0)
    max_b = verts.max(axis=0)

    # 1. Untextured Shaded Clay Mesh (no vertex colors, pure PBR surface)
    mesh_clay = o3d.geometry.TriangleMesh(mesh)
    mesh_clay.vertex_colors = o3d.utility.Vector3dVector([])

    mat_clay = o3d.visualization.rendering.MaterialRecord()
    mat_clay.shader = "defaultLit"
    mat_clay.base_color = [0.82, 0.82, 0.84, 1.0]  # Smooth studio clay
    mat_clay.base_roughness = 0.40
    mat_clay.base_reflectance = 0.35

    # 2. Material (Textured / RGB) Mesh
    mesh_material = o3d.geometry.TriangleMesh(mesh)
    mat_material = o3d.visualization.rendering.MaterialRecord()
    mat_material.shader = "defaultLit"
    mat_material.base_roughness = 0.45
    mat_material.base_reflectance = 0.30

    # 3. Depth Elevation (Turbo Heatmap) Mesh
    mesh_elevation = o3d.geometry.TriangleMesh(mesh)
    z_vals = verts[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = max(1e-5, z_max - z_min)
    z_norm = (z_vals - z_min) / z_range
    cmap_turbo = plt.get_cmap("turbo")
    elevation_colors = cmap_turbo(z_norm)[:, :3].astype(np.float64)
    mesh_elevation.vertex_colors = o3d.utility.Vector3dVector(elevation_colors)

    mat_elevation = o3d.visualization.rendering.MaterialRecord()
    mat_elevation.shader = "defaultLit"
    mat_elevation.base_roughness = 0.35
    mat_elevation.base_reflectance = 0.35

    CLAY_NAME = "Untextured (Shaded Clay)"
    MAT_NAME = "Material (Textured / RGB)"
    ELEV_NAME = "Depth Elevation (Turbo)"

    is_untextured = (default_mode.lower() != "material")

    geometries = [
        {
            "name": CLAY_NAME,
            "geometry": mesh_clay,
            "material": mat_clay,
            "is_visible": is_untextured,
        },
        {
            "name": MAT_NAME,
            "geometry": mesh_material,
            "material": mat_material,
            "is_visible": not is_untextured,
        },
        {
            "name": ELEV_NAME,
            "geometry": mesh_elevation,
            "material": mat_elevation,
            "is_visible": False,
        },
    ]

    ibl_path = _get_ibl_path()

    print("==================================================")
    print("STEREO3D: PBR 3D OBJECT VIEWER (GLOBAL ILLUMINATION)")
    print("==================================================")
    print(f"Model File      : {filepath}")
    print(f"Vertices        : {num_v:,}")
    print(f"Triangles       : {num_f:,}")
    print(f"Extent (X)      : [{min_b[0]:.2f}, {max_b[0]:.2f}] px")
    print(f"Extent (Y)      : [{min_b[1]:.2f}, {max_b[1]:.2f}] px")
    print(f"Displacement (Z): [{min_b[2]:.2f}, {max_b[2]:.2f}] px")
    print(f"Initial Mode    : {'Untextured (Shaded Clay)' if is_untextured else 'Material (Textured)'}")
    print(f"Global Lighting : Image-Based Lighting (IBL) + Sun Directional Shadows")
    print("--------------------------------------------------")
    print("INTERACTIVE CONTROLS & UI ACTIONS:")
    print("  Mouse Left Drag     : Rotate / Orbit mesh")
    print("  Mouse Right Drag    : Pan / Translate (or Shift + Left Drag)")
    print("  Mouse Wheel         : Zoom in / out")
    print("  [Swap: Material / Untextured] : Toggle between textured image & untextured clay")
    print("  [Toggle Depth Heatmap]       : View Turbo height/depth gradient")
    print("  Settings Sidebar             : Customize sunlight, IBL environment, and roughness")
    print("==================================================")

    def swap_material_action(vis):
        is_mat_visible = vis.get_geometry(MAT_NAME).is_visible
        if is_mat_visible:
            vis.show_geometry(MAT_NAME, False)
            vis.show_geometry(CLAY_NAME, True)
            print("[Viewer] Active Mode: Untextured (Shaded Clay Surface)")
        else:
            vis.show_geometry(CLAY_NAME, False)
            vis.show_geometry(ELEV_NAME, False)
            vis.show_geometry(MAT_NAME, True)
            print("[Viewer] Active Mode: Material (Textured / RGB Pattern)")

    def toggle_elevation_action(vis):
        is_elev_visible = vis.get_geometry(ELEV_NAME).is_visible
        if is_elev_visible:
            vis.show_geometry(ELEV_NAME, False)
            vis.show_geometry(CLAY_NAME, True)
            print("[Viewer] Active Mode: Untextured (Shaded Clay Surface)")
        else:
            vis.show_geometry(CLAY_NAME, False)
            vis.show_geometry(MAT_NAME, False)
            vis.show_geometry(ELEV_NAME, True)
            print("[Viewer] Active Mode: Depth Elevation (Turbo Heatmap)")

    actions = [
        ("Swap: Material / Untextured", swap_material_action),
        ("Toggle Depth Heatmap", toggle_elevation_action),
    ]

    o3d.visualization.draw(
        geometry=geometries,
        title=window_name,
        width=width,
        height=height,
        actions=actions,
        bg_color=(0.10, 0.10, 0.12, 1.0),
        ibl=ibl_path,
        ibl_intensity=45000.0,
        show_ui=True,
    )
    print("Viewer closed.")
    return 0


def _launch_classic_viewer(
    filepath: Path,
    window_name: str,
    width: int,
    height: int,
    default_mode: str = "untextured",
) -> int:
    """Classic OpenGL visualizer with direct keyboard hotkey callbacks."""
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh '{filepath.name}' contains 0 vertices.")

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
    print(f"Initial Mode    : {'Untextured (Shaded Clay)' if default_mode != 'material' else 'Material (Textured)'}")
    print("--------------------------------------------------")
    print("INTERACTIVE HOTKEYS:")
    print("  Left Click + Drag   : Rotate / Orbit mesh")
    print("  Right Click + Drag  : Pan / Translate (or Shift + Left Drag)")
    print("  Mouse Wheel         : Zoom in / out (or Ctrl + Left Drag)")
    print("  [M] or [T] Key      : SWAP between Material (Textured) and Untextured (Shaded Clay)")
    print("  [C] Key             : Cycle Color (Untextured -> Material -> Elevation -> Normals)")
    print("  [W] Key             : Toggle Wireframe overlay")
    print("  [B] Key             : Cycle Background (Dark Slate -> Pitch Black -> Studio Grey)")
    print("  [R] Key             : Reset Camera View")
    print("  [Q] or [ESC]        : Close Viewer")
    print("==================================================")

    # Cache color options
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

    # Surface normals
    normals = np.asarray(mesh.vertex_normals)
    normal_colors = ((normals + 1.0) * 0.5).clip(0.0, 1.0)

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
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Default

    # Set initial color option
    is_textured = (default_mode.lower() == "material" and has_orig_colors)
    if is_textured:
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    else:
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Default

    # Color cycling list
    color_modes = [
        ("Untextured (Shaded Clay Surface)", o3d.visualization.MeshColorOption.Default, None),
        ("Material (Textured / RGB Pattern)", o3d.visualization.MeshColorOption.Color, orig_colors),
        ("Depth Elevation (Turbo Heatmap)", o3d.visualization.MeshColorOption.Color, elevation_colors),
        ("Surface Normal Orientations", o3d.visualization.MeshColorOption.Color, normal_colors),
    ]
    current_color_idx = 1 if is_textured else 0

    def toggle_material_untextured(v):
        nonlocal is_textured
        is_textured = not is_textured
        ro = v.get_render_option()
        if is_textured:
            mesh.vertex_colors = o3d.utility.Vector3dVector(orig_colors)
            ro.mesh_color_option = o3d.visualization.MeshColorOption.Color
            print("[Viewer] Mode: Material (Textured / RGB Pattern)")
        else:
            ro.mesh_color_option = o3d.visualization.MeshColorOption.Default
            print("[Viewer] Mode: Untextured (Shaded Clay Surface)")
        v.update_geometry(mesh)
        return False

    def cycle_color_modes(v):
        nonlocal current_color_idx
        current_color_idx = (current_color_idx + 1) % len(color_modes)
        mode_name, color_opt, c_arr = color_modes[current_color_idx]
        ro = v.get_render_option()
        ro.mesh_color_option = color_opt
        if c_arr is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(c_arr)
            v.update_geometry(mesh)
        print(f"[Viewer] Color Mode: {mode_name}")
        return False

    def toggle_wireframe(v):
        ro = v.get_render_option()
        ro.mesh_show_wireframe = not ro.mesh_show_wireframe
        state = "ON" if ro.mesh_show_wireframe else "OFF"
        print(f"[Viewer] Wireframe: {state}")
        return False

    def cycle_background(v):
        nonlocal current_bg_idx
        current_bg_idx = (current_bg_idx + 1) % len(bg_modes)
        bg_name, bg_val = bg_modes[current_bg_idx]
        ro = v.get_render_option()
        ro.background_color = bg_val
        print(f"[Viewer] Background: {bg_name}")
        return False

    def reset_view(v):
        v.reset_view_point(True)
        print("[Viewer] Camera view reset.")
        return False

    # Register hotkeys
    vis.register_key_callback(ord("M"), toggle_material_untextured)
    vis.register_key_callback(ord("m"), toggle_material_untextured)
    vis.register_key_callback(ord("T"), toggle_material_untextured)
    vis.register_key_callback(ord("t"), toggle_material_untextured)
    vis.register_key_callback(ord("C"), cycle_color_modes)
    vis.register_key_callback(ord("c"), cycle_color_modes)
    vis.register_key_callback(ord("W"), toggle_wireframe)
    vis.register_key_callback(ord("w"), toggle_wireframe)
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
        "--mode",
        choices=["untextured", "material", "depth"],
        default="untextured",
        help="Initial display mode: 'untextured' (shaded clay with global illumination) or 'material' (textured).",
    )
    parser.add_argument(
        "--engine",
        choices=["pbr", "classic"],
        default="pbr",
        help="Rendering engine: 'pbr' (Filament Global Illumination) or 'classic' (OpenGL hotkeys).",
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

    return launch_viewer(
        model_path,
        width=args.width,
        height=args.height,
        engine=args.engine,
        default_mode=args.mode,
    )


if __name__ == "__main__":
    sys.exit(main())
