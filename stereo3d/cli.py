import argparse
from pathlib import Path
import sys

from .core.datastructures import StereogramImage
from .core.depth import DepthExtractor
from .core.mesh import MeshGenerator
from .core.period import PeriodEstimator
from .visualization.plotting import plot_depth_analysis
from .visualization.viewer import launch_viewer


def resolve_image_path(image_arg: str, input_dir: Path = Path("input")) -> Path:
    """
    Resolve image path checking both direct path and the input/ directory.
    """
    direct_path = Path(image_arg)
    if direct_path.is_file():
        return direct_path

    # Check inside input_dir
    input_candidate = input_dir / image_arg
    if input_candidate.is_file():
        return input_candidate

    # Try matching with common image extensions if suffix was omitted
    if not direct_path.suffix:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            cand = input_dir / f"{image_arg}{ext}"
            if cand.is_file():
                return cand

    raise FileNotFoundError(
        f"Image '{image_arg}' not found. Searched at '{direct_path.resolve()}' "
        f"and inside '{input_dir.resolve()}'."
    )


def process_image(args: argparse.Namespace) -> int:
    try:
        image_path = resolve_image_path(args.image, input_dir=Path(args.input_dir))
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STEREO3D: RECONSTRUCTION & 3D MESH GENERATION")
    print("==================================================")
    print(f"Input Image     : {image_path}")

    stereo_img = StereogramImage.from_file(image_path)
    print(f"Dimensions      : {stereo_img.width}x{stereo_img.height} (WxH)")

    # 1. Base period determination
    period_result = None
    if args.period is not None:
        period = args.period
        print(f"Base Period     : {period} px (manual override)")
    else:
        print(f"Estimating base period via '{args.method}' method...")
        estimator = PeriodEstimator(
            method=args.method,
            row_sampling=args.sampling,
            min_period=args.min_period,
            max_period=args.max_period,
        )
        period_result = estimator.estimate(stereo_img)
        period = period_result.period
        print(f"Detected Period : {period} px (confidence: {period_result.confidence:.3f})")
        if period_result.harmonics:
            print(f"Harmonics       : {period_result.harmonics} px")

    # 2. Dense Depth Extraction
    max_d = args.max_disparity
    print(f"Extracting dense depth map (view mode: '{args.view_mode}')...")
    extractor = DepthExtractor(
        base_period=period,
        max_disparity=max_d,
        view_mode=args.view_mode,
        window_size=args.window_size,
        subpixel=not args.no_subpixel,
        robust_norm=not args.no_robust_norm,
        bg_prior=not args.no_bg_prior,
        bg_threshold=args.bg_threshold,
    )
    depth_result = extractor.extract(stereo_img)
    print(f"Disparity Range : [0, {depth_result.max_disparity}] px")

    # 3. Save Depth Outputs
    stem = image_path.stem
    depth_png = out_dir / f"{stem}_depth.png"
    depth_16bit = out_dir / f"{stem}_depth_16bit.png"
    depth_color = out_dir / f"{stem}_depth_color.png"
    depth_npy = out_dir / f"{stem}_depth.npy"
    analysis_plot = out_dir / f"{stem}_analysis.png"

    depth_result.save_png(depth_png)
    depth_result.save_16bit(depth_16bit)
    depth_result.save_color(depth_color, cmap=args.colormap)
    depth_result.save_numpy(depth_npy)

    plot_depth_analysis(
        image=stereo_img,
        depth_map=depth_result.depth,
        period_result=period_result,
        save_path=analysis_plot,
        show=args.show,
    )

    # 4. Generate 3D Surface Mesh
    mesh_obj = out_dir / f"{stem}_mesh.obj"
    mesh_ply = out_dir / f"{stem}_mesh.ply"
    mesh_result = None

    if not args.no_mesh:
        print(f"Generating 3D mesh (step: {args.mesh_step}, z_scale: {args.z_scale or 'auto'})...")
        mesh_gen = MeshGenerator(
            z_scale=args.z_scale,
            step=args.mesh_step,
            color_mode=args.mesh_color,
            colormap=args.colormap,
            smooth_normals=True,
            invert_z=args.invert_z,
        )
        mesh_result = mesh_gen.generate(depth_result.depth, stereogram=stereo_img)
        mesh_result.save_obj(mesh_obj)
        mesh_result.save_ply(mesh_ply)

    print("--------------------------------------------------")
    print("OUTPUT FILES SAVED:")
    print(f"  Standard Depth (8-bit PNG)  : {depth_png}")
    print(f"  3D Displacement (16-bit PNG): {depth_16bit}")
    print(f"  Colorized Depth (RGB PNG)   : {depth_color}")
    print(f"  Raw Array (NumPy .npy)      : {depth_npy}")
    print(f"  Diagnostic Report (Plot PNG): {analysis_plot}")
    if mesh_result is not None:
        print(f"  3D Model (Wavefront OBJ)    : {mesh_obj}")
        print(f"  3D Model (Stanford PLY)     : {mesh_ply}")
        print(f"  Mesh Statistics             : {mesh_result.num_vertices:,} vertices, {mesh_result.num_triangles:,} triangles")
    print("==================================================")
    print("Reconstruction completed successfully.")

    if args.view and mesh_result is not None:
        print("\nLaunching 3D object viewer...")
        launch_viewer(
            mesh_obj,
            engine=args.viewer_engine,
            default_mode=args.view_mode_3d,
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct dense 3D depth maps from autostereograms."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path or filename of the stereogram image (searches current directory and input/ folder)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="Manual base period S_0 in pixels. If omitted, automatically estimated.",
    )
    parser.add_argument(
        "--max-disparity",
        type=int,
        default=None,
        help="Maximum disparity shift in pixels. Defaults to 35%% of base period.",
    )
    parser.add_argument(
        "--method",
        choices=["autocorrelation", "difference"],
        default="autocorrelation",
        help="Period estimation method ('autocorrelation' or 'difference').",
    )
    parser.add_argument(
        "--sampling",
        choices=["distributed", "middle", "all"],
        default="distributed",
        help="Row sampling strategy for period estimation.",
    )
    parser.add_argument(
        "--view-mode",
        choices=["parallel", "cross"],
        default="parallel",
        help="Stereogram viewing mode ('parallel' for standard Magic Eye, 'cross' for cross-eyed).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Spatial aggregation window size (odd integer).",
    )
    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="Disable subpixel quadratic disparity refinement.",
    )
    parser.add_argument(
        "--no-robust-norm",
        action="store_true",
        help="Disable robust active depth percentile scaling.",
    )
    parser.add_argument(
        "--no-bg-prior",
        action="store_true",
        help="Disable background ambiguity prior.",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.88,
        help="Ratio threshold for background prior ambiguity suppression (default: 0.88).",
    )
    parser.add_argument(
        "--min-period",
        type=int,
        default=25,
        help="Minimum period in pixels for search range.",
    )
    parser.add_argument(
        "--max-period",
        type=int,
        default=None,
        help="Maximum period in pixels for search range.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        help="Matplotlib colormap for colorized depth output (default: 'turbo').",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Default folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Folder to store reconstructed depth maps.",
    )
    parser.add_argument(
        "--mesh-step",
        type=int,
        default=2,
        help="Subsampling step for 3D mesh grid (default: 2; 1 = full 1:1 pixel resolution).",
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=None,
        help="3D extrusion height scale factor (default: 20%% of maximum image dimension).",
    )
    parser.add_argument(
        "--mesh-color",
        choices=["rgb", "depth", "uniform"],
        default="rgb",
        help="Coloring strategy for 3D mesh ('rgb', 'depth', 'uniform').",
    )
    parser.add_argument(
        "--invert-z",
        action="store_true",
        help="Invert depth displacement direction (concave vs convex).",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip 3D mesh generation.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open interactive 3D viewer immediately after reconstruction.",
    )
    parser.add_argument(
        "--view-mode-3d",
        choices=["untextured", "material", "depth"],
        default="untextured",
        help="Initial 3D viewer shading mode (default: 'untextured' shaded clay).",
    )
    parser.add_argument(
        "--viewer-engine",
        choices=["pbr", "classic"],
        default="pbr",
        help="3D rendering engine ('pbr' for Filament Global Illumination or 'classic' for OpenGL).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display diagnostic figure interactively.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return process_image(args)


if __name__ == "__main__":
    sys.exit(main())
