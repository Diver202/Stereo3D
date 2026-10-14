"""
Standalone script to generate a synthetic autostereogram with known ground-truth geometry
and test period estimation and depth map reconstruction algorithms.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from stereo3d.core.datastructures import StereogramImage
from stereo3d.core.depth import DepthExtractor
from stereo3d.core.mesh import MeshGenerator
from stereo3d.core.period import PeriodEstimator
from stereo3d.utils.synthetic import generate_synthetic_stereogram


def run_generate_and_test(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STEREO3D: SYNTHETIC GENERATION & RECONSTRUCTION TEST")
    print("==================================================")
    print(f"Dimensions      : {args.width}x{args.height}")
    print(f"Ground Truth S_0: {args.period} px")
    print(f"Shape Type      : {args.shape}")
    print(f"Max Disparity   : {args.max_disparity} px")
    print("--------------------------------------------------")

    # 1. Generate synthetic stereogram
    print("Generating synthetic autostereogram...")
    stereo_arr, gt_depth = generate_synthetic_stereogram(
        width=args.width,
        height=args.height,
        base_period=args.period,
        shape_type=args.shape,
        max_disparity=args.max_disparity,
    )

    stereo_path = out_dir / f"test_{args.shape}_stereogram.png"
    Image.fromarray(stereo_arr).save(stereo_path)
    print(f"Saved synthetic stereogram -> {stereo_path}")

    gt_depth_path = out_dir / f"test_{args.shape}_gt_depth.png"
    Image.fromarray((gt_depth * 255.0).astype(np.uint8), mode="L").save(gt_depth_path)
    print(f"Saved ground truth depth  -> {gt_depth_path}")

    # 2. Test base period estimation
    stereo_img = StereogramImage.from_numpy(stereo_arr, source_path=stereo_path)

    print("\nEstimating base period via autocorrelation...")
    period_est_corr = PeriodEstimator(method="autocorrelation")
    res_corr = period_est_corr.estimate(stereo_img)

    print("Estimating base period via difference metric...")
    period_est_diff = PeriodEstimator(method="difference")
    res_diff = period_est_diff.estimate(stereo_img)

    print(f"  Autocorrelation Result : {res_corr.period} px (confidence: {res_corr.confidence:.3f})")
    print(f"  Difference Result      : {res_diff.period} px (confidence: {res_diff.confidence:.3f})")

    # 3. Test depth map extraction
    print("\nExtracting dense depth map...")
    extractor = DepthExtractor(
        base_period=res_corr.period,
        max_disparity=args.max_disparity,
    )
    depth_res = extractor.extract(stereo_img)

    rec_depth = depth_res.depth

    # 4. Compute metrics against ground truth
    corr = float(np.corrcoef(rec_depth.ravel(), gt_depth.ravel())[0, 1])
    mae = float(np.mean(np.abs(rec_depth - gt_depth)))

    print("\n--------------------------------------------------")
    print("EVALUATION METRICS:")
    print(f"  Base Period Error : {abs(res_corr.period - args.period)} px")
    print(f"  Depth Correlation : {corr:.4f} (target: > 0.95)")
    print(f"  Depth MAE         : {mae:.4f}")
    print("--------------------------------------------------")

    # 5. Save reconstruction outputs
    depth_png = out_dir / f"test_{args.shape}_depth.png"
    depth_res.save_png(depth_png)
    print(f"Saved reconstructed depth (8-bit)  -> {depth_png}")

    depth_16bit = out_dir / f"test_{args.shape}_depth_16bit.png"
    depth_res.save_16bit(depth_16bit)
    print(f"Saved reconstructed depth (16-bit) -> {depth_16bit}")

    depth_color = out_dir / f"test_{args.shape}_depth_color.png"
    depth_res.save_color(depth_color, cmap="turbo")
    print(f"Saved colorized depth map          -> {depth_color}")

    # 6. Generate 3D surface mesh
    mesh_gen = MeshGenerator(step=2, color_mode="rgb")
    mesh_res = mesh_gen.generate(rec_depth, stereogram=stereo_img)
    mesh_obj = out_dir / f"test_{args.shape}_mesh.obj"
    mesh_ply = out_dir / f"test_{args.shape}_mesh.ply"
    mesh_res.save_obj(mesh_obj)
    mesh_res.save_ply(mesh_ply)
    print(f"Saved 3D mesh (Wavefront OBJ)      -> {mesh_obj}")
    print(f"Saved 3D mesh (Stanford PLY)       -> {mesh_ply}")
    print(f"  Mesh Geometry                    : {mesh_res.num_vertices:,} vertices, {mesh_res.num_triangles:,} triangles")

    # 7. Save visual comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(stereo_arr)
    axes[0].set_title(f"Synthetic Stereogram ({args.width}x{args.height})")
    axes[0].set_axis_off()

    axes[1].imshow(gt_depth, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[1].set_title("Ground Truth Depth")
    axes[1].set_axis_off()

    im = axes[2].imshow(rec_depth, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Reconstructed Depth (Corr: {corr:.3f})")
    axes[2].set_axis_off()
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    comparison_path = out_dir / f"test_{args.shape}_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visual evaluation plot       -> {comparison_path}")

    print("\nAll tests completed successfully.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic autostereograms and test 3D depth recovery."
    )
    parser.add_argument("--shape", choices=["circle", "box"], default="circle", help="Shape of hidden 3D object")
    parser.add_argument("--period", type=int, default=120, help="Base repetition period in pixels")
    parser.add_argument("--max-disparity", type=int, default=24, help="Maximum disparity in pixels")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save test outputs")

    args = parser.parse_args()
    sys.exit(run_generate_and_test(args))


if __name__ == "__main__":
    main()
