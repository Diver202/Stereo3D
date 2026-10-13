import argparse
from pathlib import Path
import sys

from PIL import Image

from .core.datastructures import StereogramImage
from .core.period import PeriodEstimator
from .visualization.plotting import plot_period_analysis
from .utils.synthetic import generate_synthetic_stereogram


def run_analysis(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return 1

    print(f"Loading stereogram from: {image_path}")
    stereo_img = StereogramImage.from_file(image_path)
    print(f"Image dimensions: {stereo_img.width}x{stereo_img.height} (WxH)")

    estimator = PeriodEstimator(
        min_period=args.min_period,
        max_period=args.max_period,
        method=args.method,
        row_sampling=args.sampling,
        row_count=args.rows,
        aggregation=args.aggregation,
    )

    print(f"Estimating base period using '{args.method}' method (sampling: '{args.sampling}')...")
    result = estimator.estimate(stereo_img)

    print("\n" + "=" * 45)
    print(f"ESTIMATION RESULTS:")
    print(f"  Fundamental Period (S_0) : {result.period} pixels")
    print(f"  Confidence Score         : {result.confidence:.3f}")
    print(f"  Method                   : {result.method}")
    if result.harmonics:
        print(f"  Harmonics Detected       : {result.harmonics} pixels")
    print("=" * 45 + "\n")

    if args.plot or args.show:
        plot_path = args.plot or f"output/period_analysis_{image_path.stem}.png"
        print(f"Generating diagnostic plot -> {plot_path}")
        plot_period_analysis(result, stereo_img, save_path=plot_path, show=args.show)

    return 0


def run_demo(args: argparse.Namespace) -> int:
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_period = args.ground_truth_period
    print(f"Generating synthetic autostereogram (ground-truth base period: {gt_period}px)...")
    stereo_arr, depth_map = generate_synthetic_stereogram(
        width=args.width,
        height=args.height,
        base_period=gt_period,
        shape_type=args.shape,
    )

    demo_img_path = out_dir / "synthetic_stereogram.png"
    Image.fromarray(stereo_arr).save(demo_img_path)
    print(f"Saved synthetic stereogram to: {demo_img_path}")

    stereo_img = StereogramImage.from_numpy(stereo_arr, source_path=demo_img_path)

    # 1. Test Autocorrelation
    est_corr = PeriodEstimator(
        method="autocorrelation",
        row_sampling=args.sampling,
        min_period=25,
    )
    res_corr = est_corr.estimate(stereo_img)

    # 2. Test Photoshop Difference method
    est_diff = PeriodEstimator(
        method="difference",
        row_sampling=args.sampling,
        min_period=25,
    )
    res_diff = est_diff.estimate(stereo_img)

    print("\n" + "=" * 50)
    print("DEMO VERIFICATION RESULTS:")
    print(f"  Ground Truth Period : {gt_period} px")
    print(f"  Autocorrelation Est : {res_corr.period} px (Confidence: {res_corr.confidence:.3f})")
    print(f"  Difference (PS) Est : {res_diff.period} px (Confidence: {res_diff.confidence:.3f})")
    print("=" * 50 + "\n")

    plot_path = out_dir / "demo_period_analysis.png"
    plot_period_analysis(res_diff, stereo_img, save_path=plot_path, show=args.show)
    print(f"Saved period analysis visualization to: {plot_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stereo3D: Autostereogram depth extraction and period analyzer."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze an autostereogram image file")
    analyze_parser.add_argument("image", type=str, help="Path to stereogram image file")
    analyze_parser.add_argument(
        "--method",
        choices=["autocorrelation", "difference"],
        default="autocorrelation",
        help="Method to use (autocorrelation or difference/Photoshop)",
    )
    analyze_parser.add_argument(
        "--sampling",
        choices=["distributed", "middle", "all"],
        default="distributed",
        help="Row sampling strategy",
    )
    analyze_parser.add_argument(
        "--rows",
        type=int,
        default=64,
        help="Number of rows to sample",
    )
    analyze_parser.add_argument(
        "--aggregation",
        choices=["median", "mean"],
        default="median",
        help="Row aggregation method",
    )
    analyze_parser.add_argument(
        "--min-period",
        type=int,
        default=25,
        help="Minimum period in pixels to search",
    )
    analyze_parser.add_argument(
        "--max-period",
        type=int,
        default=None,
        help="Maximum period in pixels to search",
    )
    analyze_parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output path for diagnostic plot",
    )
    analyze_parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window",
    )

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run self-test demo on synthetic stereogram")
    demo_parser.add_argument(
        "--ground-truth-period",
        type=int,
        default=120,
        help="Base period for synthetic stereogram",
    )
    demo_parser.add_argument(
        "--shape",
        choices=["circle", "box"],
        default="circle",
        help="Hidden 3D shape type",
    )
    demo_parser.add_argument("--width", type=int, default=640, help="Image width")
    demo_parser.add_argument("--height", type=int, default=480, help="Image height")
    demo_parser.add_argument(
        "--sampling",
        choices=["distributed", "middle", "all"],
        default="distributed",
        help="Row sampling strategy",
    )
    demo_parser.add_argument("--show", action="store_true", help="Display plot interactively")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        return run_analysis(args)
    elif args.command == "demo":
        return run_demo(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
