"""
Standalone pipeline for generating high-entropy autostereograms from depth maps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Union

from stereo3d.generators.stereogram_generator import (
    COLOR_PALETTES,
    AutostereogramGenerator,
    generate_autostereogram,
)


def find_depth_map(
    depth_arg: Optional[str] = None,
    depthmaps_dir: Union[str, Path] = "depthmaps",
) -> Path:
    """
    Locate depth map file from explicit path, stem name, or newest file in depthmaps/.
    """
    dm_dir = Path(depthmaps_dir)

    if depth_arg:
        p = Path(depth_arg)
        # 1. Direct path
        if p.is_file():
            return p.resolve()

        # 2. Inside depthmaps directory
        cand = dm_dir / depth_arg
        if cand.is_file():
            return cand.resolve()

        # 3. Inside input directory
        cand_in = Path("input") / depth_arg
        if cand_in.is_file():
            return cand_in.resolve()

        # 4. Try with common suffixes
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
            for folder in [dm_dir, Path("input"), Path(".")]:
                cand_ext = folder / f"{depth_arg}{ext}"
                if cand_ext.is_file():
                    return cand_ext.resolve()

        raise FileNotFoundError(
            f"Depth map '{depth_arg}' not found. Searched at '{p.resolve()}' "
            f"and inside '{dm_dir.resolve()}'."
        )

    # If no argument: search depthmaps/ for newest image
    if not dm_dir.is_dir():
        dm_dir.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"Depthmaps directory '{dm_dir.resolve()}' was empty. "
            "Please place your depth maps (e.g. MonkeyDepthMap.png) inside depthmaps/."
        )

    candidates: List[Path] = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff"]:
        candidates.extend(dm_dir.glob(ext))

    if not candidates:
        raise FileNotFoundError(
            f"No depth map images found in '{dm_dir.resolve()}'. "
            "Place your depth maps inside depthmaps/ and run again."
        )

    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate high-entropy, undecipherable autostereograms from 2D/3D depth maps."
    )
    parser.add_argument(
        "depthmap",
        nargs="?",
        default=None,
        type=str,
        help="Path or name of input depth map file (defaults to newest image in depthmaps/).",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="blue",
        help="Predominant color theme (presets: blue, emerald, crimson, amber, purple, cyan, cyberpunk, sunset, matrix, monochrome, rainbow) or custom hex code #RRGGBB (default: 'blue').",
    )
    parser.add_argument(
        "--pattern",
        choices=["organic", "speckle", "cellular", "plasma"],
        default="organic",
        help="Noise pattern style designed to conceal monocular contours (default: 'organic').",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="Base repetition period S_0 in pixels (default: auto, approx width / 14).",
    )
    parser.add_argument(
        "--max-disparity",
        type=int,
        default=None,
        help="Maximum disparity shift in pixels (default: auto, approx 25%% of base period).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation. If omitted, uses true stochastic random entropy.",
    )
    parser.add_argument(
        "--view-mode",
        choices=["parallel", "cross"],
        default="parallel",
        help="Viewing mode ('parallel' for standard Magic Eye, 'cross' for cross-eyed).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert depth values (swap foreground elevation and background depth).",
    )
    parser.add_argument(
        "--fusion-dots",
        action="store_true",
        help="Render two alignment fusion markers at the top of the autostereogram.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="autostereogram_outputs",
        help="Directory to store generated autostereograms (default: autostereogram_outputs).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        depth_path = find_depth_map(args.depthmap)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STEREO3D: STOCHASTIC AUTOSTEREOGRAM GENERATION")
    print("==================================================")
    print(f"Input Depth Map : {depth_path}")
    print(f"Color Theme     : {args.color}")
    print(f"Pattern Style   : {args.pattern}")
    print(f"Stochastic Seed : {args.seed if args.seed is not None else 'Random (Stochastic)'}")
    print(f"Viewing Mode    : {args.view_mode}")
    print("--------------------------------------------------")
    print("Synthesizing multi-octave stochastic noise and solving scanline linkages...")

    generator = AutostereogramGenerator(
        base_period=args.period,
        max_disparity=args.max_disparity,
        color_theme=args.color,
        pattern_style=args.pattern,
        view_mode=args.view_mode,
        seed=args.seed,
        invert_depth=args.invert,
        fusion_dots=args.fusion_dots,
    )

    result = generator.generate(depth_path)

    # Save outputs
    stem = depth_path.stem
    output_png = out_dir / f"{stem}_autostereogram.png"
    analysis_png = out_dir / f"{stem}_autostereogram_analysis.png"

    result.save_png(output_png)
    result.save_analysis(analysis_png, fusion_dots=True)

    print("--------------------------------------------------")
    print(f"Image Resolution: {result.width}x{result.height} (WxH)")
    print(f"Base Period S_0 : {result.base_period} px")
    print(f"Max Disparity   : {result.max_disparity} px")
    print(f"Resolved Seed   : {result.seed}")
    print("--------------------------------------------------")
    print("GENERATED OUTPUTS:")
    print(f"  Autostereogram Image : {output_png}")
    print(f"  Diagnostic Report    : {analysis_png}")
    print("==================================================")
    print("Generation completed successfully.")
    print("\nTo reconstruct and view the 3D surface model from this autostereogram, run:")
    print(f"  python main.py \"{output_png}\" --postprocess --view\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
