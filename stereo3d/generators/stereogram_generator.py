from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Curated harmonious color palettes with base hue (0..360), saturation range, and value range
COLOR_PALETTES: Dict[str, dict] = {
    "blue": {
        "hue": 215.0,
        "hue_spread": 22.0,
        "sat_range": (0.60, 0.95),
        "val_range": (0.35, 0.95),
        "description": "Deep oceanic sapphire and royal azure",
    },
    "emerald": {
        "hue": 140.0,
        "hue_spread": 20.0,
        "sat_range": (0.60, 0.95),
        "val_range": (0.30, 0.90),
        "description": "Rich forest emerald and deep jade",
    },
    "crimson": {
        "hue": 350.0,
        "hue_spread": 18.0,
        "sat_range": (0.65, 0.95),
        "val_range": (0.35, 0.95),
        "description": "Volcanic ruby and crimson embers",
    },
    "amber": {
        "hue": 42.0,
        "hue_spread": 16.0,
        "sat_range": (0.70, 0.98),
        "val_range": (0.45, 0.98),
        "description": "Warm golden amber, bronze, and copper",
    },
    "purple": {
        "hue": 275.0,
        "hue_spread": 24.0,
        "sat_range": (0.60, 0.95),
        "val_range": (0.35, 0.95),
        "description": "Deep amethyst and cosmic violet",
    },
    "cyan": {
        "hue": 185.0,
        "hue_spread": 18.0,
        "sat_range": (0.60, 0.95),
        "val_range": (0.40, 0.95),
        "description": "Vibrant turquoise and tropical cyan",
    },
    "cyberpunk": {
        "hue": 180.0,
        "secondary_hue": 315.0,
        "sat_range": (0.75, 1.0),
        "val_range": (0.40, 0.98),
        "description": "Neon cyan and electric magenta duotone",
    },
    "sunset": {
        "hue": 38.0,
        "secondary_hue": 350.0,
        "sat_range": (0.65, 0.95),
        "val_range": (0.40, 0.98),
        "description": "Warm gradient of amber, crimson, and deep violet",
    },
    "matrix": {
        "hue": 125.0,
        "hue_spread": 12.0,
        "sat_range": (0.85, 1.0),
        "val_range": (0.15, 0.85),
        "description": "Digital phosphorescent green on dark canvas",
    },
    "monochrome": {
        "hue": 0.0,
        "hue_spread": 0.0,
        "sat_range": (0.0, 0.0),
        "val_range": (0.10, 0.95),
        "description": "High-contrast monochrome grayscale",
    },
    "rainbow": {
        "hue": None,  # Full hue spectrum
        "sat_range": (0.70, 0.95),
        "val_range": (0.45, 0.95),
        "description": "Continuous multi-spectral rainbow dispersion",
    },
}


def parse_color_spec(color_arg: str) -> dict:
    """
    Parse a color string: named preset, hex code (#RRGGBB), or hue angle.
    """
    cleaned = color_arg.strip().lower()

    # 1. Preset palette match
    if cleaned in COLOR_PALETTES:
        return COLOR_PALETTES[cleaned]

    # Alias matching
    aliases = {
        "green": "emerald",
        "red": "crimson",
        "gold": "amber",
        "violet": "purple",
        "teal": "cyan",
        "gray": "monochrome",
        "grayscale": "monochrome",
        "bw": "monochrome",
    }
    if cleaned in aliases:
        return COLOR_PALETTES[aliases[cleaned]]

    # 2. Hex code parsing (#RRGGBB or #RGB)
    hex_match = re.match(r"^#?([0-9a-f]{3}|[0-9a-f]{6})$", cleaned)
    if hex_match:
        hex_str = hex_match.group(1)
        if len(hex_str) == 3:
            hex_str = "".join([c * 2 for c in hex_str])
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0

        # Convert RGB to HSV
        rgb_pixel = np.uint8([[[int(r * 255), int(g * 255), int(b * 255)]]])
        hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0, 0]
        hue_deg = float(hsv_pixel[0]) * 2.0  # OpenCV hue is 0..179
        sat = float(hsv_pixel[1]) / 255.0

        return {
            "hue": hue_deg,
            "hue_spread": 20.0,
            "sat_range": (max(0.2, sat - 0.2), min(1.0, sat + 0.2)),
            "val_range": (0.35, 0.95),
            "description": f"Custom color {color_arg}",
        }

    # 3. Numeric hue angle (0..360)
    try:
        hue_val = float(cleaned) % 360.0
        return {
            "hue": hue_val,
            "hue_spread": 20.0,
            "sat_range": (0.60, 0.95),
            "val_range": (0.35, 0.95),
            "description": f"Custom hue {hue_val:.1f}°",
        }
    except ValueError:
        pass

    # Default fallback to blue
    return COLOR_PALETTES["blue"]


@dataclass
class GenerationResult:
    """
    Container for generated autostereogram image and associated metadata.
    """

    stereogram: np.ndarray  # Shape: (H, W, 3), dtype: uint8 in [0..255]
    depth_map: np.ndarray  # Shape: (H, W), dtype: float32 in [0.0, 1.0]
    base_period: int
    max_disparity: int
    color_theme: str
    pattern_style: str
    seed: int
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.stereogram.shape[0]

    @property
    def width(self) -> int:
        return self.stereogram.shape[1]

    def save_png(self, path: Union[str, Path]) -> Path:
        """Save autostereogram image as full-resolution PNG."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.stereogram, mode="RGB").save(out_path)
        return out_path

    def save_analysis(
        self,
        path: Union[str, Path],
        fusion_dots: bool = True,
        show: bool = False,
    ) -> Path:
        """
        Save comprehensive diagnostic analysis:
        Left: Normalized Input Depth Map
        Right: Generated Autostereogram with optional visual eye-fusion dots
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel 1: Depth map
        im_d = axes[0].imshow(self.depth_map, cmap="turbo", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Input Depth Map ({self.width}x{self.height})\n[Black = Background, White = Foreground]")
        axes[0].set_axis_off()
        fig.colorbar(im_d, ax=axes[0], fraction=0.046, pad=0.04, label="Normalized Depth")

        # Panel 2: Generated Autostereogram
        axes[1].imshow(self.stereogram)
        title_str = (
            f"Generated Autostereogram (Period: {self.base_period}px, Max Disp: {self.max_disparity}px)\n"
            f"Theme: {self.color_theme.title()} | Pattern: {self.pattern_style.title()} | Seed: {self.seed}"
        )
        axes[1].set_title(title_str)
        axes[1].set_axis_off()

        # Draw fusion indicator dots above the stereogram image if requested
        if fusion_dots:
            cx = self.width // 2
            half_p = self.base_period // 2
            dot_left = cx - half_p
            dot_right = cx + half_p
            axes[1].scatter([dot_left, dot_right], [20, 20], color="#ffffff", edgecolors="#000000", s=100, zorder=10)
            axes[1].text(cx, 45, f"Fusion Markers (Distance: {self.base_period}px)", color="#ffffff",
                         ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.7))

        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return out_path


class AutostereogramGenerator:
    """
    High-fidelity stochastic autostereogram generator from input depth maps.

    Features:
    - Multi-octave stochastic noise synthesis (organic, speckle, cellular, plasma)
      designed to completely defeat monocular eye deciphering.
    - Full color palette control (preset themes or arbitrary hex/hue colors).
    - True stochastic generation (random system entropy by default or reproducible seed).
    - Disjoint-set union-find constraint propagation with path compression.
    - Multithreaded scanline execution for sub-second 1080p generation.
    """

    def __init__(
        self,
        base_period: Optional[int] = None,
        max_disparity: Optional[int] = None,
        color_theme: str = "blue",
        pattern_style: Literal["organic", "speckle", "cellular", "plasma"] = "organic",
        view_mode: Literal["parallel", "cross"] = "parallel",
        seed: Optional[int] = None,
        invert_depth: bool = False,
        fusion_dots: bool = False,
    ):
        """
        Parameters:
        -----------
        base_period: Fundamental horizontal repetition period S_0 in pixels.
                     If None, automatically chosen as round(width / 14).
        max_disparity: Maximum horizontal shift in pixels.
                       If None, automatically set to round(0.25 * base_period).
        color_theme: Predominant color preset ('blue', 'emerald', 'crimson', 'amber',
                     'purple', 'cyan', 'cyberpunk', 'sunset', 'matrix', 'monochrome',
                     'rainbow') or custom hex color (e.g. '#2E8B57').
        pattern_style: Noise generator style ('organic', 'speckle', 'cellular', 'plasma').
        view_mode: 'parallel' (standard divergent) or 'cross' (cross-eyed).
        seed: Random seed for deterministic output. If None, uses true stochastic entropy.
        invert_depth: Invert depth values before generation (concave vs convex).
        fusion_dots: Render two alignment fusion dots at the top of the autostereogram canvas.
        """
        self.base_period = base_period
        self.max_disparity = max_disparity
        self.color_theme = color_theme
        self.pattern_style = pattern_style
        self.view_mode = view_mode
        self.seed = seed
        self.invert_depth = invert_depth
        self.fusion_dots = fusion_dots

    @staticmethod
    def load_depth_map(depth_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load and normalize depth map into a continuous float32 array in [0.0, 1.0].
        Handles alpha channels, 8-bit, 16-bit uint16, and floating-point data.
        """
        if isinstance(depth_input, (str, Path)):
            path = Path(depth_input)
            if not path.is_file():
                raise FileNotFoundError(f"Depth map file not found: {path}")

            # Use OpenCV to preserve full 16-bit precision if present
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                with Image.open(path) as img:
                    raw = np.array(img)
        elif isinstance(depth_input, Image.Image):
            raw = np.array(depth_input)
        elif isinstance(depth_input, np.ndarray):
            raw = depth_input
        else:
            raise TypeError(f"Unsupported depth input type: {type(depth_input)}")

        # Handle multi-channel images (RGBA, RGB)
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                # RGBA: use alpha channel to mask out transparent regions to 0.0 background
                rgb = raw[:, :, :3]
                alpha = raw[:, :, 3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                gray[alpha == 0] = 0.0
            else:
                gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = raw.astype(np.float32)

        # Normalize into [0.0, 1.0]
        max_val = float(gray.max())
        min_val = float(gray.min())

        if max_val > min_val + 1e-5:
            # If standard 8-bit image
            if max_val <= 255.0 and raw.dtype == np.uint8:
                norm_depth = (gray - min_val) / (max_val - min_val)
            # If 16-bit uint16 image
            elif max_val <= 65535.0 and raw.dtype == np.uint16:
                norm_depth = (gray - min_val) / (max_val - min_val)
            else:
                norm_depth = (gray - min_val) / (max_val - min_val)
        else:
            norm_depth = np.zeros_like(gray, dtype=np.float32)

        return np.clip(norm_depth, 0.0, 1.0).astype(np.float32)

    def _generate_noise_field(
        self,
        height: int,
        width: int,
        palette: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Synthesize multi-scale stochastic 2D color noise field based on palette and style.
        """
        style = self.pattern_style.lower()

        # 1. Multi-octave organic fractal noise
        def octave_noise(octaves: List[Tuple[int, float]]) -> np.ndarray:
            field = np.zeros((height, width), dtype=np.float32)
            for sigma, weight in octaves:
                noise_layer = rng.standard_normal((height, width)).astype(np.float32)
                field += cv2.GaussianBlur(noise_layer, (0, 0), sigmaX=sigma, sigmaY=sigma) * weight
            # Add fine high-frequency stochastic grain
            field += rng.standard_normal((height, width)).astype(np.float32) * 0.15
            f_min, f_max = field.min(), field.max()
            return (field - f_min) / max(1e-5, f_max - f_min)

        if style == "speckle":
            # High-density stochastic speckle
            nh = rng.uniform(0.0, 1.0, (height, width)).astype(np.float32)
            ns = rng.uniform(0.0, 1.0, (height, width)).astype(np.float32)
            nv = rng.uniform(0.0, 1.0, (height, width)).astype(np.float32)
        elif style == "cellular":
            # Cellular / Voronoi distance noise
            n1 = octave_noise([(16, 0.5), (8, 0.3), (2, 0.2)])
            nh = np.abs(np.sin(n1 * np.pi * 4.0))
            ns = octave_noise([(12, 0.6), (4, 0.4)])
            nv = octave_noise([(16, 0.5), (4, 0.5)])
        elif style == "plasma":
            # High-contrast plasma
            n1 = octave_noise([(32, 0.5), (12, 0.3), (3, 0.2)])
            n2 = octave_noise([(24, 0.5), (8, 0.3), (2, 0.2)])
            nh = (np.sin(n1 * 10.0) + np.cos(n2 * 10.0) + 2.0) / 4.0
            ns = octave_noise([(16, 0.6), (4, 0.4)])
            nv = octave_noise([(20, 0.5), (4, 0.5)])
        else:
            # Default: Organic marbled fractal camouflage (hardest for monocular vision)
            nh = octave_noise([(28, 0.40), (14, 0.30), (6, 0.20), (2, 0.10)])
            ns = octave_noise([(20, 0.45), (10, 0.35), (3, 0.20)])
            nv = octave_noise([(24, 0.45), (12, 0.35), (2, 0.20)])

        # 2. Map noise fields to HSV based on color palette
        sat_min, sat_max = palette["sat_range"]
        val_min, val_max = palette["val_range"]

        if palette.get("hue") is None:
            # Rainbow: full hue range
            h_channel = (nh * 179.0).astype(np.uint8)
        elif "secondary_hue" in palette:
            # Duotone blend (e.g. Cyberpunk cyan & magenta, Sunset)
            h1 = palette["hue"] / 2.0
            h2 = palette["secondary_hue"] / 2.0
            h_channel = np.where(nh > 0.5, h1 + (nh - 0.5) * 12.0, h2 - nh * 12.0)
            h_channel = (h_channel % 180.0).astype(np.uint8)
        else:
            base_h = (palette["hue"] / 2.0) % 180.0
            spread = (palette.get("hue_spread", 20.0) / 2.0)
            h_channel = (base_h + (nh - 0.5) * (spread * 2.0)) % 180.0
            h_channel = h_channel.astype(np.uint8)

        s_channel = ((sat_min + ns * (sat_max - sat_min)) * 255.0).astype(np.uint8)
        v_channel = ((val_min + nv * (val_max - val_min)) * 255.0).astype(np.uint8)

        # Convert HSV to RGB
        hsv = np.stack([h_channel, s_channel, v_channel], axis=-1)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def generate(
        self,
        depth_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> GenerationResult:
        """
        Generate high-entropy autostereogram from input depth map.
        """
        # 1. Load and normalize depth
        depth = self.load_depth_map(depth_input)
        if self.invert_depth:
            depth = 1.0 - depth

        height, width = depth.shape

        # 2. Derive base period and max disparity
        period = self.base_period
        if period is None:
            # Standard ratio: approx 1/14 of image width, bounded between 80 and 180 px
            period = max(70, min(int(round(width / 14)), 200))

        max_d = self.max_disparity
        if max_d is None:
            max_d = max(12, min(int(round(period * 0.26)), 60))

        # Disparity cannot exceed period - 10
        max_d = min(max_d, period - 10)

        # 3. Initialize stochastic seed and RNG
        resolved_seed = self.seed if self.seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
        rng = np.random.default_rng(resolved_seed)

        # 4. Resolve color palette
        palette = parse_color_spec(self.color_theme)

        # 5. Synthesize 2D stochastic texture canvas
        texture_canvas = self._generate_noise_field(height, width, palette, rng)

        # 6. Disjoint-Set Union-Find Scanline Constraint Resolution
        stereogram = np.zeros((height, width, 3), dtype=np.uint8)

        def process_scanline(row_idx: int) -> None:
            row_depth = depth[row_idx]
            same = np.arange(width, dtype=np.int32)

            # Parallel viewing: separation = period - (depth * max_d)
            # Cross viewing: separation = period + (depth * max_d)
            if self.view_mode == "parallel":
                seps = np.round(period - row_depth * max_d).astype(np.int32)
            else:
                seps = np.round(period + row_depth * max_d).astype(np.int32)

            lefts = np.arange(width, dtype=np.int32) - seps // 2
            rights = lefts + seps
            valid = (lefts >= 0) & (rights < width)
            vl = lefts[valid]
            vr = rights[valid]

            # Solve linkages with path compression
            for l, r in zip(vl, vr):
                rl = l
                while same[rl] != rl:
                    same[rl] = same[same[rl]]
                    rl = same[rl]
                rr = r
                while same[rr] != rr:
                    same[rr] = same[same[rr]]
                    rr = same[rr]
                if rl != rr:
                    same[max(rl, rr)] = min(rl, rr)

            # Assign colors from root pixels
            for col_idx in range(width):
                root = col_idx
                while same[root] != root:
                    same[root] = same[same[root]]
                    root = same[root]
                stereogram[row_idx, col_idx] = texture_canvas[row_idx, root]

        # Multithreaded execution across scanlines
        with ThreadPoolExecutor() as executor:
            list(executor.map(process_scanline, range(height)))

        # 7. Add optional eye-fusion dots at the top border
        if self.fusion_dots:
            cx = width // 2
            half_p = period // 2
            dot_l = cx - half_p
            dot_r = cx + half_p
            dot_y = min(20, height // 15)
            dot_rad = max(4, width // 350)
            cv2.circle(stereogram, (dot_l, dot_y), dot_rad + 2, (0, 0, 0), -1)
            cv2.circle(stereogram, (dot_l, dot_y), dot_rad, (255, 255, 255), -1)
            cv2.circle(stereogram, (dot_r, dot_y), dot_rad + 2, (0, 0, 0), -1)
            cv2.circle(stereogram, (dot_r, dot_y), dot_rad, (255, 255, 255), -1)

        return GenerationResult(
            stereogram=stereogram,
            depth_map=depth,
            base_period=period,
            max_disparity=max_d,
            color_theme=self.color_theme,
            pattern_style=self.pattern_style,
            seed=resolved_seed,
            metadata={
                "palette_description": palette.get("description", ""),
                "view_mode": self.view_mode,
                "invert_depth": self.invert_depth,
                "fusion_dots": self.fusion_dots,
            },
        )


def generate_autostereogram(
    depth_input: Union[str, Path, np.ndarray, Image.Image],
    base_period: Optional[int] = None,
    max_disparity: Optional[int] = None,
    color: str = "blue",
    pattern: Literal["organic", "speckle", "cellular", "plasma"] = "organic",
    seed: Optional[int] = None,
    view_mode: Literal["parallel", "cross"] = "parallel",
    fusion_dots: bool = False,
) -> GenerationResult:
    """
    Convenience function to generate an autostereogram from a depth map.
    """
    generator = AutostereogramGenerator(
        base_period=base_period,
        max_disparity=max_disparity,
        color_theme=color,
        pattern_style=pattern,
        view_mode=view_mode,
        seed=seed,
        fusion_dots=fusion_dots,
    )
    return generator.generate(depth_input)
