"""
Synthetic autostereogram generator for testing and validation.
Implements the classic Single Image Random Dot Stereogram (SIRDS) algorithm.
"""

from typing import Tuple
import numpy as np


def generate_synthetic_stereogram(
    width: int = 640,
    height: int = 480,
    base_period: int = 120,
    shape_type: str = "circle",
    max_disparity: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic autostereogram with known ground-truth base period and depth map.

    Returns:
    --------
    (stereogram_rgb, depth_map) where:
    - stereogram_rgb has shape (H, W, 3), dtype uint8
    - depth_map has shape (H, W), values in [0.0, 1.0]
    """
    # 1. Create depth map (0.0 = background / infinite depth, 1.0 = closest foreground)
    y, x = np.ogrid[:height, :width]
    depth_map = np.zeros((height, width), dtype=np.float32)

    cx, cy = width // 2, height // 2
    if shape_type == "circle":
        radius = min(width, height) // 4
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist < radius
        # Hemisphere depth
        depth_map[mask] = np.sqrt(np.maximum(0.0, 1.0 - (dist[mask] / radius) ** 2))
    elif shape_type == "box":
        bx, by = width // 6, height // 6
        depth_map[cy - by : cy + by, cx - bx : cx + bx] = 0.8
    else:
        # Default gradient ramp
        depth_map = np.clip((width - np.abs(x - cx)) / width, 0.0, 1.0)

    # 2. Compute pixel-to-pixel disparity links
    # For parallel viewing: separation S(x, y) = base_period - (depth * max_disparity)
    # Using classic Thimbleby / Inglis stereogram algorithm:
    stereogram = np.zeros((height, width, 3), dtype=np.uint8)

    rng = np.random.default_rng(42)

    for row_idx in range(height):
        # same[x] stores the linked pixel index
        same = np.arange(width)
        for col_idx in range(width):
            depth_val = depth_map[row_idx, col_idx]
            separation = int(round(base_period - depth_val * max_disparity))
            left = col_idx - separation // 2
            right = left + separation

            if 0 <= left and right < width:
                # Find root for left
                r_left = left
                while same[r_left] != r_left:
                    r_left = same[r_left]

                # Find root for right
                r_right = right
                while same[r_right] != r_right:
                    r_right = same[r_right]

                if r_left != r_right:
                    same[max(r_left, r_right)] = min(r_left, r_right)

        # Assign random colors to linked clusters
        row_colors = np.zeros((width, 3), dtype=np.uint8)
        for col_idx in range(width):
            root = col_idx
            while same[root] != root:
                root = same[root]

            if root == col_idx:
                # Root pixel gets a new random color
                row_colors[col_idx] = rng.integers(0, 256, size=3, dtype=np.uint8)
            else:
                row_colors[col_idx] = row_colors[root]

        stereogram[row_idx] = row_colors

    return stereogram, depth_map
