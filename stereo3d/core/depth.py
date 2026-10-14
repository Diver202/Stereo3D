from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

from .datastructures import StereogramImage
from .period import PeriodEstimator, estimate_base_period


@dataclass
class DepthMapResult:
    """
    Container for extracted depth map and disparity information.
    """

    depth: np.ndarray  # Shape: (H, W), float32 in [0.0, 1.0] (0 = background, 1 = foreground)
    disparity: np.ndarray  # Shape: (H, W), float32 in pixels
    base_period: int
    max_disparity: int
    confidence: np.ndarray  # Shape: (H, W), float32 in [0.0, 1.0]
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

    def save_png(self, path: Union[str, Path]) -> Path:
        """
        Save depth map as standard 8-bit grayscale PNG [0..255].
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        uint8_depth = (np.clip(self.depth, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(uint8_depth, mode="L").save(out_path)
        return out_path

    def save_16bit(self, path: Union[str, Path]) -> Path:
        """
        Save depth map as high-precision 16-bit grayscale PNG [0..65535].
        Optimal for 3D displacement maps and mesh heightfields.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        uint16_depth = (np.clip(self.depth, 0.0, 1.0) * 65535.0).astype(np.uint16)
        # cv2.imwrite correctly writes uint16 PNGs
        cv2.imwrite(str(out_path), uint16_depth)
        return out_path

    def save_color(self, path: Union[str, Path], cmap: str = "turbo") -> Path:
        """
        Save colorized depth map visualization using a colormap.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        colormap = plt.get_cmap(cmap)
        colored = colormap(np.clip(self.depth, 0.0, 1.0))[:, :, :3]
        colored_uint8 = (colored * 255.0).astype(np.uint8)
        Image.fromarray(colored_uint8, mode="RGB").save(out_path)
        return out_path

    def save_numpy(self, path: Union[str, Path]) -> Path:
        """
        Save raw floating-point depth array as NumPy .npy file.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self.depth.astype(np.float32))
        return out_path


class DepthExtractor:
    """
    Extracts dense depth maps from autostereograms using symmetric correspondence matching.

    Ensures 100% spatial coverage across all pixels from border to border without leaving
    blank margins.
    """

    def __init__(
        self,
        base_period: Optional[int] = None,
        max_disparity: Optional[int] = None,
        view_mode: Literal["parallel", "cross"] = "parallel",
        window_size: int = 7,
        subpixel: bool = True,
        median_radius: int = 7,
        bilateral_sigma_color: float = 3.0,
        bilateral_sigma_space: float = 7.0,
        robust_norm: bool = True,
        bg_prior: bool = True,
        bg_threshold: float = 0.88,
    ):
        """
        Parameters:
        -----------
        base_period: Fundamental repetition interval S_0 in pixels. If None, automatically estimated.
        max_disparity: Maximum horizontal shift in pixels. If None, set to min(int(0.35 * base_period), 75).
        view_mode: 'parallel' (standard) or 'cross' (cross-eyed).
        window_size: Spatial aggregation kernel size (odd integer).
        subpixel: Whether to apply quadratic subpixel refinement.
        median_radius: Median filter size for removing impulse noise (odd integer).
        bilateral_sigma_color: Bilateral filter color/intensity threshold.
        bilateral_sigma_space: Bilateral filter spatial threshold.
        robust_norm: Normalize relative to active foreground disparity instead of hard max_d ceiling.
        bg_prior: Suppress ambiguous false-positive matches in flat background regions.
        bg_threshold: Ratio threshold relative to background cost for ambiguity suppression.
        """
        self.base_period = base_period
        self.max_disparity = max_disparity
        self.view_mode = view_mode
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.subpixel = subpixel
        self.median_radius = median_radius if median_radius % 2 == 1 else median_radius + 1
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.robust_norm = robust_norm
        self.bg_prior = bg_prior
        self.bg_threshold = bg_threshold

    def extract(self, image: StereogramImage) -> DepthMapResult:
        """
        Extract dense depth map from a StereogramImage.
        """
        height, width = image.shape

        # 1. Determine base period
        period = self.base_period
        if period is None:
            period_result = estimate_base_period(image)
            period = period_result.period

        # 2. Determine disparity range
        max_d = self.max_disparity
        if max_d is None:
            max_d = min(int(round(period * 0.35)), 75)
            max_d = max(max_d, 15)

        max_d = min(max_d, period - 10)
        if max_d < 1:
            raise ValueError(f"Invalid max_disparity ({max_d}) for period ({period})")

        # 3. Construct 3-channel normalized float image
        rgb_float = image.rgb.astype(np.float32) / 255.0

        # Cost volume: (H, W, max_d + 1)
        cost_vol = np.zeros((height, width, max_d + 1), dtype=np.float32)

        for d in range(max_d + 1):
            if self.view_mode == "parallel":
                sep = period - d
            else:
                sep = period + d

            half = sep // 2
            s_left = half
            s_right = sep - half

            diff_rgb = np.zeros((height, width), dtype=np.float32)
            valid_cols = np.arange(s_left, width - s_right)

            if len(valid_cols) > 0:
                p_left = rgb_float[:, valid_cols - s_left, :]
                p_right = rgb_float[:, valid_cols + s_right, :]
                diff_rgb[:, valid_cols] = np.mean(np.abs(p_left - p_right), axis=2)

                # Boundary padding ensures complete coverage across leftmost and rightmost columns
                if s_left > 0:
                    diff_rgb[:, :s_left] = diff_rgb[:, s_left : s_left + 1]
                if s_right > 0:
                    diff_rgb[:, width - s_right :] = diff_rgb[:, width - s_right - 1 : width - s_right]

            # Spatial cost aggregation
            ksize = (self.window_size, self.window_size)
            cost_vol[:, :, d] = cv2.boxFilter(diff_rgb, ddepth=-1, ksize=ksize)

        # 4. Disparity selection (Winner-Takes-All)
        best_d = np.argmin(cost_vol, axis=2).astype(np.float32)

        # Background prior: suppress ambiguous non-zero disparities in background regions
        if self.bg_prior:
            cost_bg = cost_vol[:, :, 0]
            min_cost = np.min(cost_vol, axis=2)
            ambiguous = (min_cost >= self.bg_threshold * cost_bg) & (best_d > 4)
            best_d[ambiguous] = 0.0

        # Confidence map: difference between second lowest cost and lowest cost
        sorted_costs = np.sort(cost_vol, axis=2)
        min_cost = sorted_costs[:, :, 0]
        second_cost = sorted_costs[:, :, min(1, max_d)]
        confidence = np.clip((second_cost - min_cost) / (second_cost + 1e-6), 0.0, 1.0)

        # 5. Sub-pixel quadratic interpolation
        if self.subpixel and max_d >= 2:
            d_int = np.clip(best_d.astype(int), 1, max_d - 1)
            y_coords, x_coords = np.ogrid[:height, :width]
            c_prev = cost_vol[y_coords, x_coords, d_int - 1]
            c_curr = cost_vol[y_coords, x_coords, d_int]
            c_next = cost_vol[y_coords, x_coords, d_int + 1]
            denom = 2.0 * (c_prev - 2.0 * c_curr + c_next + 1e-6)
            delta = np.clip((c_prev - c_next) / denom, -0.5, 0.5)
            refined_d = np.clip(best_d + delta, 0.0, float(max_d))
        else:
            refined_d = best_d

        # 6. Post-processing: Median and Bilateral filtering
        if self.median_radius > 1:
            filtered_d = median_filter(refined_d, size=self.median_radius)
        else:
            filtered_d = refined_d

        if self.bilateral_sigma_color > 0 and self.bilateral_sigma_space > 0:
            filtered_d = cv2.bilateralFilter(
                filtered_d.astype(np.float32),
                d=9,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )

        # Normalized depth in [0.0, 1.0]
        if self.robust_norm:
            active_mask = filtered_d > 1.0
            if np.any(active_mask):
                d_max_active = float(np.percentile(filtered_d[active_mask], 99.0))
            else:
                d_max_active = float(max_d)
            d_max_active = max(d_max_active, 1.0)
            norm_depth = np.clip(filtered_d / d_max_active, 0.0, 1.0).astype(np.float32)
        else:
            norm_depth = np.clip(filtered_d / float(max_d), 0.0, 1.0).astype(np.float32)

        return DepthMapResult(
            depth=norm_depth,
            disparity=filtered_d.astype(np.float32),
            base_period=period,
            max_disparity=max_d,
            confidence=confidence.astype(np.float32),
            metadata={
                "view_mode": self.view_mode,
                "window_size": self.window_size,
                "subpixel": self.subpixel,
                "median_radius": self.median_radius,
            },
        )


def extract_depth_map(
    image: Union[StereogramImage, np.ndarray, str],
    base_period: Optional[int] = None,
    max_disparity: Optional[int] = None,
    view_mode: Literal["parallel", "cross"] = "parallel",
    window_size: int = 7,
    robust_norm: bool = True,
    bg_prior: bool = True,
) -> DepthMapResult:
    """
    Convenience function to extract dense depth map from an autostereogram image.
    """
    if isinstance(image, str):
        stereo_img = StereogramImage.from_file(image)
    elif isinstance(image, np.ndarray):
        stereo_img = StereogramImage.from_numpy(image)
    else:
        stereo_img = image

    extractor = DepthExtractor(
        base_period=base_period,
        max_disparity=max_disparity,
        view_mode=view_mode,
        window_size=window_size,
        robust_norm=robust_norm,
        bg_prior=bg_prior,
    )
    return extractor.extract(stereo_img)
