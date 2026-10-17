from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .datastructures import StereogramImage
from .depth import DepthMapResult


@dataclass
class PostprocessResult:
    """
    Container for post-processed depth map and diagnostic metadata.
    """

    depth: np.ndarray  # Shape: (H, W), float32 in [0.0, 1.0]
    spike_mask: np.ndarray  # Shape: (H, W), uint8 in {0, 255} (255 = flagged & inpainted)
    cropped_margins: Tuple[int, int]  # (left_margin, right_margin) dropped in pixels
    base_period: int
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

    @property
    def spike_percentage(self) -> float:
        return float(np.mean(self.spike_mask > 0) * 100.0)

    def save_png(self, path: Union[str, Path]) -> Path:
        """Save depth map as standard 8-bit grayscale PNG [0..255]."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        uint8_depth = (np.clip(self.depth, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(uint8_depth, mode="L").save(out_path)
        return out_path

    def save_16bit(self, path: Union[str, Path]) -> Path:
        """Save depth map as high-precision 16-bit uint16 grayscale PNG [0..65535]."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        uint16_depth = (np.clip(self.depth, 0.0, 1.0) * 65535.0).astype(np.uint16)
        cv2.imwrite(str(out_path), uint16_depth)
        return out_path

    def save_color(self, path: Union[str, Path], cmap: str = "turbo") -> Path:
        """Save colorized depth map visualization using a colormap."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        colormap = plt.get_cmap(cmap)
        colored = colormap(np.clip(self.depth, 0.0, 1.0))[:, :, :3]
        colored_uint8 = (colored * 255.0).astype(np.uint8)
        Image.fromarray(colored_uint8, mode="RGB").save(out_path)
        return out_path

    def save_numpy(self, path: Union[str, Path]) -> Path:
        """Save raw floating-point depth array as NumPy .npy file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self.depth.astype(np.float32))
        return out_path

    def save_mask(self, path: Union[str, Path]) -> Path:
        """Save the binary inpainting / spike mask as an 8-bit PNG."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.spike_mask, mode="L").save(out_path)
        return out_path


class DepthPostprocessor:
    """
    Dedicated post-processing engine for autostereogram depth maps.

    Addresses:
    1. Unrecoverable side bars: drops the non-overlapping horizontal margins
       (floor(S_0 / 2) on the left and ceil(S_0 / 2) on the right).
    2. Salt-and-pepper noise & medium spikes: multi-scale median kernel difference thresholding.
    3. Large false-match splotches: connected component area filtering to detect isolated
       elevated islands and small internal dropouts/holes.
    4. Edge-aware reconstruction: Navier-Stokes / Telea inpainting across all flagged pixels.
    5. Final edge-preserving bilateral smoothing to ensure continuous surfaces with sharp boundaries.
    """

    def __init__(
        self,
        drop_side_margins: bool = True,
        margin_factor: float = 0.5,
        median_scales: Tuple[int, ...] = (9, 19, 31),
        diff_thresholds: Tuple[float, ...] = (0.10, 0.15, 0.20),
        min_island_area: int = 300,
        min_hole_area: int = 300,
        inpaint_radius: int = 5,
        inpaint_method: Literal["navier_stokes", "telea"] = "navier_stokes",
        bilateral_sigma_color: float = 12.0,
        bilateral_sigma_space: float = 5.0,
    ):
        """
        Parameters:
        -----------
        drop_side_margins: Whether to drop the unrecoverable side bands (approx S_0 / 2 on each side).
        margin_factor: Fraction of the base period S_0 to crop on each side (default 0.5 = half period).
        median_scales: Kernel sizes for multi-scale median filtering.
        diff_thresholds: Deviation thresholds corresponding to median_scales to flag spike pixels.
        min_island_area: Connected component pixel area below which isolated elevated regions are flagged as noise splotches.
        min_hole_area: Connected component pixel area below which internal zero-depth holes are flagged for inpainting.
        inpaint_radius: Pixel neighborhood radius for OpenCV inpainting.
        inpaint_method: Inpainting algorithm ('navier_stokes' for PDE fluid flow, 'telea' for fast marching).
        bilateral_sigma_color: Bilateral filter intensity parameter for post-inpaint smoothing.
        bilateral_sigma_space: Bilateral filter spatial parameter for post-inpaint smoothing.
        """
        self.drop_side_margins = drop_side_margins
        self.margin_factor = margin_factor
        self.median_scales = tuple(k if k % 2 == 1 else k + 1 for k in median_scales)
        self.diff_thresholds = diff_thresholds
        self.min_island_area = min_island_area
        self.min_hole_area = min_hole_area
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = inpaint_method
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def postprocess(
        self,
        depth: np.ndarray,
        base_period: int,
        stereogram: Optional[StereogramImage] = None,
    ) -> Tuple[PostprocessResult, Optional[StereogramImage]]:
        """
        Execute full post-processing pipeline on an extracted depth map.

        Returns:
            Tuple of (PostprocessResult, Optional[cropped_stereogram])
        """
        raw_h, raw_w = depth.shape
        margin_left = 0
        margin_right = 0
        cropped_stereogram = stereogram

        # 1. Drop unrecoverable side margins
        if self.drop_side_margins and base_period > 0:
            margin_left = int(round(base_period * self.margin_factor))
            margin_right = base_period - margin_left

            # Safety check: keep at least 100 pixels in width
            if margin_left + margin_right < raw_w - 100:
                cur_depth = depth[:, margin_left : raw_w - margin_right].copy()
                if stereogram is not None:
                    cropped_stereogram = stereogram.crop(margin_left, margin_right)
            else:
                cur_depth = depth.copy()
                margin_left = 0
                margin_right = 0
        else:
            cur_depth = depth.copy()

        h, w = cur_depth.shape
        d_u8 = (np.clip(cur_depth, 0.0, 1.0) * 255.0).astype(np.uint8)

        # 2. Multi-scale median difference spike detection
        spike_mask = np.zeros((h, w), dtype=bool)
        for k, th in zip(self.median_scales, self.diff_thresholds):
            med = cv2.medianBlur(d_u8, k)
            diff = np.abs(d_u8.astype(np.float32) - med.astype(np.float32)) / 255.0
            spike_mask |= (diff > th)

        # 3. Isolated small island detection (false-match splotches in background)
        if self.min_island_area > 0:
            elevated = (cur_depth > 0.05).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(elevated)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.min_island_area:
                    spike_mask[labels == i] = True

        # 4. Small hole detection (dark dropouts inside elevated objects)
        if self.min_hole_area > 0:
            holes = (cur_depth <= 0.05).astype(np.uint8)
            num_labels_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes)
            for i in range(1, num_labels_h):
                x, y, comp_w, comp_h, area = stats_h[i, :]
                # Only flag holes strictly enclosed inside the image (not the global background border)
                if x > 0 and y > 0 and (x + comp_w) < w and (y + comp_h) < h:
                    if area < self.min_hole_area:
                        spike_mask[labels_h == i] = True

        # 5. Morphological dilation: expand mask by 1px so inpainting covers transition borders
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.dilate((spike_mask * 255).astype(np.uint8), kernel, iterations=1)

        # 6. Edge-aware Inpainting
        flags = (
            cv2.INPAINT_NS
            if self.inpaint_method.lower() == "navier_stokes"
            else cv2.INPAINT_TELEA
        )
        if np.any(mask_u8 > 0):
            inpainted_u8 = cv2.inpaint(
                d_u8,
                mask_u8,
                inpaintRadius=self.inpaint_radius,
                flags=flags,
            )
        else:
            inpainted_u8 = d_u8

        # 7. Edge-preserving bilateral smoothing
        if self.bilateral_sigma_color > 0 and self.bilateral_sigma_space > 0:
            smoothed_u8 = cv2.bilateralFilter(
                inpainted_u8,
                d=7,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )
        else:
            smoothed_u8 = inpainted_u8

        # 8. Final normalization into [0.0, 1.0]
        final_depth = smoothed_u8.astype(np.float32) / 255.0

        # Dynamic range re-anchoring: preserve zero background and crisp maximum
        d_min = float(final_depth.min())
        d_max = float(final_depth.max())
        if d_max > d_min + 1e-4:
            # If minimum is slightly above 0 due to smoothing, re-clamp background
            if d_min < 0.08:
                final_depth = np.clip((final_depth - d_min) / (d_max - d_min), 0.0, 1.0)

        result = PostprocessResult(
            depth=final_depth.astype(np.float32),
            spike_mask=mask_u8,
            cropped_margins=(margin_left, margin_right),
            base_period=base_period,
            metadata={
                "drop_side_margins": self.drop_side_margins,
                "margin_factor": self.margin_factor,
                "median_scales": self.median_scales,
                "diff_thresholds": self.diff_thresholds,
                "min_island_area": self.min_island_area,
                "min_hole_area": self.min_hole_area,
                "inpaint_method": self.inpaint_method,
            },
        )

        return result, cropped_stereogram


def postprocess_depth_map(
    depth_input: Union[DepthMapResult, np.ndarray],
    base_period: int,
    stereogram: Optional[StereogramImage] = None,
    drop_side_margins: bool = True,
    min_island_area: int = 300,
    inpaint_method: Literal["navier_stokes", "telea"] = "navier_stokes",
) -> Tuple[PostprocessResult, Optional[StereogramImage]]:
    """
    Convenience function to run post-processing on a depth map.
    """
    if isinstance(depth_input, DepthMapResult):
        raw_depth = depth_input.depth
        period = depth_input.base_period or base_period
    else:
        raw_depth = depth_input
        period = base_period

    postprocessor = DepthPostprocessor(
        drop_side_margins=drop_side_margins,
        min_island_area=min_island_area,
        inpaint_method=inpaint_method,
    )
    return postprocessor.postprocess(raw_depth, base_period=period, stereogram=stereogram)
