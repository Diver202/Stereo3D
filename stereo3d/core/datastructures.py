from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
from PIL import Image


@dataclass
class StereogramImage:
    """
    Core data structure representing an autostereogram image.

    Stores normalized floating point representations as well as raw color channels,
    providing fast access for signal analysis and disparity computation.
    """

    rgb: np.ndarray  # Shape: (H, W, 3), dtype: uint8 or float32 [0..1]
    gray: np.ndarray  # Shape: (H, W), dtype: float32 in [0, 1]
    source_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Validate dimensions
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"RGB array must have shape (H, W, 3), got {self.rgb.shape}")
        if self.gray.ndim != 2:
            raise ValueError(f"Gray array must have shape (H, W), got {self.gray.shape}")
        if self.rgb.shape[:2] != self.gray.shape:
            raise ValueError(
                f"RGB dimensions {self.rgb.shape[:2]} do not match Gray dimensions {self.gray.shape}"
            )

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "StereogramImage":
        """
        Load an autostereogram image from disk.

        Handles alpha transparency removal, RGB conversion, and normalized grayscale generation.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        with Image.open(path) as img:
            # Convert RGBA / P / L to RGB
            rgb_img = img.convert("RGB")
            rgb_arr = np.array(rgb_img, dtype=np.uint8)

            # High-accuracy luminance weights (Rec. 709)
            # Y = 0.2126 R + 0.7152 G + 0.0722 B
            rgb_float = rgb_arr.astype(np.float32) / 255.0
            gray_arr = (
                0.2126 * rgb_float[:, :, 0]
                + 0.7152 * rgb_float[:, :, 1]
                + 0.0722 * rgb_float[:, :, 2]
            ).astype(np.float32)

            meta = {
                "format": img.format,
                "original_mode": img.mode,
                "size": img.size,  # (width, height)
            }

        return cls(
            rgb=rgb_arr,
            gray=gray_arr,
            source_path=path,
            metadata=meta,
        )

    @classmethod
    def from_numpy(
        cls,
        rgb_array: np.ndarray,
        source_path: Optional[Path] = None,
        metadata: Optional[dict] = None,
    ) -> "StereogramImage":
        """
        Create a StereogramImage directly from a NumPy array (H, W, 3) or (H, W).
        """
        if rgb_array.ndim == 2:
            # Grayscale provided: replicate to 3 channels
            gray = rgb_array.astype(np.float32)
            if gray.max() > 1.0:
                gray = gray / 255.0
            rgb = np.repeat((gray * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        elif rgb_array.ndim == 3 and rgb_array.shape[2] in (3, 4):
            rgb = rgb_array[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            rgb_float = rgb.astype(np.float32) / 255.0
            gray = (
                0.2126 * rgb_float[:, :, 0]
                + 0.7152 * rgb_float[:, :, 1]
                + 0.0722 * rgb_float[:, :, 2]
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported array shape: {rgb_array.shape}")

        return cls(
            rgb=rgb,
            gray=gray,
            source_path=source_path,
            metadata=metadata or {},
        )

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def get_row_slice(
        self,
        start_ratio: float = 0.45,
        end_ratio: float = 0.55,
        as_gray: bool = True,
    ) -> np.ndarray:
        """
        Extract a vertical slice of rows (e.g. the middle 10% of rows by default).
        """
        start_y = int(np.clip(start_ratio * self.height, 0, self.height - 1))
        end_y = max(start_y + 1, int(np.clip(end_ratio * self.height, 0, self.height)))
        target = self.gray if as_gray else self.rgb
        return target[start_y:end_y]

    def compute_difference_map(self, lag: int) -> np.ndarray:
        """
        Computes the horizontal difference map shifted by `lag`:
        |I(x, y) - I(x + lag, y)|

        Where repeating patterns match, the difference approaches 0.
        Where depth introduces disparities, residual contours emerge.

        Returns an array of shape (H, W - lag) with absolute difference values in [0.0, 1.0].
        """
        if lag <= 0 or lag >= self.width:
            raise ValueError(f"Lag must be between 1 and width - 1 ({self.width - 1}), got {lag}")

        diff = np.abs(self.gray[:, lag:] - self.gray[:, :-lag])
        return diff

    def crop(self, margin_left: int, margin_right: int) -> "StereogramImage":
        """
        Crop horizontal margins (e.g. dropping unrecoverable side bands).
        """
        if margin_left < 0 or margin_right < 0:
            raise ValueError("Margins must be non-negative.")
        if margin_left + margin_right >= self.width:
            raise ValueError(
                f"Total margins ({margin_left + margin_right}) exceed image width ({self.width})."
            )

        rgb_c = self.rgb[:, margin_left : self.width - margin_right, :].copy()
        gray_c = self.gray[:, margin_left : self.width - margin_right].copy()
        meta = dict(self.metadata)
        meta["cropped_margins"] = (margin_left, margin_right)
        return StereogramImage(
            rgb=rgb_c,
            gray=gray_c,
            source_path=self.source_path,
            metadata=meta,
        )

    def __repr__(self) -> str:
        name = self.source_path.name if self.source_path else "in-memory"
        return f"<StereogramImage '{name}' shape=({self.height}x{self.width})>"
