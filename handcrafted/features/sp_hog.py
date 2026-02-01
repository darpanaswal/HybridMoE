import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass
from handcrafted.features.hog import HOGConfig, extract_hog


@dataclass(frozen=True)
class SpatialPyramidHOGConfig:
    """
    Spatial Pyramid HOG:
      - level 0: whole ROI
      - level 1: 2x2 grid patches
    The final feature is [HOG(whole) || HOG(patch_00) || ... || HOG(patch_11)]
    """
    hog_cfg: HOGConfig = HOGConfig()
    grid: int = 2  # fixed 2x2 for now


def _validate_bgr(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape (H,W,3); got {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"image_bgr must be uint8; got {image_bgr.dtype}")


def extract_spatial_pyramid_hog(
    roi_bgr: np.ndarray,
    cfg: Optional[SpatialPyramidHOGConfig] = None,
) -> np.ndarray:
    _validate_bgr(roi_bgr)
    cfg = cfg or SpatialPyramidHOGConfig()

    # Level 0: whole
    feats = [extract_hog(roi_bgr, cfg.hog_cfg)]

    # Level 1: 2x2 patches after resizing to a stable size (hog_cfg.resize)
    # We resize the ROI to (R,R) and then split into equal quadrants.
    R = int(cfg.hog_cfg.resize)
    grid = int(cfg.grid)
    if grid != 2:
        raise ValueError("Only grid=2 is supported currently.")

    roi_resized = cv2.resize(roi_bgr, (R, R), interpolation=cv2.INTER_AREA)
    step = R // grid

    for gy in range(grid):
        for gx in range(grid):
            y1 = gy * step
            x1 = gx * step
            # last patch takes remainder if R not divisible
            y2 = (gy + 1) * step if gy < grid - 1 else R
            x2 = (gx + 1) * step if gx < grid - 1 else R
            patch = roi_resized[y1:y2, x1:x2].copy()
            feats.append(extract_hog(patch, cfg.hog_cfg))

    return np.concatenate(feats, axis=0).astype(np.float32)