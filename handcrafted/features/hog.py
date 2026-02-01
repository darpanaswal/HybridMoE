import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class HOGConfig:
    resize: int = 64  # ROI resized to (resize, resize)
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 2


def _validate_bgr(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape (H,W,3); got {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"image_bgr must be uint8; got {image_bgr.dtype}")


def _hog_skimage(gray_u8: np.ndarray, cfg: HOGConfig) -> np.ndarray:
    # Lazy import so project runs even if skimage isn't installed
    from skimage.feature import hog  # type: ignore

    feat = hog(
        gray_u8,
        orientations=cfg.orientations,
        pixels_per_cell=(cfg.pixels_per_cell, cfg.pixels_per_cell),
        cells_per_block=(cfg.cells_per_block, cfg.cells_per_block),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )
    return np.asarray(feat, dtype=np.float32)


def _hog_opencv(gray_u8: np.ndarray, cfg: HOGConfig) -> np.ndarray:
    """
    OpenCV HOGDescriptor fallback (feature length differs vs skimage but is consistent).
    """
    win = (cfg.resize, cfg.resize)
    block = (cfg.cells_per_block * cfg.pixels_per_cell, cfg.cells_per_block * cfg.pixels_per_cell)
    stride = (cfg.pixels_per_cell, cfg.pixels_per_cell)
    cell = (cfg.pixels_per_cell, cfg.pixels_per_cell)
    nbins = cfg.orientations

    hog = cv2.HOGDescriptor(win, block, stride, cell, nbins)
    feat = hog.compute(gray_u8)
    feat = feat.reshape(-1).astype(np.float32)
    return feat


def extract_hog(
    roi_bgr: np.ndarray,
    cfg: Optional[HOGConfig] = None,
) -> np.ndarray:
    """
    Returns HOG feature vector (float32).
    Uses skimage if available, otherwise OpenCV.
    """
    _validate_bgr(roi_bgr)
    cfg = cfg or HOGConfig()

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (cfg.resize, cfg.resize), interpolation=cv2.INTER_AREA)

    try:
        return _hog_skimage(gray, cfg)
    except Exception:
        return _hog_opencv(gray, cfg)