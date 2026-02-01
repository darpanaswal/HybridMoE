import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from handcrafted.hsv_roi import HSVSegmentationConfig, build_color_mask, clean_mask

@dataclass(frozen=True)
class HSVStatsConfig:
    # Histogram bins
    h_bins: int = 18
    s_bins: int = 8
    v_bins: int = 8

    # If True, normalize histograms to sum to 1
    normalize_hist: bool = True

    # reuse segmentation cfg for red/blue pixel fractions
    seg_cfg: HSVSegmentationConfig = HSVSegmentationConfig()


def _validate_bgr(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape (H,W,3); got {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"image_bgr must be uint8; got {image_bgr.dtype}")


def _safe_hist(x: np.ndarray, bins: int, rng: Tuple[int, int], normalize: bool) -> np.ndarray:
    hist = cv2.calcHist([x], [0], None, [bins], [rng[0], rng[1]])
    hist = hist.reshape(-1).astype(np.float32)
    if normalize:
        s = float(hist.sum())
        if s > 0:
            hist /= s
    return hist


def extract_hsv_stats(
    roi_bgr: np.ndarray,
    cfg: Optional[HSVStatsConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict of feature blocks (all float32):
      - hsv_mean_std: 6 dims (mean H,S,V + std H,S,V)
      - hsv_hist: h_bins + s_bins + v_bins dims
      - color_frac: 1 dim = fraction of pixels that match (red OR blue) in ROI
    """
    _validate_bgr(roi_bgr)
    cfg = cfg or HSVStatsConfig()

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mean_h, std_h = float(h.mean()), float(h.std())
    mean_s, std_s = float(s.mean()), float(s.std())
    mean_v, std_v = float(v.mean()), float(v.std())
    hsv_mean_std = np.array([mean_h, mean_s, mean_v, std_h, std_s, std_v], dtype=np.float32)

    h_hist = _safe_hist(h, cfg.h_bins, (0, 180), cfg.normalize_hist)
    s_hist = _safe_hist(s, cfg.s_bins, (0, 256), cfg.normalize_hist)
    v_hist = _safe_hist(v, cfg.v_bins, (0, 256), cfg.normalize_hist)
    hsv_hist = np.concatenate([h_hist, s_hist, v_hist], axis=0).astype(np.float32)

    mask = build_color_mask(hsv, cfg.seg_cfg)
    mask = clean_mask(mask, cfg.seg_cfg)
    frac = float((mask > 0).mean())
    color_frac = np.array([frac], dtype=np.float32)

    return {
        "hsv_mean_std": hsv_mean_std,
        "hsv_hist": hsv_hist,
        "color_frac": color_frac,
    }