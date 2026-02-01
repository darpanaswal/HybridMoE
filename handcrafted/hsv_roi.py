import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


@dataclass(frozen=True)
class HSVRange:
    """
    OpenCV HSV ranges:
      H: [0, 179]
      S: [0, 255]
      V: [0, 255]
    """
    h_min: int
    s_min: int
    v_min: int
    h_max: int
    s_max: int
    v_max: int

    def as_lower_upper(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        return lower, upper


@dataclass(frozen=True)
class HSVSegmentationConfig:
    """
    Default thresholds are conservative for traffic-sign-like saturated colors.
    Tune per dataset lighting/camera if needed.

    red is special because hue wraps around at 0/179, so we support two ranges.
    """
    red1: HSVRange = HSVRange(0, 80, 50, 10, 255, 255)
    red2: HSVRange = HSVRange(170, 80, 50, 179, 255, 255)
    blue: HSVRange = HSVRange(95, 80, 50, 140, 255, 255)

    # mask cleanup
    morph_kernel: int = 5  # odd size recommended
    morph_open_iters: int = 1
    morph_close_iters: int = 2

    # bbox filtering
    min_area_frac: float = 0.001  # relative to image area
    max_area_frac: float = 0.95

    # fallback crop if segmentation fails
    fallback_center_crop_frac: float = 0.7  # fraction of min(H, W)


def _validate_bgr(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape (H,W,3); got {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"image_bgr must be uint8; got {image_bgr.dtype}")


def bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    _validate_bgr(image_bgr)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def build_color_mask(hsv: np.ndarray, cfg: HSVSegmentationConfig) -> np.ndarray:
    """
    Returns a binary mask (uint8) where candidate sign pixels are 255.
    """
    if hsv.ndim != 3 or hsv.shape[2] != 3:
        raise ValueError(f"hsv must have shape (H,W,3); got {hsv.shape}")

    r1_lo, r1_hi = cfg.red1.as_lower_upper()
    r2_lo, r2_hi = cfg.red2.as_lower_upper()
    b_lo, b_hi = cfg.blue.as_lower_upper()

    mask_r1 = cv2.inRange(hsv, r1_lo, r1_hi)
    mask_r2 = cv2.inRange(hsv, r2_lo, r2_hi)
    mask_b = cv2.inRange(hsv, b_lo, b_hi)

    return cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_b)


def clean_mask(mask: np.ndarray, cfg: HSVSegmentationConfig) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D; got {mask.shape}")

    k = int(cfg.morph_kernel)
    if k < 1:
        return mask
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out = mask
    if cfg.morph_open_iters > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=int(cfg.morph_open_iters))
    if cfg.morph_close_iters > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.morph_close_iters))

    return out


def largest_connected_component_bbox(mask: np.ndarray, cfg: HSVSegmentationConfig) -> Optional[BBox]:
    """
    Finds the largest contour and returns its bounding box, or None if no valid region.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D; got {mask.shape}")

    h, w = mask.shape[:2]
    img_area = float(h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue

        frac = area / img_area
        if frac < cfg.min_area_frac or frac > cfg.max_area_frac:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 1 or bh <= 1:
            continue

        if area > best_area:
            best_area = area
            best = (x, y, x + bw, y + bh)

    return best


def center_crop_bbox(h: int, w: int, frac: float) -> BBox:
    frac = float(frac)
    frac = max(0.05, min(1.0, frac))
    side = int(min(h, w) * frac)
    side = max(2, side)

    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    # ensure valid
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return (x1, y1, x2, y2)


def extract_roi(image_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    _validate_bgr(image_bgr)
    x1, y1, x2, y2 = bbox
    return image_bgr[y1:y2, x1:x2].copy()


def segment_sign_roi(
    image_bgr: np.ndarray,
    cfg: Optional[HSVSegmentationConfig] = None,
) -> Dict[str, object]:
    """
    Main entrypoint.

    Returns dict:
      {
        "success": bool,
        "bbox": (x1,y1,x2,y2),
        "mask": uint8(H,W),
        "roi_bgr": uint8(h,w,3),
        "hsv": uint8(H,W,3),
      }
    """
    _validate_bgr(image_bgr)
    cfg = cfg or HSVSegmentationConfig()

    hsv = bgr_to_hsv(image_bgr)
    mask = build_color_mask(hsv, cfg)
    mask = clean_mask(mask, cfg)

    bbox = largest_connected_component_bbox(mask, cfg)
    success = bbox is not None

    if not success:
        h, w = image_bgr.shape[:2]
        bbox = center_crop_bbox(h, w, cfg.fallback_center_crop_frac)

    roi = extract_roi(image_bgr, bbox)

    return {
        "success": bool(success),
        "bbox": bbox,
        "mask": mask,
        "roi_bgr": roi,
        "hsv": hsv,
    }