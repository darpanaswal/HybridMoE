import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict



@dataclass(frozen=True)
class HoughConfig:
    resize: int = 96  # ROI resized to (resize, resize) before edge/Hough
    canny1: int = 80
    canny2: int = 160

    # Circles
    dp: float = 1.2
    min_dist: float = 16.0
    param1: float = 120.0
    param2: float = 25.0
    min_radius: int = 6
    max_radius: int = 60

    # Lines
    hough_threshold: int = 50
    min_line_length: int = 20
    max_line_gap: int = 10


def _validate_bgr(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape (H,W,3); got {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"image_bgr must be uint8; got {image_bgr.dtype}")


def extract_hough_features(
    roi_bgr: np.ndarray,
    cfg: Optional[HoughConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict of feature blocks (float32):
      - geom: [num_circles, mean_r, std_r, max_r, num_lines, mean_len, std_len, edge_density]
    """
    _validate_bgr(roi_bgr)
    cfg = cfg or HoughConfig()

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (cfg.resize, cfg.resize), interpolation=cv2.INTER_AREA)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, cfg.canny1, cfg.canny2)
    edge_density = float((edges > 0).mean())

    # Circles
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=cfg.dp,
        minDist=cfg.min_dist,
        param1=cfg.param1,
        param2=cfg.param2,
        minRadius=cfg.min_radius,
        maxRadius=cfg.max_radius,
    )

    if circles is None:
        num_circles = 0.0
        mean_r = 0.0
        std_r = 0.0
        max_r = 0.0
    else:
        c = circles[0]  # (N, 3)
        rs = c[:, 2].astype(np.float32)
        num_circles = float(len(rs))
        mean_r = float(rs.mean()) if len(rs) else 0.0
        std_r = float(rs.std()) if len(rs) else 0.0
        max_r = float(rs.max()) if len(rs) else 0.0

    # Lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.min_line_length,
        maxLineGap=cfg.max_line_gap,
    )

    if lines is None:
        num_lines = 0.0
        mean_len = 0.0
        std_len = 0.0
    else:
        lens = []
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, l.tolist())
            lens.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        lens = np.asarray(lens, dtype=np.float32)
        num_lines = float(len(lens))
        mean_len = float(lens.mean()) if len(lens) else 0.0
        std_len = float(lens.std()) if len(lens) else 0.0

    geom = np.array(
        [num_circles, mean_r, std_r, max_r, num_lines, mean_len, std_len, edge_density],
        dtype=np.float32,
    )

    return {"geom": geom}