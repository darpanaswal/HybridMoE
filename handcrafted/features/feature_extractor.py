import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Any
from handcrafted.features.hog import HOGConfig, extract_hog
from handcrafted.hsv_roi import HSVSegmentationConfig, segment_sign_roi
from handcrafted.features.hough import HoughConfig, extract_hough_features
from handcrafted.features.hsv_stats import HSVStatsConfig, extract_hsv_stats
from handcrafted.features.sp_hog import SpatialPyramidHOGConfig, extract_spatial_pyramid_hog


@dataclass(frozen=True)
class FeatureExtractorConfig:
    # ROI segmentation
    roi_cfg: HSVSegmentationConfig = HSVSegmentationConfig()
    use_roi: bool = True  # NEW: allow disabling ROI cropping

    # Feature configs
    hsv_cfg: HSVStatsConfig = HSVStatsConfig()
    hog_cfg: HOGConfig = HOGConfig()
    hough_cfg: HoughConfig = HoughConfig()

    # NEW: Spatial pyramid HOG (1x1 + 2x2)
    use_sp_hog: bool = False
    sp_hog_cfg: SpatialPyramidHOGConfig = SpatialPyramidHOGConfig()

    # Output
    return_blocks: bool = False  # if True, include per-block vectors in output dict


def extract_handcrafted_features(
    image_bgr: np.ndarray,
    cfg: Optional[FeatureExtractorConfig] = None,
) -> Dict[str, Any]:
    """
    Pipeline:
      1) (optional) HSV ROI segmentation -> roi_bgr + bbox + success
      2) Extract HSV stats + HOG + Hough on roi_bgr (or full image if use_roi=False)
      3) Optionally add Spatial Pyramid HOG
      4) Concatenate to a single float32 vector
    """
    cfg = cfg or FeatureExtractorConfig()

    if cfg.use_roi:
        roi_out = segment_sign_roi(image_bgr, cfg.roi_cfg)
        roi_bgr = roi_out["roi_bgr"]
        roi_success = bool(roi_out["success"])
        bbox = roi_out["bbox"]
    else:
        roi_bgr = image_bgr
        roi_success = True
        h, w = image_bgr.shape[:2]
        bbox = (0, 0, w, h)

    hsv_blocks = extract_hsv_stats(roi_bgr, cfg.hsv_cfg)
    hog_vec = extract_hog(roi_bgr, cfg.hog_cfg)
    hough_blocks = extract_hough_features(roi_bgr, cfg.hough_cfg)

    blocks = [
        hsv_blocks["hsv_mean_std"],
        hsv_blocks["hsv_hist"],
        hsv_blocks["color_frac"],
        hog_vec,
        hough_blocks["geom"],
    ]

    sp_hog_vec = None
    if cfg.use_sp_hog:
        sp_hog_vec = extract_spatial_pyramid_hog(roi_bgr, cfg.sp_hog_cfg)
        blocks.append(sp_hog_vec)

    x = np.concatenate(blocks, axis=0).astype(np.float32)

    out: Dict[str, Any] = {
        "x": x,
        "roi_success": bool(roi_success),
        "bbox": bbox,
    }

    if cfg.return_blocks:
        blk: Dict[str, np.ndarray] = {
            "hsv_mean_std": hsv_blocks["hsv_mean_std"],
            "hsv_hist": hsv_blocks["hsv_hist"],
            "color_frac": hsv_blocks["color_frac"],
            "hog": hog_vec,
            "hough_geom": hough_blocks["geom"],
        }
        if sp_hog_vec is not None:
            blk["sp_hog"] = sp_hog_vec
        out["blocks"] = blk

    return out


def feature_dim(cfg: Optional[FeatureExtractorConfig] = None) -> int:
    """
    Returns the feature dimensionality for a given config.
    Computes it by running on a dummy image.
    """
    cfg = cfg or FeatureExtractorConfig()
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    return int(extract_handcrafted_features(dummy, cfg)["x"].shape[0])