# extract_features.py
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import config as cfg
from dataclasses import dataclass
from torchvision.datasets import GTSRB
from sklearn.cluster import MiniBatchKMeans
from typing import Any, Dict, Optional, Tuple, List
from handcrafted.features.feature_extractor import (
    FeatureExtractorConfig,
    extract_handcrafted_features,
    feature_dim as base_feature_dim,
)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def bgr_from_pil(pil_img) -> np.ndarray:
    rgb = np.array(pil_img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image; got {rgb.shape}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr.astype(np.uint8)


def load_gtsrb_split(data_dir: str, split: str):
    return GTSRB(root=data_dir, split=split, download=False, transform=None)


# -----------------------------
# Extras: LBP + Color histogram
# -----------------------------
@dataclass(frozen=True)
class ExtraFeatureConfig:
    # LBP
    use_lbp: bool = True
    lbp_resize: int = 64
    lbp_bins: int = 256  # fixed 8-bit LBP

    # Color hist
    use_color_hist: bool = True
    color_resize: int = 64
    color_bins: int = 16
    color_space: str = "hsv"  # hsv|rgb
    color_hist_norm: str = "l1"  # l1|none

    # BoVW
    use_bovw: bool = True
    bovw_type: str = "dsift"  # dsift|orb
    bovw_vocab_size: int = 128
    bovw_sample_images: int = 16000
    bovw_max_desc: int = 800000
    bovw_orb_max_kp: int = 1500
    bovw_orb_fast_threshold: int = 10

    # Dense-SIFT options
    dsift_step: int = 6
    dsift_kp_size: int = 8
    dsift_resize: int = 64


def extra_feature_dim(extra: ExtraFeatureConfig, actual_bovw_k: Optional[int] = None) -> int:
    d = 0
    if extra.use_lbp:
        d += int(extra.lbp_bins)
    if extra.use_color_hist:
        d += int(extra.color_bins) * 3
    if extra.use_bovw:
        d += int(actual_bovw_k if actual_bovw_k is not None else extra.bovw_vocab_size)
    return d


def lbp_histogram(bgr: np.ndarray, resize: int, bins: int) -> np.ndarray:
    if bins != 256:
        raise ValueError("This LBP implementation supports bins=256 only.")
    img = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    g = gray
    center = g[1:-1, 1:-1]

    n0 = g[0:-2, 0:-2]
    n1 = g[0:-2, 1:-1]
    n2 = g[0:-2, 2:]
    n3 = g[1:-1, 2:]
    n4 = g[2:, 2:]
    n5 = g[2:, 1:-1]
    n6 = g[2:, 0:-2]
    n7 = g[1:-1, 0:-2]

    lbp = (
        ((n0 >= center).astype(np.uint8) << 7)
        | ((n1 >= center).astype(np.uint8) << 6)
        | ((n2 >= center).astype(np.uint8) << 5)
        | ((n3 >= center).astype(np.uint8) << 4)
        | ((n4 >= center).astype(np.uint8) << 3)
        | ((n5 >= center).astype(np.uint8) << 2)
        | ((n6 >= center).astype(np.uint8) << 1)
        | ((n7 >= center).astype(np.uint8) << 0)
    )

    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def color_histogram(bgr: np.ndarray, resize: int, bins: int, color_space: str, norm: str) -> np.ndarray:
    img = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_AREA)

    if color_space.lower() == "hsv":
        x = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space.lower() == "rgb":
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown color_space: {color_space}")

    feats: List[np.ndarray] = []
    for c in range(3):
        h = cv2.calcHist([x], [c], None, [bins], [0, 256]).reshape(-1).astype(np.float32)
        feats.append(h)

    feat = np.concatenate(feats, axis=0)
    if norm == "l1":
        s = float(feat.sum())
        if s > 0:
            feat /= s
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return feat


# -----------------------------
# BoVW: ORB and Dense-SIFT
# -----------------------------
def orb_descriptors(bgr: np.ndarray, resize: int, max_kp: int, fast_threshold: int) -> np.ndarray:
    img = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=int(max_kp), fastThreshold=int(fast_threshold))
    _kps, des = orb.detectAndCompute(gray, None)
    if des is None:
        return np.zeros((0, 32), dtype=np.float32)
    return des.astype(np.float32)


def dsift_descriptors(bgr: np.ndarray, resize: int, step: int, kp_size: int) -> np.ndarray:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT not available (cv2.SIFT_create missing). Install opencv-contrib-python.")

    img = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[:2]
    step = max(2, int(step))
    kp_size = max(2, int(kp_size))

    keypoints: List[cv2.KeyPoint] = []
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(kp_size)))

    sift = cv2.SIFT_create()
    _kps, des = sift.compute(gray, keypoints)
    if des is None:
        return np.zeros((0, 128), dtype=np.float32)
    return des.astype(np.float32)


def bovw_encode_hist(des: np.ndarray, centers: np.ndarray) -> np.ndarray:
    k = int(centers.shape[0])
    if des.shape[0] == 0:
        return np.zeros((k,), dtype=np.float32)

    d2 = ((des[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nn = np.argmin(d2, axis=1)
    hist = np.bincount(nn, minlength=k).astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def build_bovw_codebook(
    ds_train,
    out_path: Path,
    seed: int,
    bovw_type: str,
    vocab_size: int,
    sample_images: int,
    max_desc: int,
    orb_resize: int,
    orb_max_kp: int,
    orb_fast_threshold: int,
    dsift_resize: int,
    dsift_step: int,
    dsift_kp_size: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    n_total = len(ds_train)
    n_sample = min(int(sample_images), n_total)
    idxs = rng.choice(n_total, size=n_sample, replace=False)

    desc_list: List[np.ndarray] = []
    total = 0

    for idx in tqdm(idxs, desc=f"BoVW({bovw_type}): collect desc"):
        img_pil, _lab = ds_train[int(idx)]
        bgr = bgr_from_pil(img_pil)

        if bovw_type == "orb":
            des = orb_descriptors(
                bgr,
                resize=orb_resize,
                max_kp=orb_max_kp,
                fast_threshold=orb_fast_threshold,
            )
        elif bovw_type == "dsift":
            des = dsift_descriptors(
                bgr,
                resize=dsift_resize,
                step=dsift_step,
                kp_size=dsift_kp_size,
            )
        else:
            raise ValueError(f"Unknown bovw_type: {bovw_type}")

        if des.shape[0] == 0:
            continue

        if total + des.shape[0] > max_desc:
            remain = max_desc - total
            if remain <= 0:
                break
            take = min(remain, des.shape[0])
            sel = rng.choice(des.shape[0], size=take, replace=False)
            des = des[sel]

        desc_list.append(des)
        total += des.shape[0]
        if total >= max_desc:
            break

    if total == 0:
        raise RuntimeError(f"No descriptors found for BoVW({bovw_type}).")

    max_k = max(16, int(total // 5))
    k = min(int(vocab_size), max_k)
    if k < int(vocab_size):
        print(
            f"[WARN] BoVW({bovw_type}): descriptors={total} too small for requested k={vocab_size}. "
            f"Auto-shrinking k -> {k}."
        )

    D = np.concatenate(desc_list, axis=0).astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        batch_size=8192,
        n_init="auto",
        max_iter=200,
        verbose=0,
    )
    kmeans.fit(D)
    centers = kmeans.cluster_centers_.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, centers)

    meta = {
        "bovw_type": bovw_type,
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": int(k),
        "desc_dim": int(centers.shape[1]),
        "n_desc_used": int(D.shape[0]),
        "orb_resize": int(orb_resize),
        "orb_max_kp": int(orb_max_kp),
        "orb_fast_threshold": int(orb_fast_threshold),
        "dsift_resize": int(dsift_resize),
        "dsift_step": int(dsift_step),
        "dsift_kp_size": int(dsift_kp_size),
    }
    return meta


# -----------------------------
# Dataset feature extraction
# -----------------------------
_GLOBALS: Dict[str, Any] = {}


def _worker_init(base_cfg: FeatureExtractorConfig, extra_cfg: ExtraFeatureConfig, bovw_centers: Optional[np.ndarray]) -> None:
    _GLOBALS["base_cfg"] = base_cfg
    _GLOBALS["extra_cfg"] = extra_cfg
    _GLOBALS["bovw_centers"] = bovw_centers


def _extract_one(args: Tuple[int, Any]) -> Tuple[int, np.ndarray, int, int]:
    idx, sample = args
    img_pil, label = sample
    bgr = bgr_from_pil(img_pil)

    base_out = extract_handcrafted_features(bgr, _GLOBALS["base_cfg"])
    x_base = base_out["x"].astype(np.float32)
    roi_ok = int(bool(base_out.get("roi_success", True)))

    extra_cfg: ExtraFeatureConfig = _GLOBALS["extra_cfg"]
    extras: List[np.ndarray] = []

    if extra_cfg.use_lbp:
        extras.append(lbp_histogram(bgr, resize=extra_cfg.lbp_resize, bins=extra_cfg.lbp_bins))

    if extra_cfg.use_color_hist:
        extras.append(
            color_histogram(
                bgr,
                resize=extra_cfg.color_resize,
                bins=extra_cfg.color_bins,
                color_space=extra_cfg.color_space,
                norm=extra_cfg.color_hist_norm,
            )
        )

    if extra_cfg.use_bovw:
        centers = _GLOBALS["bovw_centers"]
        if centers is None:
            raise RuntimeError("BoVW enabled but centers not provided.")
        if extra_cfg.bovw_type == "orb":
            des = orb_descriptors(
                bgr,
                resize=extra_cfg.color_resize,
                max_kp=extra_cfg.bovw_orb_max_kp,
                fast_threshold=extra_cfg.bovw_orb_fast_threshold,
            )
        elif extra_cfg.bovw_type == "dsift":
            des = dsift_descriptors(
                bgr,
                resize=extra_cfg.dsift_resize,
                step=extra_cfg.dsift_step,
                kp_size=extra_cfg.dsift_kp_size,
            )
        else:
            raise ValueError(f"Unknown bovw_type: {extra_cfg.bovw_type}")

        extras.append(bovw_encode_hist(des, centers))

    x = np.concatenate([x_base] + extras, axis=0).astype(np.float32) if extras else x_base
    return idx, x, int(label), roi_ok


def extract_features_dataset(
    ds,
    base_cfg: FeatureExtractorConfig,
    extra_cfg: ExtraFeatureConfig,
    bovw_centers: Optional[np.ndarray],
    max_samples: Optional[int],
    feat_workers: int,
    desc: str,
    total_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    d = int(total_dim)

    X = np.zeros((n, d), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    roi_ok = np.zeros((n,), dtype=np.uint8)

    items = [(i, ds[i]) for i in range(n)]

    roi_success = 0
    if feat_workers <= 1:
        _worker_init(base_cfg, extra_cfg, bovw_centers)
        for i, sample in tqdm(items, desc=desc):
            idx, x, lab, ok = _extract_one((i, sample))
            X[idx] = x
            y[idx] = lab
            roi_ok[idx] = ok
            roi_success += ok
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=feat_workers, initializer=_worker_init, initargs=(base_cfg, extra_cfg, bovw_centers)) as pool:
            for idx, x, lab, ok in tqdm(
                pool.imap_unordered(_extract_one, items, chunksize=32),
                total=n,
                desc=desc,
            ):
                X[idx] = x
                y[idx] = lab
                roi_ok[idx] = ok
                roi_success += ok

    stats = {"n": float(n), "roi_success_rate": float(roi_success / max(n, 1))}
    return X, y, roi_ok, stats


def save_npz(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    roi_ok: np.ndarray,
    stats: Dict[str, float],
    cache_dtype: str,
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if cache_dtype == "float16":
        X_store = X.astype(np.float16)
    elif cache_dtype == "float32":
        X_store = X.astype(np.float32)
    else:
        raise ValueError(f"Unknown cache_dtype: {cache_dtype}")

    np.savez_compressed(
        path,
        X=X_store,
        y=y.astype(np.int64),
        roi_ok=roi_ok.astype(np.uint8),
        stats=np.array(stats, dtype=object),
        meta=np.array(meta, dtype=object),
        feature_dim=np.array([X.shape[1]], dtype=np.int64),
        dtype=np.array([cache_dtype], dtype=object),
    )


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    roi_ok = data["roi_ok"].astype(np.uint8) if "roi_ok" in data else np.ones((len(y),), dtype=np.uint8)
    stats = dict(data["stats"].item()) if "stats" in data else {}
    meta = dict(data["meta"].item()) if "meta" in data else {}
    return X, y, roi_ok, stats, meta


def build_cache_key(
    split: str,
    hog_resize: int,
    hough_resize: int,
    use_roi: bool,
    use_sp_hog: bool,
    extra_cfg: ExtraFeatureConfig,
    actual_bovw_k: Optional[int],
    max_samples_arg: int,
    cache_dtype: str,
) -> str:
    toks: List[str] = []
    toks.append(str(split))
    toks.append(f"hog{int(hog_resize)}")
    toks.append(f"hough{int(hough_resize)}")

    if use_roi:
        toks.append("roi")
    if use_sp_hog:
        toks.append("sphog")

    if extra_cfg.use_lbp:
        toks.append("lbp")
    if extra_cfg.use_color_hist:
        toks.append("ch")

    if extra_cfg.use_bovw:
        toks.append(f"bovw{extra_cfg.bovw_type}")
        k = int(actual_bovw_k) if actual_bovw_k is not None else int(extra_cfg.bovw_vocab_size)
        toks.append(f"k{k}")

    if int(max_samples_arg) != 0:
        toks.append(f"max{int(max_samples_arg)}")

    toks.append(f"dtype{str(cache_dtype)}")
    return "_".join(toks)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract handcrafted features to disk (.npz)")

    p.add_argument("--data_dir", type=str, default=str(cfg.DATA_PATH / "data_raw"))
    p.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "features"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--split", type=str, choices=["train", "test"], required=True)
    p.add_argument("--max_samples", type=int, default=0)

    # base extractor knobs
    p.add_argument("--use_roi", action="store_true", help="If set, base extractor crops ROI before features")
    p.add_argument("--use_sp_hog", action="store_true", help="If set, base extractor adds spatial pyramid HOG")

    p.add_argument("--hog_resize", type=int, default=64)
    p.add_argument("--hough_resize", type=int, default=96)

    # extras: default ON, flags disable them
    p.add_argument("--no_lbp", action="store_true", help="Disable LBP (default: enabled)")
    p.add_argument("--lbp_resize", type=int, default=64)
    p.add_argument("--lbp_bins", type=int, default=256)

    p.add_argument("--no_color_hist", action="store_true", help="Disable color histogram (default: enabled)")
    p.add_argument("--color_resize", type=int, default=64)
    p.add_argument("--color_bins", type=int, default=16)
    p.add_argument("--color_space", type=str, choices=["hsv", "rgb"], default="hsv")
    p.add_argument("--color_hist_norm", type=str, choices=["l1", "none"], default="l1")

    # BoVW: default ON, dsift, k=128
    p.add_argument("--no_bovw", action="store_true", help="Disable BoVW (default: enabled)")
    p.add_argument("--bovw_type", type=str, choices=["dsift", "orb"], default="dsift")
    p.add_argument("--bovw_vocab_size", type=int, default=128)
    p.add_argument("--bovw_sample_images", type=int, default=16000)
    p.add_argument("--bovw_max_desc", type=int, default=800000)

    # ORB BoVW knobs
    p.add_argument("--bovw_orb_max_kp", type=int, default=1500)
    p.add_argument("--bovw_orb_fast_threshold", type=int, default=10)

    # Dense-SIFT knobs
    p.add_argument("--dsift_resize", type=int, default=64)
    p.add_argument("--dsift_step", type=int, default=6)
    p.add_argument("--dsift_kp_size", type=int, default=8)

    # perf/cache
    p.add_argument("--feat_workers", type=int, default=1)
    p.add_argument("--cache_dtype", type=str, choices=["float16", "float32"], default="float16")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = FeatureExtractorConfig(
        use_roi=bool(args.use_roi),
        use_sp_hog=bool(args.use_sp_hog),
    )
    base_cfg = FeatureExtractorConfig(
        roi_cfg=base_cfg.roi_cfg,
        use_roi=base_cfg.use_roi,
        hsv_cfg=base_cfg.hsv_cfg,
        hog_cfg=base_cfg.hog_cfg.__class__(
            resize=int(args.hog_resize),
            orientations=base_cfg.hog_cfg.orientations,
            pixels_per_cell=base_cfg.hog_cfg.pixels_per_cell,
            cells_per_block=base_cfg.hog_cfg.cells_per_block,
        ),
        hough_cfg=base_cfg.hough_cfg.__class__(
            resize=int(args.hough_resize),
            canny1=base_cfg.hough_cfg.canny1,
            canny2=base_cfg.hough_cfg.canny2,
            dp=base_cfg.hough_cfg.dp,
            min_dist=base_cfg.hough_cfg.min_dist,
            param1=base_cfg.hough_cfg.param1,
            param2=base_cfg.hough_cfg.param2,
            min_radius=base_cfg.hough_cfg.min_radius,
            max_radius=base_cfg.hough_cfg.max_radius,
            hough_threshold=base_cfg.hough_cfg.hough_threshold,
            min_line_length=base_cfg.hough_cfg.min_line_length,
            max_line_gap=base_cfg.hough_cfg.max_line_gap,
        ),
        use_sp_hog=bool(args.use_sp_hog),
        sp_hog_cfg=base_cfg.sp_hog_cfg,
        return_blocks=False,
    )

    max_samples = None if args.max_samples == 0 else int(args.max_samples)
    ds = load_gtsrb_split(args.data_dir, args.split)

    extra_cfg = ExtraFeatureConfig(
        use_lbp=not bool(args.no_lbp),
        lbp_resize=int(args.lbp_resize),
        lbp_bins=int(args.lbp_bins),
        use_color_hist=not bool(args.no_color_hist),
        color_resize=int(args.color_resize),
        color_bins=int(args.color_bins),
        color_space=str(args.color_space),
        color_hist_norm=str(args.color_hist_norm),
        use_bovw=not bool(args.no_bovw),
        bovw_type=str(args.bovw_type),
        bovw_vocab_size=int(args.bovw_vocab_size),
        bovw_sample_images=int(args.bovw_sample_images),
        bovw_max_desc=int(args.bovw_max_desc),
        bovw_orb_max_kp=int(args.bovw_orb_max_kp),
        bovw_orb_fast_threshold=int(args.bovw_orb_fast_threshold),
        dsift_step=int(args.dsift_step),
        dsift_kp_size=int(args.dsift_kp_size),
        dsift_resize=int(args.dsift_resize),
    )

    bovw_centers: Optional[np.ndarray] = None
    bovw_meta: Dict[str, Any] = {}
    actual_bovw_k: Optional[int] = None

    if extra_cfg.use_bovw:
        bovw_dir = out_dir / "bovw"
        centers_path = bovw_dir / (
            f"{extra_cfg.bovw_type}_centers_k{extra_cfg.bovw_vocab_size}"
            f"_r{extra_cfg.dsift_resize if extra_cfg.bovw_type=='dsift' else extra_cfg.color_resize}.npy"
        )
        centers_meta_path = centers_path.with_suffix(".meta.json")

        if args.split == "train":
            if (not centers_path.exists()) or args.overwrite:
                ds_train_full = load_gtsrb_split(args.data_dir, "train")
                bovw_meta = build_bovw_codebook(
                    ds_train=ds_train_full,
                    out_path=centers_path,
                    seed=args.seed,
                    bovw_type=extra_cfg.bovw_type,
                    vocab_size=extra_cfg.bovw_vocab_size,
                    sample_images=extra_cfg.bovw_sample_images,
                    max_desc=extra_cfg.bovw_max_desc,
                    orb_resize=extra_cfg.color_resize,
                    orb_max_kp=extra_cfg.bovw_orb_max_kp,
                    orb_fast_threshold=extra_cfg.bovw_orb_fast_threshold,
                    dsift_resize=extra_cfg.dsift_resize,
                    dsift_step=extra_cfg.dsift_step,
                    dsift_kp_size=extra_cfg.dsift_kp_size,
                )
                centers_meta_path.write_text(json.dumps(bovw_meta, indent=2))
                print(f"Saved BoVW centers: {centers_path}")
            else:
                bovw_meta = json.loads(centers_meta_path.read_text()) if centers_meta_path.exists() else {}
        else:
            if not centers_path.exists():
                raise RuntimeError(f"BoVW centers missing: {centers_path}. Run train extraction first.")
            bovw_meta = json.loads(centers_meta_path.read_text()) if centers_meta_path.exists() else {}

        bovw_centers = np.load(centers_path).astype(np.float32)
        actual_bovw_k = int(bovw_centers.shape[0])

    base_d = int(base_feature_dim(base_cfg))
    total_d = base_d + extra_feature_dim(extra_cfg, actual_bovw_k=actual_bovw_k)

    key = build_cache_key(
        split=args.split,
        hog_resize=int(args.hog_resize),
        hough_resize=int(args.hough_resize),
        use_roi=bool(args.use_roi),
        use_sp_hog=bool(args.use_sp_hog),
        extra_cfg=extra_cfg,
        actual_bovw_k=actual_bovw_k,
        max_samples_arg=int(args.max_samples),
        cache_dtype=str(args.cache_dtype),
    )
    out_npz = out_dir / f"{key}.npz"
    out_meta = out_dir / f"{key}.meta.json"

    if out_npz.exists() and not args.overwrite:
        X, y, roi_ok, st, mt = load_npz(out_npz)
        print(f"Cache exists, not overwriting: {out_npz}")
        print(f"Loaded X={X.shape} y={y.shape} roi_ok_rate={roi_ok.mean():.3f} stats={st}")
        print(f"Meta: {mt}")
        return

    X, y, roi_ok, stats = extract_features_dataset(
        ds=ds,
        base_cfg=base_cfg,
        extra_cfg=extra_cfg,
        bovw_centers=bovw_centers,
        max_samples=max_samples,
        feat_workers=int(args.feat_workers),
        desc=f"Extract {args.split} feats",
        total_dim=total_d,
    )

    meta = {
        "split": args.split,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "out_npz": str(out_npz),
        "n": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "base_dim": int(base_d),
        "extra_dim": int(X.shape[1] - base_d),
        "stats": stats,
        "roi_ok_rate": float(roi_ok.mean()),
        "base_feature_config": {
            "use_roi": bool(args.use_roi),
            "use_sp_hog": bool(args.use_sp_hog),
            "hog_resize": int(args.hog_resize),
            "hough_resize": int(args.hough_resize),
        },
        "extra_feature_config": {
            "use_lbp": bool(extra_cfg.use_lbp),
            "use_color_hist": bool(extra_cfg.use_color_hist),
            "use_bovw": bool(extra_cfg.use_bovw),
            "bovw_type": extra_cfg.bovw_type if extra_cfg.use_bovw else "none",
            "bovw_vocab_size_requested": int(extra_cfg.bovw_vocab_size),
            "bovw_vocab_size_actual": int(actual_bovw_k) if actual_bovw_k is not None else 0,
            "dsift_resize": int(extra_cfg.dsift_resize),
            "dsift_step": int(extra_cfg.dsift_step),
            "dsift_kp_size": int(extra_cfg.dsift_kp_size),
        },
        "bovw_meta": bovw_meta,
        "speed_config": {"feat_workers": int(args.feat_workers), "cache_dtype": args.cache_dtype},
    }

    save_npz(out_npz, X, y, roi_ok, stats, cache_dtype=args.cache_dtype, meta=meta)
    out_meta.write_text(json.dumps(meta, indent=2))

    print("\n=== FEATURE EXTRACTION DONE ===")
    print(json.dumps(meta, indent=2))
    print("==============================\n")


if __name__ == "__main__":
    main()