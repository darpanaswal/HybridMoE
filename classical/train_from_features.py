import json
import time
import joblib
import argparse
import numpy as np
from pathlib import Path
from utils import config as cfg
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Tuple, List, Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit

def log(msg: str) -> None:
    print(msg, flush=True)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    stats = dict(data["stats"].item()) if "stats" in data else {}
    meta = dict(data["meta"].item()) if "meta" in data else {}
    return X, y, stats, meta


def eval_acc(clf: Any, X: np.ndarray, y: np.ndarray) -> float:
    pred = clf.predict(X)
    return float(accuracy_score(y, pred))


def build_preprocessor(pca_dim: int) -> List[Tuple[str, Any]]:
    steps: List[Tuple[str, Any]] = [("scaler", StandardScaler(with_mean=True, with_std=True))]
    if int(pca_dim) > 0:
        steps.append(("pca", PCA(n_components=int(pca_dim), random_state=0)))
    return steps


def fit_sgd_linear(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
    loss: str,
    epochs: int,
    alpha: float,
    tol: float | None,
    early_stopping: bool,
    calibrate: str,
    pca_dim: int,
    n_jobs: int,
) -> Any:
    if loss not in {"hinge", "log_loss"}:
        raise ValueError(f"Unsupported loss: {loss}")

    sgd_kwargs: Dict[str, Any] = {
        "loss": loss,
        "alpha": float(alpha),
        "max_iter": int(epochs),
        "tol": tol,
        "n_jobs": int(n_jobs),
        "random_state": int(seed),
        "class_weight": "balanced",
        "early_stopping": bool(early_stopping),
    }
    if early_stopping:
        sgd_kwargs["validation_fraction"] = 0.1

    sgd = SGDClassifier(**sgd_kwargs)

    steps = build_preprocessor(pca_dim)
    steps.append(("clf", sgd))
    base = Pipeline(steps)
    base.fit(X_tr, y_tr)

    if calibrate == "prefit":
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        cal.fit(X_cal, y_cal)
        return cal

    return base


def build_feature_key(
    split: str,
    hog_resize: int,
    hough_resize: int,
    use_roi: bool,
    use_sp_hog: bool,
    use_lbp: bool,
    use_color_hist: bool,
    use_bovw: bool,
    bovw_type: str,
    bovw_k: int,
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

    if use_lbp:
        toks.append("lbp")
    if use_color_hist:
        toks.append("ch")

    if use_bovw:
        toks.append(f"bovw{str(bovw_type)}")
        toks.append(f"k{int(bovw_k)}")

    toks.append(f"dtype{str(cache_dtype)}")
    return "_".join(toks)


def build_feature_paths(
    features_dir: Path,
    hog_resize: int,
    hough_resize: int,
    use_roi: bool,
    use_sp_hog: bool,
    use_lbp: bool,
    use_color_hist: bool,
    use_bovw: bool,
    bovw_type: str,
    bovw_k: int,
    cache_dtype: str,
) -> Tuple[Path, Path]:
    train_key = build_feature_key(
        split="train",
        hog_resize=hog_resize,
        hough_resize=hough_resize,
        use_roi=use_roi,
        use_sp_hog=use_sp_hog,
        use_lbp=use_lbp,
        use_color_hist=use_color_hist,
        use_bovw=use_bovw,
        bovw_type=bovw_type,
        bovw_k=bovw_k,
        cache_dtype=cache_dtype,
    )
    test_key = build_feature_key(
        split="test",
        hog_resize=hog_resize,
        hough_resize=hough_resize,
        use_roi=use_roi,
        use_sp_hog=use_sp_hog,
        use_lbp=use_lbp,
        use_color_hist=use_color_hist,
        use_bovw=use_bovw,
        bovw_type=bovw_type,
        bovw_k=bovw_k,
        cache_dtype=cache_dtype,
    )
    return features_dir / f"{train_key}.npz", features_dir / f"{test_key}.npz"


def build_out_dir_name(
    pca_dim: int,
    hog_resize: int,
    hough_resize: int,
    use_roi: bool,
    use_sp_hog: bool,
    use_lbp: bool,
    use_color_hist: bool,
    use_bovw: bool,
    bovw_type: str,
    bovw_k: int,
    cache_dtype: str,
) -> str:
    toks: List[str] = []
    toks.append(f"pca{int(pca_dim)}")
    toks.append(f"hog{int(hog_resize)}")
    toks.append(f"hough{int(hough_resize)}")

    if use_roi:
        toks.append("roi")
    if use_sp_hog:
        toks.append("sphog")

    if use_lbp:
        toks.append("lbp")
    if use_color_hist:
        toks.append("ch")

    if use_bovw:
        toks.append(f"bovw{str(bovw_type)}")
        toks.append(f"k{int(bovw_k)}")

    toks.append(f"dtype{str(cache_dtype)}")
    return "_".join(toks)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Train SGD model from pre-extracted features (.npz)")

    p.add_argument("--features_dir", type=str, default=str(cfg.OUTPUT_DIR / "features"))
    p.add_argument("--out_root", type=str, default=str(cfg.OUTPUT_DIR / "classical"))

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--pca_dim", type=int, default=0, help="0 disables PCA")

    # Must match feature-extraction naming/flags
    p.add_argument("--hog_resize", type=int, default=64)
    p.add_argument("--hough_resize", type=int, default=96)
    p.add_argument("--use_roi", action="store_true")
    p.add_argument("--use_sp_hog", action="store_true")

    # extras default ON, flags disable them (matches previous script)
    p.add_argument("--no_lbp", action="store_true")
    p.add_argument("--no_color_hist", action="store_true")
    p.add_argument("--no_bovw", action="store_true")

    p.add_argument("--bovw_type", type=str, choices=["dsift", "orb"], default="dsift")
    p.add_argument("--bovw_vocab_size", type=int, default=128)

    p.add_argument("--cache_dtype", type=str, choices=["float16", "float32"], default="float16")

    # SGD linear
    p.add_argument("--sgd_loss", type=str, choices=["hinge", "log_loss"], default="log_loss")
    p.add_argument("--sgd_epochs", type=int, default=10000)
    p.add_argument("--sgd_alpha", type=float, default=1e-5)
    p.add_argument("--sgd_tol", type=float, default=1e-4, help="Use -1 to disable tol-based stop")
    p.add_argument("--sgd_early_stopping", action="store_false")
    p.add_argument("--sgd_calibrate", type=str, choices=["prefit", "none"], default="none")
    p.add_argument("--sgd_jobs", type=int, default=8)

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    features_dir = Path(args.features_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    use_lbp = not bool(args.no_lbp)
    use_color_hist = not bool(args.no_color_hist)
    use_bovw = not bool(args.no_bovw)

    train_npz, test_npz = build_feature_paths(
        features_dir=features_dir,
        hog_resize=int(args.hog_resize),
        hough_resize=int(args.hough_resize),
        use_roi=bool(args.use_roi),
        use_sp_hog=bool(args.use_sp_hog),
        use_lbp=bool(use_lbp),
        use_color_hist=bool(use_color_hist),
        use_bovw=bool(use_bovw),
        bovw_type=str(args.bovw_type),
        bovw_k=int(args.bovw_vocab_size),
        cache_dtype=str(args.cache_dtype),
    )

    out_dir_name = build_out_dir_name(
        pca_dim=int(args.pca_dim),
        hog_resize=int(args.hog_resize),
        hough_resize=int(args.hough_resize),
        use_roi=bool(args.use_roi),
        use_sp_hog=bool(args.use_sp_hog),
        use_lbp=bool(use_lbp),
        use_color_hist=bool(use_color_hist),
        use_bovw=bool(use_bovw),
        bovw_type=str(args.bovw_type),
        bovw_k=int(args.bovw_vocab_size),
        cache_dtype=str(args.cache_dtype),
    )
    out_dir = out_root / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    log("[1/5] Loading features...")
    X_tr_full, y_tr_full, tr_stats, tr_meta = load_npz(str(train_npz))
    X_te, y_te, te_stats, te_meta = load_npz(str(test_npz))

    raw_dim = int(X_tr_full.shape[1])
    used_dim = int(args.pca_dim) if int(args.pca_dim) > 0 else raw_dim

    log(f"  Train: X={X_tr_full.shape} y={y_tr_full.shape} ({train_npz})")
    log(f"  Test : X={X_te.shape} y={y_te.shape} ({test_npz})")
    log(f"  Dimensionality: raw={raw_dim} -> used={used_dim}")

    log("[2/5] Creating train/val split...")
    val_frac = max(0.01, min(0.5, float(args.val_frac)))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=args.seed)
    tr_idx, va_idx = next(splitter.split(X_tr_full, y_tr_full))
    X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
    X_va, y_va = X_tr_full[va_idx], y_tr_full[va_idx]
    log(f"  Train split: {X_tr.shape}, Val split: {X_va.shape}")

    tol = None if float(args.sgd_tol) < 0 else float(args.sgd_tol)

    log(
        f"[3/5] Training SGD (loss={args.sgd_loss}, pca_dim={args.pca_dim}, "
        f"epochs={args.sgd_epochs}, alpha={args.sgd_alpha}) ..."
    )
    t_lin0 = time.perf_counter()
    model = fit_sgd_linear(
        X_tr=X_tr,
        y_tr=y_tr,
        X_cal=X_va,
        y_cal=y_va,
        seed=int(args.seed),
        loss=str(args.sgd_loss),
        epochs=int(args.sgd_epochs),
        alpha=float(args.sgd_alpha),
        tol=tol,
        early_stopping=bool(args.sgd_early_stopping),
        calibrate=str(args.sgd_calibrate),
        pca_dim=int(args.pca_dim),
        n_jobs=int(args.sgd_jobs),
    )
    log(f"  SGD done in {time.perf_counter() - t_lin0:.1f}s")

    log("[4/5] Evaluating...")
    results: Dict[str, Any] = {
        "sgd": {
            "train_acc": eval_acc(model, X_tr, y_tr),
            "val_acc": eval_acc(model, X_va, y_va),
            "test_acc": eval_acc(model, X_te, y_te),
        }
    }

    log("[5/5] Saving...")
    joblib.dump(model, out_dir / "sgd.joblib")

    # Feature meta should reflect the *chosen flags* and the resolved paths, not arbitrary long defaults.
    feature_meta: Dict[str, Any] = {
        "resolved": {
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
        },
        "flags": {
            "hog_resize": int(args.hog_resize),
            "hough_resize": int(args.hough_resize),
            "use_roi": bool(args.use_roi),
            "use_sp_hog": bool(args.use_sp_hog),
            "use_lbp": bool(use_lbp),
            "use_color_hist": bool(use_color_hist),
            "use_bovw": bool(use_bovw),
            "bovw_type": str(args.bovw_type) if use_bovw else "none",
            "bovw_vocab_size": int(args.bovw_vocab_size) if use_bovw else 0,
            "cache_dtype": str(args.cache_dtype),
        },
        "npz_meta": {
            "train": tr_meta,
            "test": te_meta,
        },
        "npz_stats": {
            "train": tr_stats,
            "test": te_stats,
        },
    }

    metrics: Dict[str, Any] = {
        "seed": int(args.seed),
        "dims": {"raw": raw_dim, "used": used_dim, "pca_dim_flag": int(args.pca_dim)},
        "inputs": {"val_frac": float(val_frac)},
        "config": {
            "sgd_loss": str(args.sgd_loss),
            "sgd_epochs": int(args.sgd_epochs),
            "sgd_alpha": float(args.sgd_alpha),
            "sgd_tol": float(args.sgd_tol),
            "sgd_early_stopping": bool(args.sgd_early_stopping),
            "sgd_calibrate": str(args.sgd_calibrate),
            "sgd_jobs": int(args.sgd_jobs),
        },
        "feature_meta": feature_meta,
        "results": results,
        "timing_sec": {"total": float(time.perf_counter() - t0)},
        "out_dir": str(out_dir),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log("\n=== CLASSICAL TRAIN DONE ===")
    log(json.dumps(metrics, indent=2))
    log(f"Saved to: {out_dir}")
    log("============================\n")


if __name__ == "__main__":
    main()