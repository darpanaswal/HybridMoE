# attacks/eval_classical_transfer_attacks.py
import csv
import json
import joblib
import torch
import argparse
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
from torchvision.datasets import GTSRB
from torchvision import transforms as T
from torchvision.utils import save_image
from torchmetrics.classification import MulticlassAccuracy

from utils import config as cfg
from attacks.fgsm import FGSMConfig, fgsm_attack
from attacks.pgd import PGDConfig, pgd_linf_attack
from models.resnet50_pl import ResNet50Classifier
from handcrafted.features.feature_extractor import FeatureExtractorConfig, extract_handcrafted_features


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------
# Utils
# -----------------------------
def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def parse_list_of_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_list_of_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_attack_set(s: str) -> List[str]:
    toks = [t.strip().lower() for t in str(s).split(",") if t.strip()]
    ok = {"clean", "fgsm", "pgd"}
    for t in toks:
        if t not in ok:
            raise ValueError(f"Unknown attack in --attack_set: {t}. Allowed: {sorted(ok)}")
    if not toks:
        return ["clean", "fgsm", "pgd"]
    return toks


def resolve_ckpt_path(output_dir: Path, run_name: str, ckpt_epoch: int) -> Path:
    ckpt_root = output_dir / "checkpoints" / run_name
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    primary = ckpt_root / f"epochepoch={ckpt_epoch}-valaccval"
    if primary.exists() and primary.is_dir():
        cands = sorted(primary.glob("*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"No .ckpt found in folder: {primary}")
        return cands[0]

    marker = f"epochepoch={ckpt_epoch}"
    hits = sorted([p for p in ckpt_root.rglob("*.ckpt") if marker in p.parent.name])
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Could not find ckpt for epoch={ckpt_epoch} under {ckpt_root}")


def log(msg: str) -> None:
    print(msg, flush=True)


def attack_stats(x: torch.Tensor, x_adv: torch.Tensor) -> Dict[str, float]:
    delta = (x_adv - x).detach()
    linf = delta.abs().flatten(1).max(dim=1).values
    l2 = torch.sqrt((delta ** 2).flatten(1).sum(dim=1) + 1e-12)
    return {
        "linf_mean": float(linf.mean().item()),
        "linf_max": float(linf.max().item()),
        "l2_mean": float(l2.mean().item()),
    }


# -----------------------------
# Feature extraction (same spirit as your simplified extract_features.py)
# -----------------------------
def tensor_rgb01_to_bgr_u8(img_rgb01: torch.Tensor) -> np.ndarray:
    """
    img_rgb01: torch Tensor (3,H,W), range [0,1]
    returns: uint8 BGR (H,W,3)
    """
    x = (img_rgb01.detach().clamp(0, 1) * 255.0).to(torch.uint8)
    hwc = x.permute(1, 2, 0).cpu().numpy()  # RGB u8
    bgr = hwc[:, :, ::-1].copy()  # BGR
    return bgr


def lbp_histogram(bgr: np.ndarray, resize: int = 64, bins: int = 256) -> np.ndarray:
    if bins != 256:
        raise ValueError("This LBP implementation supports bins=256 only.")
    img = cv2.resize(bgr, (int(resize), int(resize)), interpolation=cv2.INTER_AREA)
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


def color_histogram(
    bgr: np.ndarray,
    resize: int = 64,
    bins: int = 16,
    color_space: str = "hsv",
    norm: str = "l1",
) -> np.ndarray:
    img = cv2.resize(bgr, (int(resize), int(resize)), interpolation=cv2.INTER_AREA)

    cs = str(color_space).lower()
    if cs == "hsv":
        x = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rng = (0, 256)
    elif cs == "rgb":
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rng = (0, 256)
    else:
        raise ValueError(f"Unknown color_space: {color_space}")

    feats: List[np.ndarray] = []
    for c in range(3):
        h = cv2.calcHist([x], [c], None, [int(bins)], [rng[0], rng[1]]).reshape(-1).astype(np.float32)
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


def orb_descriptors(bgr: np.ndarray, resize: int = 64, max_kp: int = 1500, fast_threshold: int = 10) -> np.ndarray:
    img = cv2.resize(bgr, (int(resize), int(resize)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=int(max_kp), fastThreshold=int(fast_threshold))
    _kps, des = orb.detectAndCompute(gray, None)
    if des is None:
        return np.zeros((0, 32), dtype=np.float32)
    return des.astype(np.float32)


def dsift_descriptors(bgr: np.ndarray, resize: int = 64, step: int = 6, kp_size: int = 8) -> np.ndarray:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT not available (cv2.SIFT_create missing). Install opencv-contrib-python.")

    img = cv2.resize(bgr, (int(resize), int(resize)), interpolation=cv2.INTER_AREA)
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


def make_base_cfg(args: argparse.Namespace) -> FeatureExtractorConfig:
    """
    Your FeatureExtractorConfig has changed over time. This constructor is defensive:
    it tries to pass use_roi/use_sp_hog if your version supports it.
    """
    kwargs: Dict[str, Any] = {}
    for k in ["use_roi", "use_sp_hog"]:
        if hasattr(args, k):
            kwargs[k] = bool(getattr(args, k))

    try:
        base_cfg = FeatureExtractorConfig(**kwargs)  # type: ignore[arg-type]
    except TypeError:
        base_cfg = FeatureExtractorConfig()

    # Set resize knobs if your config uses HOG/Hough sub-configs like before
    # (safe even if your dataclasses differ, because we only re-create if attrs exist)
    try:
        hog_cfg = base_cfg.hog_cfg
        base_cfg = FeatureExtractorConfig(  # type: ignore[call-arg]
            roi_cfg=getattr(base_cfg, "roi_cfg", None) or FeatureExtractorConfig().roi_cfg,
            hsv_cfg=getattr(base_cfg, "hsv_cfg", None) or FeatureExtractorConfig().hsv_cfg,
            hog_cfg=hog_cfg.__class__(  # type: ignore[call-arg]
                resize=int(args.hog_resize),
                orientations=hog_cfg.orientations,
                pixels_per_cell=hog_cfg.pixels_per_cell,
                cells_per_block=hog_cfg.cells_per_block,
            ),
            hough_cfg=base_cfg.hough_cfg.__class__(  # type: ignore[call-arg]
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
            return_blocks=False,
            **kwargs,  # if supported
        )
    except Exception:
        pass

    return base_cfg


def load_bovw_centers(args: argparse.Namespace) -> Optional[np.ndarray]:
    if args.no_bovw:
        return None

    if args.bovw_centers_path:
        p = Path(args.bovw_centers_path)
        if not p.exists():
            raise FileNotFoundError(f"--bovw_centers_path not found: {p}")
        return np.load(p).astype(np.float32)

    bovw_dir = Path(args.features_dir) / "bovw"
    resize = int(args.dsift_resize) if args.bovw_type == "dsift" else int(args.color_resize)
    centers_path = bovw_dir / f"{args.bovw_type}_centers_k{int(args.bovw_vocab_size)}_r{resize}.npy"
    if not centers_path.exists():
        raise FileNotFoundError(
            f"BoVW centers missing: {centers_path}\n"
            f"Run feature extraction on TRAIN split first with BoVW enabled (same type/k/resize)."
        )
    return np.load(centers_path).astype(np.float32)


def extract_features_for_batch(
    x_rgb01: torch.Tensor,
    base_cfg: FeatureExtractorConfig,
    use_lbp: bool,
    use_color_hist: bool,
    use_bovw: bool,
    bovw_type: str,
    centers: Optional[np.ndarray],
    lbp_resize: int,
    lbp_bins: int,
    color_resize: int,
    color_bins: int,
    color_space: str,
    color_hist_norm: str,
    orb_max_kp: int,
    orb_fast_threshold: int,
    dsift_resize: int,
    dsift_step: int,
    dsift_kp_size: int,
) -> np.ndarray:
    """
    x_rgb01: (B,3,H,W) on CPU or GPU; we will move per-sample to CPU for OpenCV.
    returns: (B,D) float32
    """
    B = int(x_rgb01.shape[0])
    feats: List[np.ndarray] = []

    # Move to CPU once to avoid device sync per-sample
    x_cpu = x_rgb01.detach().cpu()

    if use_bovw:
        if centers is None:
            raise RuntimeError("BoVW enabled but centers is None.")
        centers_np = centers
    else:
        centers_np = None

    for i in range(B):
        bgr = tensor_rgb01_to_bgr_u8(x_cpu[i])

        base_out = extract_handcrafted_features(bgr, base_cfg)
        x_base = np.asarray(base_out["x"], dtype=np.float32)

        extras: List[np.ndarray] = []
        if use_lbp:
            extras.append(lbp_histogram(bgr, resize=lbp_resize, bins=lbp_bins))
        if use_color_hist:
            extras.append(
                color_histogram(
                    bgr,
                    resize=color_resize,
                    bins=color_bins,
                    color_space=color_space,
                    norm=color_hist_norm,
                )
            )
        if use_bovw:
            if str(bovw_type) == "orb":
                des = orb_descriptors(
                    bgr,
                    resize=color_resize,
                    max_kp=orb_max_kp,
                    fast_threshold=orb_fast_threshold,
                )
            elif str(bovw_type) == "dsift":
                des = dsift_descriptors(
                    bgr,
                    resize=dsift_resize,
                    step=dsift_step,
                    kp_size=dsift_kp_size,
                )
            else:
                raise ValueError(f"Unknown bovw_type: {bovw_type}")
            extras.append(bovw_encode_hist(des, centers_np))  # type: ignore[arg-type]

        if extras:
            x = np.concatenate([x_base] + extras, axis=0).astype(np.float32)
        else:
            x = x_base.astype(np.float32)

        feats.append(x)

    X = np.stack(feats, axis=0).astype(np.float32)
    return X


# -----------------------------
# DataModule
# -----------------------------
class GTSRBTestDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, img_size: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.test_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        tfm = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),  # [0,1], RGB
        ])
        self.test_ds = GTSRB(root=self.data_dir, split="test", download=False, transform=tfm)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )


# -----------------------------
# LightningModule for eval
# -----------------------------
class ClassicalTransferEvalModule(pl.LightningModule):
    """
    Generate adversarial examples using CNN (ResNet50Classifier) gradients,
    then evaluate a classical sklearn model on handcrafted features extracted
    from those (possibly adversarial) images.
    """
    def __init__(
        self,
        cnn_ckpt_path: str,
        classical_joblib: str,
        mode: str,  # "clean" | "fgsm" | "pgd"
        num_classes: int,
        base_cfg: FeatureExtractorConfig,
        use_lbp: bool,
        use_color_hist: bool,
        use_bovw: bool,
        bovw_type: str,
        bovw_centers: Optional[np.ndarray],
        lbp_resize: int,
        lbp_bins: int,
        color_resize: int,
        color_bins: int,
        color_space: str,
        color_hist_norm: str,
        orb_max_kp: int,
        orb_fast_threshold: int,
        dsift_resize: int,
        dsift_step: int,
        dsift_kp_size: int,
        fgsm_cfg: Optional[FGSMConfig] = None,
        pgd_cfg: Optional[PGDConfig] = None,
        save_examples_dir: Optional[str] = None,
        save_n_batches: int = 0,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.fgsm_cfg = fgsm_cfg
        self.pgd_cfg = pgd_cfg

        # CNN surrogate
        self.cnn = ResNet50Classifier.load_from_checkpoint(cnn_ckpt_path, map_location="cpu")
        self.cnn.eval()

        # Classical model
        self.classical = joblib.load(classical_joblib)

        self.num_classes = int(num_classes)
        self.acc = MulticlassAccuracy(num_classes=self.num_classes)

        # Feature config
        self.base_cfg = base_cfg
        self.use_lbp = bool(use_lbp)
        self.use_color_hist = bool(use_color_hist)
        self.use_bovw = bool(use_bovw)
        self.bovw_type = str(bovw_type)
        self.bovw_centers = bovw_centers

        self.lbp_resize = int(lbp_resize)
        self.lbp_bins = int(lbp_bins)

        self.color_resize = int(color_resize)
        self.color_bins = int(color_bins)
        self.color_space = str(color_space)
        self.color_hist_norm = str(color_hist_norm)

        self.orb_max_kp = int(orb_max_kp)
        self.orb_fast_threshold = int(orb_fast_threshold)

        self.dsift_resize = int(dsift_resize)
        self.dsift_step = int(dsift_step)
        self.dsift_kp_size = int(dsift_kp_size)

        # Examples
        self.save_examples_dir = Path(save_examples_dir) if save_examples_dir else None
        self.save_n_batches = int(save_n_batches)

        # Manual stats aggregation
        self._stat_sum = {"linf_mean": 0.0, "linf_max": 0.0, "l2_mean": 0.0}
        self._stat_n = 0

    def on_test_start(self) -> None:
        # We only need gradients w.r.t. input x; freezing weights is fine.
        for p in self.cnn.parameters():
            p.requires_grad_(False)

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # Generate adversarial (or clean)
        if self.mode == "clean":
            x_use = x
        elif self.mode == "fgsm":
            assert self.fgsm_cfg is not None
            with torch.enable_grad():
                x_use = fgsm_attack(self.cnn, x, y, normalize_batch, self.fgsm_cfg)
            self._accumulate_stats(attack_stats(x, x_use))
            self._maybe_save_examples(x, x_use, batch_idx)
        elif self.mode == "pgd":
            assert self.pgd_cfg is not None
            with torch.enable_grad():
                x_use = pgd_linf_attack(self.cnn, x, y, normalize_batch, self.pgd_cfg)
            self._accumulate_stats(attack_stats(x, x_use))
            self._maybe_save_examples(x, x_use, batch_idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Extract handcrafted features on CPU
        X_feat = extract_features_for_batch(
            x_rgb01=x_use,
            base_cfg=self.base_cfg,
            use_lbp=self.use_lbp,
            use_color_hist=self.use_color_hist,
            use_bovw=self.use_bovw,
            bovw_type=self.bovw_type,
            centers=self.bovw_centers,
            lbp_resize=self.lbp_resize,
            lbp_bins=self.lbp_bins,
            color_resize=self.color_resize,
            color_bins=self.color_bins,
            color_space=self.color_space,
            color_hist_norm=self.color_hist_norm,
            orb_max_kp=self.orb_max_kp,
            orb_fast_threshold=self.orb_fast_threshold,
            dsift_resize=self.dsift_resize,
            dsift_step=self.dsift_step,
            dsift_kp_size=self.dsift_kp_size,
        )

        # Predict with sklearn model
        pred_np = self.classical.predict(X_feat)
        pred = torch.from_numpy(np.asarray(pred_np, dtype=np.int64)).to(self.device)
        self.acc.update(pred, y)

    def _accumulate_stats(self, st: Dict[str, float]) -> None:
        self._stat_sum["linf_mean"] += st["linf_mean"]
        self._stat_sum["linf_max"] = max(self._stat_sum["linf_max"], st["linf_max"])
        self._stat_sum["l2_mean"] += st["l2_mean"]
        self._stat_n += 1

    def _maybe_save_examples(self, x: torch.Tensor, x_adv: torch.Tensor, batch_idx: int) -> None:
        if self.save_examples_dir is None:
            return
        if not self.trainer.is_global_zero:
            return
        if batch_idx >= self.save_n_batches:
            return

        self.save_examples_dir.mkdir(parents=True, exist_ok=True)
        k = min(32, x.size(0))
        save_image(x[:k].detach().cpu(), self.save_examples_dir / f"clean_batch{batch_idx:03d}.png", nrow=8)
        save_image(x_adv[:k].detach().cpu(), self.save_examples_dir / f"adv_batch{batch_idx:03d}.png", nrow=8)

    def on_test_epoch_end(self) -> None:
        acc_val = self.acc.compute()
        self.log("test/acc", acc_val, prog_bar=True, sync_dist=True)

        if self._stat_n > 0:
            self.log("attack/linf_mean", self._stat_sum["linf_mean"] / self._stat_n, prog_bar=False, sync_dist=False)
            self.log("attack/linf_max", self._stat_sum["linf_max"], prog_bar=False, sync_dist=False)
            self.log("attack/l2_mean", self._stat_sum["l2_mean"] / self._stat_n, prog_bar=False, sync_dist=False)

        self.acc.reset()
        self._stat_sum = {"linf_mean": 0.0, "linf_max": 0.0, "l2_mean": 0.0}
        self._stat_n = 0

    def configure_optimizers(self):
        return None


def run_test(trainer: pl.Trainer, dm: pl.LightningDataModule, module: pl.LightningModule) -> Dict[str, float]:
    out = trainer.test(module, datamodule=dm, verbose=False)
    if not out:
        return {}
    return {k: float(v) for k, v in out[0].items()}


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Transfer attacks: craft on CNN, evaluate on classical SGD (handcrafted features)")

    # Data / CNN
    p.add_argument("--data_dir", type=str, default=str(cfg.DATA_PATH / "data_raw"))
    p.add_argument("--run_name", type=str, default="resnet50_baseline")
    p.add_argument("--ckpt_epoch", type=int, required=True)

    # Classical model
    p.add_argument("--classical_joblib", type=str, required=True, help="Path to sgd.joblib from classical training")
    p.add_argument("--num_classes", type=int, default=43)

    # Dataloader / device
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)

    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strategy", type=str, default="ddp")
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--matmul_precision", type=str, choices=["highest", "high", "medium"], default="high")

    # Which attacks to run
    p.add_argument("--attack_set", type=str, default="clean,fgsm,pgd")

    # eps sweeps
    p.add_argument("--fgsm_eps", type=str, default="0.0,0.002,0.004,0.008,0.016")
    p.add_argument("--pgd_eps", type=str, default="0.0,0.002,0.004,0.008,0.016")
    p.add_argument("--pgd_steps_list", type=str, default="10", help="Comma-separated, e.g. 5,10,20")
    p.add_argument("--pgd_alpha", type=float, default=0.002)
    p.add_argument("--pgd_random_start", action="store_true")

    # Feature flags (must match your cached features / classical training)
    p.add_argument("--features_dir", type=str, default=str(cfg.OUTPUT_DIR / "features"))
    p.add_argument("--hog_resize", type=int, default=64)
    p.add_argument("--hough_resize", type=int, default=96)
    p.add_argument("--use_roi", action="store_true")
    p.add_argument("--use_sp_hog", action="store_true")

    p.add_argument("--no_lbp", action="store_true")
    p.add_argument("--lbp_resize", type=int, default=64)
    p.add_argument("--lbp_bins", type=int, default=256)

    p.add_argument("--no_color_hist", action="store_true")
    p.add_argument("--color_resize", type=int, default=64)
    p.add_argument("--color_bins", type=int, default=16)
    p.add_argument("--color_space", type=str, choices=["hsv", "rgb"], default="hsv")
    p.add_argument("--color_hist_norm", type=str, choices=["l1", "none"], default="l1")

    p.add_argument("--no_bovw", action="store_true")
    p.add_argument("--bovw_type", type=str, choices=["dsift", "orb"], default="dsift")
    p.add_argument("--bovw_vocab_size", type=int, default=128)
    p.add_argument("--bovw_centers_path", type=str, default="", help="Optional explicit .npy centers path")

    p.add_argument("--bovw_orb_max_kp", type=int, default=1500)
    p.add_argument("--bovw_orb_fast_threshold", type=int, default=10)

    p.add_argument("--dsift_resize", type=int, default=64)
    p.add_argument("--dsift_step", type=int, default=6)
    p.add_argument("--dsift_kp_size", type=int, default=8)

    # examples
    p.add_argument("--save_examples", action="store_true")
    p.add_argument("--save_n_batches", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)

    ckpt_path = resolve_ckpt_path(Path(cfg.OUTPUT_DIR), args.run_name, int(args.ckpt_epoch))
    attack_set = parse_attack_set(args.attack_set)

    # Feature config
    base_cfg = make_base_cfg(args)
    bovw_centers = load_bovw_centers(args)

    use_lbp = not bool(args.no_lbp)
    use_color_hist = not bool(args.no_color_hist)
    use_bovw = not bool(args.no_bovw)

    dm = GTSRBTestDataModule(
        data_dir=args.data_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        img_size=int(args.img_size),
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=int(args.devices),
        strategy=args.strategy if int(args.devices) > 1 and args.accelerator == "gpu" else "auto",
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,  # needed for adversarial generation
    )

    out_root = Path(cfg.OUTPUT_DIR) / "attacks" / f"{args.run_name}_transfer_classical"
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_name": args.run_name,
        "ckpt_epoch": int(args.ckpt_epoch),
        "ckpt_path": str(ckpt_path),
        "classical_joblib": str(args.classical_joblib),
        "data_dir": args.data_dir,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "img_size": int(args.img_size),
        "devices": int(args.devices),
        "strategy": trainer.strategy.__class__.__name__,
        "precision": str(args.precision),
        "attack_set": attack_set,
        "feature_flags": {
            "hog_resize": int(args.hog_resize),
            "hough_resize": int(args.hough_resize),
            "use_roi": bool(args.use_roi),
            "use_sp_hog": bool(args.use_sp_hog),
            "use_lbp": bool(use_lbp),
            "use_color_hist": bool(use_color_hist),
            "use_bovw": bool(use_bovw),
            "bovw_type": str(args.bovw_type) if use_bovw else "none",
            "bovw_vocab_size": int(args.bovw_vocab_size) if use_bovw else 0,
            "dsift_resize": int(args.dsift_resize),
            "dsift_step": int(args.dsift_step),
            "dsift_kp_size": int(args.dsift_kp_size),
            "color_resize": int(args.color_resize),
            "color_bins": int(args.color_bins),
            "color_space": str(args.color_space),
            "lbp_resize": int(args.lbp_resize),
            "lbp_bins": int(args.lbp_bins),
        },
        "clean": {},
        "fgsm": [],
        "pgd": [],
    }

    if trainer.is_global_zero:
        log(f"\nResolved CNN checkpoint: {ckpt_path}")
        log(f"Classical model: {args.classical_joblib}\n")

    # CLEAN
    if "clean" in attack_set:
        if trainer.is_global_zero:
            log("Running CLEAN transfer eval (classical on clean images)...")
        clean_module = ClassicalTransferEvalModule(
            cnn_ckpt_path=str(ckpt_path),
            classical_joblib=str(args.classical_joblib),
            mode="clean",
            num_classes=int(args.num_classes),
            base_cfg=base_cfg,
            use_lbp=use_lbp,
            use_color_hist=use_color_hist,
            use_bovw=use_bovw,
            bovw_type=str(args.bovw_type),
            bovw_centers=bovw_centers,
            lbp_resize=int(args.lbp_resize),
            lbp_bins=int(args.lbp_bins),
            color_resize=int(args.color_resize),
            color_bins=int(args.color_bins),
            color_space=str(args.color_space),
            color_hist_norm=str(args.color_hist_norm),
            orb_max_kp=int(args.bovw_orb_max_kp),
            orb_fast_threshold=int(args.bovw_orb_fast_threshold),
            dsift_resize=int(args.dsift_resize),
            dsift_step=int(args.dsift_step),
            dsift_kp_size=int(args.dsift_kp_size),
        )
        results["clean"] = run_test(trainer, dm, clean_module)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # FGSM sweep
    if "fgsm" in attack_set:
        fgsm_eps_list = parse_list_of_floats(args.fgsm_eps)
        if trainer.is_global_zero:
            log("\nRunning FGSM transfer sweep...")
        for eps in tqdm(fgsm_eps_list, disable=not trainer.is_global_zero, desc="FGSM eps"):
            ex_dir = None
            if args.save_examples and eps > 0:
                ex_dir = str(out_root / "examples" / f"fgsm_eps{eps}")

            m = ClassicalTransferEvalModule(
                cnn_ckpt_path=str(ckpt_path),
                classical_joblib=str(args.classical_joblib),
                mode="fgsm",
                fgsm_cfg=FGSMConfig(eps=float(eps)),
                num_classes=int(args.num_classes),
                base_cfg=base_cfg,
                use_lbp=use_lbp,
                use_color_hist=use_color_hist,
                use_bovw=use_bovw,
                bovw_type=str(args.bovw_type),
                bovw_centers=bovw_centers,
                lbp_resize=int(args.lbp_resize),
                lbp_bins=int(args.lbp_bins),
                color_resize=int(args.color_resize),
                color_bins=int(args.color_bins),
                color_space=str(args.color_space),
                color_hist_norm=str(args.color_hist_norm),
                orb_max_kp=int(args.bovw_orb_max_kp),
                orb_fast_threshold=int(args.bovw_orb_fast_threshold),
                dsift_resize=int(args.dsift_resize),
                dsift_step=int(args.dsift_step),
                dsift_kp_size=int(args.dsift_kp_size),
                save_examples_dir=ex_dir,
                save_n_batches=int(args.save_n_batches) if ex_dir else 0,
            )
            metrics = run_test(trainer, dm, m)
            results["fgsm"].append({"eps": float(eps), **metrics})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # PGD sweep (eps x steps)
    if "pgd" in attack_set:
        pgd_eps_list = parse_list_of_floats(args.pgd_eps)
        pgd_steps_list = parse_list_of_ints(args.pgd_steps_list)

        if trainer.is_global_zero:
            log("\nRunning PGD transfer sweep...")
        for steps in tqdm(pgd_steps_list, disable=not trainer.is_global_zero, desc="PGD steps"):
            for eps in tqdm(
                pgd_eps_list,
                disable=not trainer.is_global_zero,
                desc=f"PGD eps (steps={steps})",
                leave=False,
            ):
                pgd_cfg = PGDConfig(
                    eps=float(eps),
                    alpha=float(args.pgd_alpha),
                    steps=int(steps),
                    random_start=bool(args.pgd_random_start),
                )

                ex_dir = None
                if args.save_examples and eps > 0:
                    ex_dir = str(out_root / "examples" / f"pgd_eps{eps}_steps{steps}_rs{int(args.pgd_random_start)}")

                m = ClassicalTransferEvalModule(
                    cnn_ckpt_path=str(ckpt_path),
                    classical_joblib=str(args.classical_joblib),
                    mode="pgd",
                    pgd_cfg=pgd_cfg,
                    num_classes=int(args.num_classes),
                    base_cfg=base_cfg,
                    use_lbp=use_lbp,
                    use_color_hist=use_color_hist,
                    use_bovw=use_bovw,
                    bovw_type=str(args.bovw_type),
                    bovw_centers=bovw_centers,
                    lbp_resize=int(args.lbp_resize),
                    lbp_bins=int(args.lbp_bins),
                    color_resize=int(args.color_resize),
                    color_bins=int(args.color_bins),
                    color_space=str(args.color_space),
                    color_hist_norm=str(args.color_hist_norm),
                    orb_max_kp=int(args.bovw_orb_max_kp),
                    orb_fast_threshold=int(args.bovw_orb_fast_threshold),
                    dsift_resize=int(args.dsift_resize),
                    dsift_step=int(args.dsift_step),
                    dsift_kp_size=int(args.dsift_kp_size),
                    save_examples_dir=ex_dir,
                    save_n_batches=int(args.save_n_batches) if ex_dir else 0,
                )
                metrics = run_test(trainer, dm, m)
                results["pgd"].append({
                    "eps": float(eps),
                    "steps": int(steps),
                    "alpha": float(args.pgd_alpha),
                    "random_start": bool(args.pgd_random_start),
                    **metrics,
                })

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # SAVE (rank0)
    json_path = out_root / f"transfer_classical_epoch{int(args.ckpt_epoch)}.json"
    csv_path = out_root / f"transfer_classical_epoch{int(args.ckpt_epoch)}.csv"

    if trainer.is_global_zero:
        json_path.write_text(json.dumps(results, indent=2))

        clean_acc = results.get("clean", {}).get("test/acc", "")

        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "attack", "eps", "steps", "alpha", "random_start",
                "test/acc", "attack/linf_mean", "attack/linf_max", "attack/l2_mean",
                "clean_test/acc", "ckpt_epoch", "ckpt_path", "classical_joblib",
            ])

            # FGSM
            for r in results.get("fgsm", []):
                w.writerow([
                    "fgsm",
                    r.get("eps", ""),
                    "",
                    "",
                    "",
                    r.get("test/acc", ""),
                    r.get("attack/linf_mean", ""),
                    r.get("attack/linf_max", ""),
                    r.get("attack/l2_mean", ""),
                    clean_acc,
                    int(args.ckpt_epoch),
                    str(ckpt_path),
                    str(args.classical_joblib),
                ])

            # PGD
            for r in results.get("pgd", []):
                w.writerow([
                    "pgd",
                    r.get("eps", ""),
                    r.get("steps", ""),
                    r.get("alpha", ""),
                    r.get("random_start", ""),
                    r.get("test/acc", ""),
                    r.get("attack/linf_mean", ""),
                    r.get("attack/linf_max", ""),
                    r.get("attack/l2_mean", ""),
                    clean_acc,
                    int(args.ckpt_epoch),
                    str(ckpt_path),
                    str(args.classical_joblib),
                ])

        log("\n===== TRANSFER ATTACK EVAL SUMMARY (rank0) =====")
        log(f"Saved JSON: {json_path}")
        log(f"Saved CSV : {csv_path}")
        log("===============================================\n")


if __name__ == "__main__":
    main()