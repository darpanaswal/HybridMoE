# attacks/eval_hybrid_attacks.py
# Evaluate the HYBRID (CNN + classical) model under the SAME adversarial settings used for CNN eval.
#
# Hybrid rule:
#   - Compute CNN logits on (clean or adversarial) input
#   - Gate outputs p(use_cnn)
#   - If p >= threshold => use CNN prediction
#   - Else => use classical prediction computed on-the-fly from the (clean/adv) image
#
# Outputs:
#   - JSON + CSV with per-attack:
#       test/acc_hybrid, test/acc_cnn, test/acc_classical_used (only where routed),
#       route_cnn_frac, route_classical_frac,
#       attack stats (linf_mean, linf_max, l2_mean)
#
# Notes:
#   - Classical features are extracted on-the-fly ONLY for samples routed to classical (speed).
#   - For fairness, classical features are computed on the adversarial images too.
#   - Gate is loaded from gate.pt (saved by train_hybrid_gate.py).
#
# Example:
# python -m attacks.eval_hybrid_attacks \
#   --run_name resnet50_baseline --ckpt_epoch 13 \
#   --gate_pt /.../outputs/hybrid_gate/resnet50_baseline_epoch13/gate.pt \
#   --classical_joblib /.../outputs/classical/.../sgd.joblib \
#   --devices 4 --strategy ddp --batch_size 64 --num_workers 0 \
#   --fgsm_eps "0,0.002,0.004,0.008,0.016,0.032" \
#   --pgd_eps  "0,0.002,0.004,0.008,0.016,0.032" \
#   --pgd_steps_list "5,10,20" \
#   --pgd_alpha 0.002 \
#   --pgd_random_start

from __future__ import annotations

import csv
import json
import torch
import joblib
import argparse
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
from torchvision.datasets import GTSRB
from torchvision import transforms as T
from torchmetrics.classification import MulticlassAccuracy

from utils import config as cfg
from models.resnet50_pl import ResNet50Classifier
from attacks.fgsm import FGSMConfig, fgsm_attack
from attacks.pgd import PGDConfig, pgd_linf_attack

# Base handcrafted extractor (your project module)
from handcrafted.features.feature_extractor import FeatureExtractorConfig, extract_handcrafted_features, feature_dim as base_feature_dim


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def parse_list_of_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_of_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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
# Gate net (must match training)
# -----------------------------
import torch.nn as nn
import torch.nn.functional as F


class GateNet(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        in_dim = int(num_classes) + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        maxp = probs.max(dim=1).values.unsqueeze(1)
        ent = (-probs * (probs.clamp_min(1e-12).log())).sum(dim=1, keepdim=True)
        feat = torch.cat([logits, maxp, ent], dim=1)
        return self.net(feat).squeeze(1)  # [B]


# -----------------------------
# Extra features (LBP + color hist + BoVW(DSIFT/ORB))
# Mirrors the simplified extract_features.py you posted.
# -----------------------------
def tensor_to_bgr_u8(x_chw_01: torch.Tensor) -> np.ndarray:
    # x: [3,H,W] float [0,1]
    x = x_chw_01.detach().cpu().clamp(0, 1)
    rgb = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def lbp_histogram(bgr: np.ndarray, resize: int = 64, bins: int = 256) -> np.ndarray:
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


def color_histogram(
    bgr: np.ndarray,
    resize: int = 64,
    bins: int = 16,
    color_space: str = "hsv",
    norm: str = "l1",
) -> np.ndarray:
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


def orb_descriptors(bgr: np.ndarray, resize: int = 64, max_kp: int = 1500, fast_threshold: int = 10) -> np.ndarray:
    img = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=int(max_kp), fastThreshold=int(fast_threshold))
    _kps, des = orb.detectAndCompute(gray, None)
    if des is None:
        return np.zeros((0, 32), dtype=np.float32)
    return des.astype(np.float32)


def dsift_descriptors(bgr: np.ndarray, resize: int = 64, step: int = 6, kp_size: int = 8) -> np.ndarray:
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


@dataclass
class FeatureFlags:
    hog_resize: int = 64
    hough_resize: int = 96
    use_roi: bool = False
    use_sp_hog: bool = False
    use_lbp: bool = True
    use_color_hist: bool = True
    use_bovw: bool = True
    bovw_type: str = "dsift"  # dsift|orb
    bovw_vocab_size: int = 128

    # extras knobs (match your extraction defaults)
    lbp_resize: int = 64
    lbp_bins: int = 256

    color_resize: int = 64
    color_bins: int = 16
    color_space: str = "hsv"
    color_hist_norm: str = "l1"

    dsift_resize: int = 64
    dsift_step: int = 6
    dsift_kp_size: int = 8

    orb_max_kp: int = 1500
    orb_fast_threshold: int = 10


def load_bovw_centers(features_dir: Path, flags: FeatureFlags) -> Optional[np.ndarray]:
    if not flags.use_bovw:
        return None
    bovw_dir = features_dir / "bovw"
    r = flags.dsift_resize if flags.bovw_type == "dsift" else flags.color_resize
    centers_path = bovw_dir / f"{flags.bovw_type}_centers_k{int(flags.bovw_vocab_size)}_r{int(r)}.npy"
    if not centers_path.exists():
        raise FileNotFoundError(f"BoVW centers not found: {centers_path}")
    return np.load(centers_path).astype(np.float32)


def make_base_cfg(flags: FeatureFlags) -> FeatureExtractorConfig:
    # Your FeatureExtractorConfig may or may not have use_roi/use_sp_hog args (it changed over time).
    # We handle both.
    try:
        base_cfg = FeatureExtractorConfig(use_roi=bool(flags.use_roi), use_sp_hog=bool(flags.use_sp_hog))  # type: ignore
    except TypeError:
        base_cfg = FeatureExtractorConfig()

    # Adjust HOG/Hough resize if possible by reconstructing cfg sub-dataclasses
    try:
        base_cfg = FeatureExtractorConfig(
            roi_cfg=base_cfg.roi_cfg,
            hsv_cfg=base_cfg.hsv_cfg,
            hog_cfg=base_cfg.hog_cfg.__class__(
                resize=int(flags.hog_resize),
                orientations=base_cfg.hog_cfg.orientations,
                pixels_per_cell=base_cfg.hog_cfg.pixels_per_cell,
                cells_per_block=base_cfg.hog_cfg.cells_per_block,
            ),
            hough_cfg=base_cfg.hough_cfg.__class__(
                resize=int(flags.hough_resize),
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
            **({"use_roi": bool(flags.use_roi), "use_sp_hog": bool(flags.use_sp_hog)} if hasattr(base_cfg, "__dict__") else {}),
        )
    except Exception:
        # fallback: keep whatever base cfg uses internally
        pass

    return base_cfg


def extract_full_feature_vector(bgr: np.ndarray, base_cfg: FeatureExtractorConfig, flags: FeatureFlags, centers: Optional[np.ndarray]) -> np.ndarray:
    base_out = extract_handcrafted_features(bgr, base_cfg)
    x_base = base_out["x"].astype(np.float32)

    extras: List[np.ndarray] = []

    if flags.use_lbp:
        extras.append(lbp_histogram(bgr, resize=flags.lbp_resize, bins=flags.lbp_bins))

    if flags.use_color_hist:
        extras.append(
            color_histogram(
                bgr,
                resize=flags.color_resize,
                bins=flags.color_bins,
                color_space=flags.color_space,
                norm=flags.color_hist_norm,
            )
        )

    if flags.use_bovw:
        if centers is None:
            raise RuntimeError("BoVW enabled but centers are None.")
        if flags.bovw_type == "orb":
            des = orb_descriptors(
                bgr,
                resize=flags.color_resize,
                max_kp=flags.orb_max_kp,
                fast_threshold=flags.orb_fast_threshold,
            )
        else:
            des = dsift_descriptors(
                bgr,
                resize=flags.dsift_resize,
                step=flags.dsift_step,
                kp_size=flags.dsift_kp_size,
            )
        extras.append(bovw_encode_hist(des, centers))

    if extras:
        return np.concatenate([x_base] + extras, axis=0).astype(np.float32)
    return x_base


# -----------------------------
# DataModule (test only)
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
            T.ToTensor(),
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
# Lightning eval module
# -----------------------------
class HybridAttackEvalModule(pl.LightningModule):
    """
    Evaluates hybrid under:
      mode = clean | fgsm | pgd
    """
    def __init__(
        self,
        ckpt_path: str,
        gate_pt: str,
        classical_joblib: str,
        features_dir: str,
        gate_threshold: float,
        mode: str,
        flags: FeatureFlags,
        fgsm_cfg: Optional[FGSMConfig] = None,
        pgd_cfg: Optional[PGDConfig] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.fgsm_cfg = fgsm_cfg
        self.pgd_cfg = pgd_cfg
        self.gate_threshold = float(gate_threshold)

        # CNN (frozen)
        self.cnn = ResNet50Classifier.load_from_checkpoint(ckpt_path, map_location="cpu")
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad_(False)

        self.num_classes = int(self.cnn.cfg.num_classes)

        # Gate
        gate_blob = torch.load(gate_pt, map_location="cpu")
        hparams = gate_blob.get("hparams", {})
        hidden = int(hparams.get("gate_hidden", 128))
        self.gate = GateNet(num_classes=self.num_classes, hidden=hidden)
        self.gate.load_state_dict(gate_blob["gate_state_dict"])
        self.gate.eval()
        for p in self.gate.parameters():
            p.requires_grad_(False)

        # Classical
        self.classical = joblib.load(classical_joblib)

        # Feature extractor config & BoVW centers
        self.features_dir = Path(features_dir)
        self.flags = flags
        self.base_cfg = make_base_cfg(flags)
        self.bovw_centers = load_bovw_centers(self.features_dir, flags)

        # Metrics
        self.acc_hybrid = MulticlassAccuracy(num_classes=self.num_classes)
        self.acc_cnn = MulticlassAccuracy(num_classes=self.num_classes)

        # routing + classical-used accuracy (manual sums, distributed safe via all_gather)
        self._sum_total = 0
        self._sum_route_cnn = 0
        self._sum_route_cls = 0
        self._sum_correct_cls_used = 0
        self._sum_total_cls_used = 0

        # attack stats (batch-averaged)
        self._stat_sum = {"linf_mean": 0.0, "linf_max": 0.0, "l2_mean": 0.0}
        self._stat_n = 0

    def on_test_start(self) -> None:
        # ensure grads off for weights, but we still enable autograd globally (Trainer inference_mode=False)
        for p in self.cnn.parameters():
            p.requires_grad_(False)

    def _accumulate_attack_stats(self, st: Dict[str, float]) -> None:
        self._stat_sum["linf_mean"] += st["linf_mean"]
        self._stat_sum["linf_max"] = max(self._stat_sum["linf_max"], st["linf_max"])
        self._stat_sum["l2_mean"] += st["l2_mean"]
        self._stat_n += 1

    def _classical_predict_for_subset(self, x_subset: torch.Tensor) -> torch.Tensor:
        # x_subset: [N,3,H,W] float [0,1] on GPU/CPU
        # We compute features on CPU and run joblib model
        feats: List[np.ndarray] = []
        for i in range(x_subset.size(0)):
            bgr = tensor_to_bgr_u8(x_subset[i])
            f = extract_full_feature_vector(bgr, self.base_cfg, self.flags, self.bovw_centers)
            feats.append(f)

        X = np.stack(feats, axis=0).astype(np.float32, copy=False)
        pred = self.classical.predict(X)
        return torch.from_numpy(np.asarray(pred, dtype=np.int64)).to(self.device)

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        x_in = x
        if self.mode == "fgsm":
            assert self.fgsm_cfg is not None
            with torch.enable_grad():
                x_adv = fgsm_attack(self.cnn, x, y, normalize_batch, self.fgsm_cfg)
            self._accumulate_attack_stats(attack_stats(x, x_adv))
            x_in = x_adv.detach()
        elif self.mode == "pgd":
            assert self.pgd_cfg is not None
            with torch.enable_grad():
                x_adv = pgd_linf_attack(self.cnn, x, y, normalize_batch, self.pgd_cfg)
            self._accumulate_attack_stats(attack_stats(x, x_adv))
            x_in = x_adv.detach()
        else:
            # clean
            pass

        with torch.no_grad():
            logits = self.cnn(normalize_batch(x_in))
            pred_cnn = logits.argmax(dim=1)

            gate_logits = self.gate(logits)
            p = torch.sigmoid(gate_logits)
            use_cnn = (p >= self.gate_threshold)

        # Route to classical only for subset
        pred_hybrid = pred_cnn.clone()
        if (~use_cnn).any():
            x_sub = x_in[~use_cnn]
            pred_cls = self._classical_predict_for_subset(x_sub)
            pred_hybrid[~use_cnn] = pred_cls

            # classical-used accuracy
            y_sub = y[~use_cnn]
            self._sum_correct_cls_used += int((pred_cls == y_sub).sum().item())
            self._sum_total_cls_used += int(y_sub.numel())

        # Update accuracies
        self.acc_hybrid.update(pred_hybrid, y)
        self.acc_cnn.update(pred_cnn, y)

        # Route counts
        self._sum_total += int(y.numel())
        self._sum_route_cnn += int(use_cnn.sum().item())
        self._sum_route_cls += int((~use_cnn).sum().item())

    def on_test_epoch_end(self) -> None:
        # sync accuracies
        acc_h = self.acc_hybrid.compute()
        acc_c = self.acc_cnn.compute()

        # distributed-safe sums
        sums = torch.tensor(
            [self._sum_total, self._sum_route_cnn, self._sum_route_cls, self._sum_correct_cls_used, self._sum_total_cls_used],
            device=self.device,
            dtype=torch.float32,
        )
        gathered = self.all_gather(sums)
        if gathered.ndim == 2:
            sums_all = gathered.sum(dim=0)
        else:
            sums_all = gathered

        total = max(1.0, float(sums_all[0].item()))
        route_cnn = float(sums_all[1].item())
        route_cls = float(sums_all[2].item())
        cls_correct = float(sums_all[3].item())
        cls_total = float(sums_all[4].item())

        frac_cnn = route_cnn / total
        frac_cls = route_cls / total
        acc_cls_used = (cls_correct / max(1.0, cls_total))

        self.log("test/acc_hybrid", acc_h, prog_bar=True, sync_dist=True)
        self.log("test/acc_cnn", acc_c, prog_bar=False, sync_dist=True)
        self.log("test/route_cnn_frac", frac_cnn, prog_bar=True, sync_dist=True)
        self.log("test/route_classical_frac", frac_cls, prog_bar=True, sync_dist=True)
        self.log("test/acc_classical_used", acc_cls_used, prog_bar=False, sync_dist=True)

        # attack stats (rank0 only meaningful; still log a reasonable per-rank average)
        if self._stat_n > 0:
            self.log("attack/linf_mean", self._stat_sum["linf_mean"] / self._stat_n, prog_bar=False, sync_dist=False)
            self.log("attack/linf_max", self._stat_sum["linf_max"], prog_bar=False, sync_dist=False)
            self.log("attack/l2_mean", self._stat_sum["l2_mean"] / self._stat_n, prog_bar=False, sync_dist=False)

        self.acc_hybrid.reset()
        self.acc_cnn.reset()

    def configure_optimizers(self):
        return None


def run_test(trainer: pl.Trainer, dm: pl.LightningDataModule, module: pl.LightningModule) -> Dict[str, float]:
    out = trainer.test(module, datamodule=dm, verbose=False)
    if not out:
        return {}
    return {k: float(v) for k, v in out[0].items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Hybrid adversarial eval (same settings as CNN eval)")

    p.add_argument("--data_dir", type=str, default=str(cfg.DATA_PATH / "data_raw"))
    p.add_argument("--run_name", type=str, default="resnet50_baseline")
    p.add_argument("--ckpt_epoch", type=int, required=True)

    p.add_argument("--gate_pt", type=str, required=True)
    p.add_argument("--classical_joblib", type=str, required=True)

    p.add_argument("--features_dir", type=str, default=str(Path(cfg.OUTPUT_DIR) / "features"))
    p.add_argument("--gate_threshold", type=float, default=-1.0, help="If <0, use threshold from gate.pt hparams if present, else 0.5")

    # Feature flags (must match the classical training / centers naming)
    p.add_argument("--hog_resize", type=int, default=64)
    p.add_argument("--hough_resize", type=int, default=96)
    p.add_argument("--use_roi", action="store_true")
    p.add_argument("--use_sp_hog", action="store_true")
    p.add_argument("--no_lbp", action="store_true")
    p.add_argument("--no_color_hist", action="store_true")
    p.add_argument("--no_bovw", action="store_true")
    p.add_argument("--bovw_type", type=str, choices=["dsift", "orb"], default="dsift")
    p.add_argument("--bovw_vocab_size", type=int, default=128)

    # Extra knobs (defaults match your extraction script)
    p.add_argument("--dsift_resize", type=int, default=64)
    p.add_argument("--dsift_step", type=int, default=6)
    p.add_argument("--dsift_kp_size", type=int, default=8)
    p.add_argument("--color_resize", type=int, default=64)
    p.add_argument("--color_bins", type=int, default=16)
    p.add_argument("--color_space", type=str, choices=["hsv", "rgb"], default="hsv")
    p.add_argument("--lbp_resize", type=int, default=64)

    # loader
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)

    # trainer
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strategy", type=str, default="ddp")
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--matmul_precision", type=str, choices=["highest", "high", "medium"], default="high")

    # sweeps (same interface as CNN eval)
    p.add_argument("--fgsm_eps", type=str, default="0.0,0.002,0.004,0.008,0.016,0.032")
    p.add_argument("--pgd_eps", type=str, default="0.0,0.002,0.004,0.008,0.016,0.032")
    p.add_argument("--pgd_steps_list", type=str, default="5,10,20")
    p.add_argument("--pgd_alpha", type=float, default=0.002)
    p.add_argument("--pgd_random_start", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)

    ckpt_path = resolve_ckpt_path(Path(cfg.OUTPUT_DIR), args.run_name, args.ckpt_epoch)

    # gate threshold: prefer saved threshold if present
    gate_blob = torch.load(args.gate_pt, map_location="cpu")
    saved_thr = None
    try:
        saved_thr = float(gate_blob.get("hparams", {}).get("gate_threshold", None))
    except Exception:
        saved_thr = None
    gate_threshold = float(args.gate_threshold) if float(args.gate_threshold) >= 0 else (saved_thr if saved_thr is not None else 0.5)

    flags = FeatureFlags(
        hog_resize=int(args.hog_resize),
        hough_resize=int(args.hough_resize),
        use_roi=bool(args.use_roi),
        use_sp_hog=bool(args.use_sp_hog),
        use_lbp=not bool(args.no_lbp),
        use_color_hist=not bool(args.no_color_hist),
        use_bovw=not bool(args.no_bovw),
        bovw_type=str(args.bovw_type),
        bovw_vocab_size=int(args.bovw_vocab_size),
        dsift_resize=int(args.dsift_resize),
        dsift_step=int(args.dsift_step),
        dsift_kp_size=int(args.dsift_kp_size),
        color_resize=int(args.color_resize),
        color_bins=int(args.color_bins),
        color_space=str(args.color_space),
        lbp_resize=int(args.lbp_resize),
    )

    dm = GTSRBTestDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy if args.devices > 1 and args.accelerator == "gpu" else "auto",
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,
    )

    out_root = Path(cfg.OUTPUT_DIR) / "attacks_hybrid" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_name": args.run_name,
        "ckpt_epoch": args.ckpt_epoch,
        "ckpt_path": str(ckpt_path),
        "gate_pt": str(args.gate_pt),
        "gate_threshold": float(gate_threshold),
        "classical_joblib": str(args.classical_joblib),
        "features_dir": str(args.features_dir),
        "feature_flags": asdict(flags),
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "img_size": args.img_size,
        "devices": args.devices,
        "strategy": trainer.strategy.__class__.__name__,
        "precision": args.precision,
        "clean": {},
        "fgsm": [],
        "pgd": [],
    }

    if trainer.is_global_zero:
        print(f"\nResolved checkpoint: {ckpt_path}")
        print(f"Gate threshold: {gate_threshold}\n")

    # CLEAN
    if trainer.is_global_zero:
        print("Running CLEAN hybrid eval...")
    clean_module = HybridAttackEvalModule(
        ckpt_path=str(ckpt_path),
        gate_pt=str(args.gate_pt),
        classical_joblib=str(args.classical_joblib),
        features_dir=str(args.features_dir),
        gate_threshold=float(gate_threshold),
        mode="clean",
        flags=flags,
    )
    results["clean"] = run_test(trainer, dm, clean_module)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FGSM sweep
    fgsm_eps_list = parse_list_of_floats(args.fgsm_eps)
    if trainer.is_global_zero:
        print("\nRunning FGSM sweep (hybrid)...")
    for eps in tqdm(fgsm_eps_list, disable=not trainer.is_global_zero, desc="FGSM eps"):
        m = HybridAttackEvalModule(
            ckpt_path=str(ckpt_path),
            gate_pt=str(args.gate_pt),
            classical_joblib=str(args.classical_joblib),
            features_dir=str(args.features_dir),
            gate_threshold=float(gate_threshold),
            mode="fgsm",
            flags=flags,
            fgsm_cfg=FGSMConfig(eps=float(eps)),
        )
        metrics = run_test(trainer, dm, m)
        results["fgsm"].append({"eps": float(eps), **metrics})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # PGD sweep
    pgd_eps_list = parse_list_of_floats(args.pgd_eps)
    pgd_steps_list = parse_list_of_ints(args.pgd_steps_list)

    if trainer.is_global_zero:
        print("\nRunning PGD sweep (hybrid)...")
    for steps in tqdm(pgd_steps_list, disable=not trainer.is_global_zero, desc="PGD steps"):
        for eps in tqdm(pgd_eps_list, disable=not trainer.is_global_zero, desc=f"PGD eps (steps={steps})", leave=False):
            pgd_cfg = PGDConfig(
                eps=float(eps),
                alpha=float(args.pgd_alpha),
                steps=int(steps),
                random_start=bool(args.pgd_random_start),
            )
            m = HybridAttackEvalModule(
                ckpt_path=str(ckpt_path),
                gate_pt=str(args.gate_pt),
                classical_joblib=str(args.classical_joblib),
                features_dir=str(args.features_dir),
                gate_threshold=float(gate_threshold),
                mode="pgd",
                flags=flags,
                pgd_cfg=pgd_cfg,
            )
            metrics = run_test(trainer, dm, m)
            results["pgd"].append({"eps": float(eps), **asdict(pgd_cfg), **metrics})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # SAVE (rank0)
    json_path = out_root / f"hybrid_attack_results_epoch{args.ckpt_epoch}.json"
    csv_path = out_root / f"hybrid_attack_results_epoch{args.ckpt_epoch}.csv"

    if trainer.is_global_zero:
        json_path.write_text(json.dumps(results, indent=2))

        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "attack", "eps", "steps", "alpha", "random_start",
                "test/acc_hybrid", "test/acc_cnn", "test/acc_classical_used",
                "test/route_cnn_frac", "test/route_classical_frac",
                "attack/linf_mean", "attack/linf_max", "attack/l2_mean",
                "clean_test/acc_hybrid",
                "ckpt_epoch", "ckpt_path", "gate_threshold"
            ])

            clean_acc_h = results["clean"].get("test/acc_hybrid", "")

            for r in results["fgsm"]:
                w.writerow([
                    "fgsm",
                    r.get("eps", ""), "", "", "",
                    r.get("test/acc_hybrid", ""),
                    r.get("test/acc_cnn", ""),
                    r.get("test/acc_classical_used", ""),
                    r.get("test/route_cnn_frac", ""),
                    r.get("test/route_classical_frac", ""),
                    r.get("attack/linf_mean", ""),
                    r.get("attack/linf_max", ""),
                    r.get("attack/l2_mean", ""),
                    clean_acc_h,
                    args.ckpt_epoch,
                    str(ckpt_path),
                    gate_threshold,
                ])

            for r in results["pgd"]:
                w.writerow([
                    "pgd",
                    r.get("eps", ""),
                    r.get("steps", ""),
                    r.get("alpha", ""),
                    r.get("random_start", ""),
                    r.get("test/acc_hybrid", ""),
                    r.get("test/acc_cnn", ""),
                    r.get("test/acc_classical_used", ""),
                    r.get("test/route_cnn_frac", ""),
                    r.get("test/route_classical_frac", ""),
                    r.get("attack/linf_mean", ""),
                    r.get("attack/linf_max", ""),
                    r.get("attack/l2_mean", ""),
                    clean_acc_h,
                    args.ckpt_epoch,
                    str(ckpt_path),
                    gate_threshold,
                ])

        print("\n===== HYBRID ATTACK EVAL SUMMARY (rank0) =====")
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV : {csv_path}")
        print("=============================================\n")


if __name__ == "__main__":
    main()