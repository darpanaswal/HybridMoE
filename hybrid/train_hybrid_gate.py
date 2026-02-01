# train_hybrid_gate.py
# Train a Mixture-of-Experts (MoE) gate that decides: use CNN vs use handcrafted (classical) classifier.
# - CNN (ResNet50Classifier) is FROZEN
# - Classical (joblib SGD pipeline) is FROZEN
# - Gate is TRAINED on a 50/50 mix of clean + adversarial inputs (label = clean(1) vs adv(0))
# - Testing reports:
#     1) Hybrid test accuracy (hard routing by gate)
#     2) Gate preference: fraction routed to CNN vs classical
#     3) CNN-only and classical-only test accuracies (clean)

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import GTSRB
from torchvision import transforms as T

from utils import config as cfg
from models.resnet50_pl import ResNet50Classifier
from attacks.fgsm import FGSMConfig, fgsm_attack
from attacks.pgd import PGDConfig, pgd_linf_attack


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class WithIndex(Dataset):
    """Wrap a torchvision dataset to also return the integer index."""
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return idx, x, y


class GTSRBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        img_size: int,
        batch_size: int,
        num_workers: int,
        val_frac: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_frac = float(val_frac)
        self.seed = int(seed)

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        tfm = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),  # [0,1]
        ])

        full_train = GTSRB(root=self.data_dir, split="train", download=False, transform=tfm)
        full_test = GTSRB(root=self.data_dir, split="test", download=False, transform=tfm)

        # Stratified split for val
        y = np.array([full_train[i][1] for i in range(len(full_train))], dtype=np.int64)
        n = len(full_train)
        idx = np.arange(n)

        # simple stratified shuffle split without sklearn dependency here
        rng = np.random.default_rng(self.seed)
        val_frac = max(0.01, min(0.5, self.val_frac))

        train_idx: List[int] = []
        val_idx: List[int] = []
        for c in np.unique(y):
            c_idx = idx[y == c]
            rng.shuffle(c_idx)
            k = int(round(len(c_idx) * val_frac))
            val_idx.extend(c_idx[:k].tolist())
            train_idx.extend(c_idx[k:].tolist())

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        self.train_ds = torch.utils.data.Subset(full_train, train_idx)
        self.val_ds = torch.utils.data.Subset(full_train, val_idx)
        self.test_ds = full_test

        # attach indices for feature lookup
        self.train_ds = WithIndex(self.train_ds)
        self.val_ds = WithIndex(self.val_ds)
        self.test_ds = WithIndex(self.test_ds)

    def train_dataloader(self):
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            drop_last=True,  # important for exact 50/50 split
        )

    def val_dataloader(self):
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )

    def test_dataloader(self):
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )


# ----------------------------
# Feature .npz resolution (clean test classical preds)
# ----------------------------
def _feature_key(
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


def resolve_feature_npz(
    features_dir: Path,
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
) -> Path:
    # Primary expected filename
    primary = features_dir / f"{_feature_key(split, hog_resize, hough_resize, use_roi, use_sp_hog, use_lbp, use_color_hist, use_bovw, bovw_type, bovw_k, cache_dtype)}.npz"
    if primary.exists():
        return primary

    # Fallback: search any npz matching split + core tokens (handles slight naming drift)
    want = [
        f"{split}_hog{int(hog_resize)}",
        f"hough{int(hough_resize)}",
        ("roi" if use_roi else ""),
        ("sphog" if use_sp_hog else ""),
        ("lbp" if use_lbp else ""),
        ("ch" if use_color_hist else ""),
        (f"bovw{bovw_type}" if use_bovw else ""),
        (f"k{int(bovw_k)}" if use_bovw else ""),
        f"dtype{cache_dtype}",
    ]
    want = [w for w in want if w]

    cands = sorted(features_dir.glob(f"{split}_*.npz"))
    for p in cands:
        name = p.name
        if all(w in name for w in want):
            return p

    raise FileNotFoundError(f"Could not resolve {split} features under {features_dir} for requested flags.")


def load_npz_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    return X, y


# ----------------------------
# Gate network
# ----------------------------
class GateNet(nn.Module):
    """
    Gate input = [logits (C dims), max_prob (1), entropy (1)] => output logit (1)
    Target:
      clean -> 1  (route to CNN)
      adv   -> 0  (route to Classical)
    """
    def __init__(self, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        in_dim = int(num_classes) + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]
        probs = F.softmax(logits, dim=1)
        maxp = probs.max(dim=1).values.unsqueeze(1)
        ent = (-probs * (probs.clamp_min(1e-12).log())).sum(dim=1, keepdim=True)
        feat = torch.cat([logits, maxp, ent], dim=1)
        return self.net(feat).squeeze(1)  # [B]


@dataclass
class AdvPolicy:
    method: str  # mixed|fgsm|pgd
    # single knobs
    fgsm_eps: float
    pgd_eps: float
    pgd_steps: int
    pgd_alpha: float
    pgd_random_start: bool

    # internal pools for "mixed"
    fgsm_pool: Tuple[float, ...] = (0.002, 0.004, 0.008, 0.016)
    pgd_pool: Tuple[float, ...] = (0.004, 0.008, 0.016, 0.032)
    mixed_pgd_prob: float = 0.75  # 75% PGD, 25% FGSM


class HybridGateModule(pl.LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        classical_joblib: str,
        features_dir: str,
        # feature flags for resolving CLEAN test features
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
        # training
        lr: float,
        gate_hidden: int,
        adv_ratio: float,
        gate_threshold: float,
        lambda_gate: float,
        adv_policy: AdvPolicy,
        # test
        report_dir: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["adv_policy"])

        # Load + freeze CNN
        self.cnn = ResNet50Classifier.load_from_checkpoint(ckpt_path, map_location="cpu")
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad_(False)

        self.num_classes = int(self.cnn.cfg.num_classes)

        # Gate
        self.gate = GateNet(num_classes=self.num_classes, hidden=int(gate_hidden))
        self.bce = nn.BCEWithLogitsLoss()

        # Classical model (CPU / joblib)
        self.classical = joblib.load(classical_joblib)

        # Clean-test cached features (loaded lazily in setup("test"))
        self.features_dir = Path(features_dir)
        self.feature_flags = dict(
            hog_resize=int(hog_resize),
            hough_resize=int(hough_resize),
            use_roi=bool(use_roi),
            use_sp_hog=bool(use_sp_hog),
            use_lbp=bool(use_lbp),
            use_color_hist=bool(use_color_hist),
            use_bovw=bool(use_bovw),
            bovw_type=str(bovw_type),
            bovw_k=int(bovw_k),
            cache_dtype=str(cache_dtype),
        )
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._test_npz: Optional[Path] = None

        # Training policy
        self.adv_ratio = float(adv_ratio)
        self.gate_threshold = float(gate_threshold)
        self.lambda_gate = float(lambda_gate)
        self.adv_policy = adv_policy

        # Reporting
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # accumulators for test (manual, per-rank)
        self._t_correct_hybrid = 0
        self._t_total = 0
        self._t_route_cnn = 0
        self._t_route_cls = 0
        self._t_correct_cnn = 0
        self._t_correct_cls = 0

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" or stage is None:
            # Resolve and load CLEAN test features for classical predictions
            test_npz = resolve_feature_npz(
                features_dir=self.features_dir,
                split="test",
                **self.feature_flags,
            )
            X, y = load_npz_features(test_npz)
            self._X_test = X
            self._y_test = y
            self._test_npz = test_npz

    def configure_optimizers(self):
        return torch.optim.AdamW(self.gate.parameters(), lr=float(self.hparams.lr))

    # ---------
    # Adversarial sampling (CNN-breaking by default)
    # ---------
    def _sample_adv(self) -> Tuple[str, Optional[FGSMConfig], Optional[PGDConfig]]:
        pol = self.adv_policy
        if pol.method == "fgsm":
            return "fgsm", FGSMConfig(eps=float(pol.fgsm_eps)), None
        if pol.method == "pgd":
            return "pgd", None, PGDConfig(
                eps=float(pol.pgd_eps),
                alpha=float(pol.pgd_alpha),
                steps=int(pol.pgd_steps),
                random_start=bool(pol.pgd_random_start),
            )

        # mixed: 75% PGD from pool, 25% FGSM from pool
        r = random.random()
        if r < float(pol.mixed_pgd_prob):
            eps = random.choice(list(pol.pgd_pool))
            return "pgd", None, PGDConfig(
                eps=float(eps),
                alpha=float(pol.pgd_alpha),
                steps=int(pol.pgd_steps),
                random_start=bool(pol.pgd_random_start),
            )
        else:
            eps = random.choice(list(pol.fgsm_pool))
            return "fgsm", FGSMConfig(eps=float(eps)), None

    def _make_half_adv(self, x_adv_src: torch.Tensor, y_adv_src: torch.Tensor) -> torch.Tensor:
        mode, fgsm_cfg, pgd_cfg = self._sample_adv()
        if mode == "fgsm":
            assert fgsm_cfg is not None
            with torch.enable_grad():
                return fgsm_attack(self.cnn, x_adv_src, y_adv_src, normalize_batch, fgsm_cfg)
        else:
            assert pgd_cfg is not None
            with torch.enable_grad():
                return pgd_linf_attack(self.cnn, x_adv_src, y_adv_src, normalize_batch, pgd_cfg)

    # ---------
    # Training/validation: gate learns clean vs adv (and optionally mixture task)
    # ---------
    def _step_gate_train(self, batch: Any, stage: str) -> torch.Tensor:
        idx, x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        B = x.size(0)
        if B < 2:
            return torch.tensor(0.0, device=self.device)

        # enforce even split for exact 50/50 (or requested ratio)
        adv_k = int(round(B * self.adv_ratio))
        adv_k = max(1, min(B - 1, adv_k))
        clean_k = B - adv_k
        if clean_k <= 0:
            clean_k = B // 2
            adv_k = B - clean_k

        # clean block and adv block
        x_clean = x[:clean_k]
        y_clean = y[:clean_k]

        x_adv_src = x[clean_k:clean_k + adv_k]
        y_adv_src = y[clean_k:clean_k + adv_k]

        x_adv = self._make_half_adv(x_adv_src, y_adv_src).detach()

        # Gate targets: clean=1, adv=0
        t_clean = torch.ones((x_clean.size(0),), device=self.device, dtype=torch.float32)
        t_adv = torch.zeros((x_adv.size(0),), device=self.device, dtype=torch.float32)

        x_mix = torch.cat([x_clean, x_adv], dim=0)
        t_mix = torch.cat([t_clean, t_adv], dim=0)

        # CNN logits (frozen)
        with torch.no_grad():
            logits = self.cnn(normalize_batch(x_mix))  # [B, C]

        gate_logits = self.gate(logits)  # [B]
        loss_gate = self.bce(gate_logits, t_mix)

        # Metrics
        with torch.no_grad():
            p = torch.sigmoid(gate_logits)
            pred_gate = (p >= 0.5).float()
            gate_acc = (pred_gate == t_mix).float().mean()

            route_cnn = (p >= self.gate_threshold).float().mean()
            route_cls = 1.0 - route_cnn

        self.log(f"{stage}/loss_gate", loss_gate, prog_bar=(stage != "train"), on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}/gate_acc", gate_acc, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}/route_cnn_frac", route_cnn, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/route_cls_frac", route_cls, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss_gate

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step_gate_train(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        _ = self._step_gate_train(batch, "val")

    # ---------
    # Testing: hard routing hybrid accuracy + preference stats (CLEAN)
    # ---------
    @torch.no_grad()
    def _classical_predict_from_cached_features(self, idxs: torch.Tensor) -> torch.Tensor:
        assert self._X_test is not None, "Test features not loaded. Did setup('test') run?"
        X = self._X_test[idxs.cpu().numpy()].astype(np.float32, copy=False)
        pred = self.classical.predict(X)
        return torch.from_numpy(np.asarray(pred, dtype=np.int64))

    def on_test_start(self) -> None:
        self._t_correct_hybrid = 0
        self._t_total = 0
        self._t_route_cnn = 0
        self._t_route_cls = 0
        self._t_correct_cnn = 0
        self._t_correct_cls = 0

    def test_step(self, batch: Any, batch_idx: int) -> None:
        idx, x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        idx = idx.to(self.device, non_blocking=True)

        logits_cnn = self.cnn(normalize_batch(x))
        pred_cnn = logits_cnn.argmax(dim=1)

        gate_logits = self.gate(logits_cnn)
        p = torch.sigmoid(gate_logits)
        use_cnn = (p >= self.gate_threshold)

        # Classical preds via cached CLEAN features
        pred_cls = self._classical_predict_from_cached_features(idx).to(self.device)

        pred_hybrid = torch.where(use_cnn, pred_cnn, pred_cls)

        self._t_total += int(y.numel())
        self._t_correct_hybrid += int((pred_hybrid == y).sum().item())
        self._t_correct_cnn += int((pred_cnn == y).sum().item())
        self._t_correct_cls += int((pred_cls == y).sum().item())
        self._t_route_cnn += int(use_cnn.sum().item())
        self._t_route_cls += int((~use_cnn).sum().item())

    def on_test_epoch_end(self) -> None:
        # Per-rank metrics; Lightning will sync_dist for logged scalars.
        total = max(1, self._t_total)
        acc_hybrid = float(self._t_correct_hybrid / total)
        acc_cnn = float(self._t_correct_cnn / total)
        acc_cls = float(self._t_correct_cls / total)
        frac_cnn = float(self._t_route_cnn / total)
        frac_cls = float(self._t_route_cls / total)

        self.log("test/acc_hybrid", acc_hybrid, prog_bar=True, sync_dist=True)
        self.log("test/acc_cnn", acc_cnn, prog_bar=False, sync_dist=True)
        self.log("test/acc_classical", acc_cls, prog_bar=False, sync_dist=True)
        self.log("test/route_cnn_frac", frac_cnn, prog_bar=True, sync_dist=True)
        self.log("test/route_classical_frac", frac_cls, prog_bar=True, sync_dist=True)

        # Save rank0 report
        if self.trainer.is_global_zero:
            report: Dict[str, Any] = {
                "ckpt_path": str(self.hparams.ckpt_path),
                "classical_joblib": str(self.hparams.classical_joblib),
                "test_feature_npz": str(self._test_npz) if self._test_npz is not None else None,
                "gate_threshold": float(self.gate_threshold),
                "feature_flags": self.feature_flags,
                "results": {
                    "acc_hybrid": acc_hybrid,
                    "acc_cnn": acc_cnn,
                    "acc_classical": acc_cls,
                    "route_cnn_frac": frac_cnn,
                    "route_classical_frac": frac_cls,
                    "n_test": int(total),
                },
            }
            out_path = self.report_dir / "test_report.json"
            out_path.write_text(json.dumps(report, indent=2))
            print("\n=== TEST REPORT (rank0) ===")
            print(json.dumps(report, indent=2))
            print("===========================\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train hybrid MoE gate (CNN vs Classical) + test hybrid accuracy and gate preference")

    # Data / model paths
    p.add_argument("--data_dir", type=str, default=str(cfg.DATA_PATH / "data_raw"))
    p.add_argument("--ckpt_epoch", type=int, required=True)
    p.add_argument("--run_name", type=str, default="resnet50_baseline")
    p.add_argument("--classical_joblib", type=str, required=True)

    # Features for CLEAN test classical predictions
    p.add_argument("--features_dir", type=str, default=str(Path(cfg.OUTPUT_DIR) / "features"))
    p.add_argument("--hog_resize", type=int, default=64)
    p.add_argument("--hough_resize", type=int, default=96)
    p.add_argument("--use_roi", action="store_true")
    p.add_argument("--use_sp_hog", action="store_true")
    p.add_argument("--no_lbp", action="store_true")
    p.add_argument("--no_color_hist", action="store_true")
    p.add_argument("--no_bovw", action="store_true")
    p.add_argument("--bovw_type", type=str, choices=["dsift", "orb"], default="dsift")
    p.add_argument("--bovw_vocab_size", type=int, default=128)
    p.add_argument("--cache_dtype", type=str, choices=["float16", "float32"], default="float16")

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--adv_ratio", type=float, default=0.5)  # 50/50 by default
    p.add_argument("--gate_threshold", type=float, default=0.5)  # p>=thr => CNN else classical
    p.add_argument("--lambda_gate", type=float, default=1.0)

    # Adversarial policy (NO long lists)
    p.add_argument("--adv_method", type=str, choices=["mixed", "fgsm", "pgd"], default="mixed")
    p.add_argument("--fgsm_eps", type=float, default=0.002)
    p.add_argument("--pgd_eps", type=float, default=0.016)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--pgd_alpha", type=float, default=0.002)
    p.add_argument("--pgd_no_random_start", action="store_true")

    # Trainer
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strategy", type=str, default="ddp")
    p.add_argument("--precision", type=str, default="16-mixed")

    # Output
    p.add_argument("--out_dir", type=str, default=str(Path(cfg.OUTPUT_DIR) / "hybrid_gate"))

    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    out_dir = Path(args.out_dir) / f"{args.run_name}_epoch{args.ckpt_epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_ckpt_path(Path(cfg.OUTPUT_DIR), str(args.run_name), int(args.ckpt_epoch))

    use_lbp = not bool(args.no_lbp)
    use_color_hist = not bool(args.no_color_hist)
    use_bovw = not bool(args.no_bovw)

    adv_policy = AdvPolicy(
        method=str(args.adv_method),
        fgsm_eps=float(args.fgsm_eps),
        pgd_eps=float(args.pgd_eps),
        pgd_steps=int(args.pgd_steps),
        pgd_alpha=float(args.pgd_alpha),
        pgd_random_start=not bool(args.pgd_no_random_start),
    )

    dm = GTSRBDataModule(
        data_dir=str(args.data_dir),
        img_size=int(args.img_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
    )

    module = HybridGateModule(
        ckpt_path=str(ckpt_path),
        classical_joblib=str(args.classical_joblib),
        features_dir=str(args.features_dir),
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
        lr=float(args.lr),
        gate_hidden=int(args.gate_hidden),
        adv_ratio=float(args.adv_ratio),
        gate_threshold=float(args.gate_threshold),
        lambda_gate=float(args.lambda_gate),
        adv_policy=adv_policy,
        report_dir=str(out_dir),
    )

    trainer = pl.Trainer(
        accelerator=str(args.accelerator),
        devices=int(args.devices),
        strategy=str(args.strategy) if int(args.devices) > 1 and str(args.accelerator) == "gpu" else "auto",
        precision=str(args.precision),
        max_epochs=int(args.epochs),
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,  # needed because we generate adversarial examples
        deterministic=True,
    )

    # Train
    trainer.fit(module, datamodule=dm)

    # Save gate weights (rank0)
    if trainer.is_global_zero:
        torch.save({"gate_state_dict": module.gate.state_dict(), "hparams": dict(module.hparams)}, out_dir / "gate.pt")

    # Test (clean hybrid accuracy + preference stats)
    trainer.test(module, datamodule=dm, verbose=False)


if __name__ == "__main__":
    main()