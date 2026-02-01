import csv
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from dataclasses import asdict
from utils import config as cfg
from torchvision.datasets import GTSRB
from torchvision import transforms as T
from torchvision.utils import save_image
from attacks.fgsm import FGSMConfig, fgsm_attack
from models.resnet50_pl import ResNet50Classifier
from attacks.pgd import PGDConfig, pgd_linf_attack
from typing import List, Optional, Any, Dict, Tuple
from torchmetrics.classification import MulticlassAccuracy


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
            T.ToTensor(),  # pixel space [0,1]
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


def attack_stats(x: torch.Tensor, x_adv: torch.Tensor) -> Dict[str, float]:
    """
    Returns quick sanity stats computed on THIS batch:
      - mean/max Linf
      - mean L2
    """
    delta = (x_adv - x).detach()
    linf = delta.abs().flatten(1).max(dim=1).values  # per-sample
    l2 = torch.sqrt((delta ** 2).flatten(1).sum(dim=1) + 1e-12)
    return {
        "linf_mean": float(linf.mean().item()),
        "linf_max": float(linf.max().item()),
        "l2_mean": float(l2.mean().item()),
    }


class AttackEvalModule(pl.LightningModule):
    """
    LightningModule used ONLY for evaluation.
    Supports: clean / fgsm / pgd, and logs accuracy + attack sanity stats.
    """
    def __init__(
        self,
        ckpt_path: str,
        mode: str,  # "clean" | "fgsm" | "pgd"
        fgsm_cfg: Optional[FGSMConfig] = None,
        pgd_cfg: Optional[PGDConfig] = None,
        save_examples_dir: Optional[str] = None,
        save_n_batches: int = 0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.fgsm_cfg = fgsm_cfg
        self.pgd_cfg = pgd_cfg

        self.model = ResNet50Classifier.load_from_checkpoint(ckpt_path, map_location="cpu")
        self.model.eval()

        self.acc = MulticlassAccuracy(num_classes=self.model.cfg.num_classes)

        self.save_examples_dir = Path(save_examples_dir) if save_examples_dir else None
        self.save_n_batches = int(save_n_batches)

        # aggregate stats over epoch (manual)
        self._stat_sum = {"linf_mean": 0.0, "linf_max": 0.0, "l2_mean": 0.0}
        self._stat_n = 0

    def on_test_start(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if self.mode == "clean":
            with torch.no_grad():
                logits = self.model(normalize_batch(x))
                preds = logits.argmax(dim=1)
                self.acc.update(preds, y)
            return

        if self.mode == "fgsm":
            assert self.fgsm_cfg is not None
            with torch.enable_grad():
                x_adv = fgsm_attack(self.model, x, y, normalize_batch, self.fgsm_cfg)
            st = attack_stats(x, x_adv)
            self._accumulate_stats(st)

            with torch.no_grad():
                logits = self.model(normalize_batch(x_adv))
                preds = logits.argmax(dim=1)
                self.acc.update(preds, y)

            self._maybe_save_examples(x, x_adv, batch_idx)
            return

        if self.mode == "pgd":
            assert self.pgd_cfg is not None
            with torch.enable_grad():
                x_adv = pgd_linf_attack(self.model, x, y, normalize_batch, self.pgd_cfg)
            st = attack_stats(x, x_adv)
            self._accumulate_stats(st)

            with torch.no_grad():
                logits = self.model(normalize_batch(x_adv))
                preds = logits.argmax(dim=1)
                self.acc.update(preds, y)

            self._maybe_save_examples(x, x_adv, batch_idx)
            return

        raise ValueError(f"Unknown mode: {self.mode}")

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

        # Log stats (sync-dist not needed for max; we will aggregate on rank0 in python after test)
        # Here, we log per-rank averages; the outer script will record rank0 values only.
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
    # out[0] includes logged metrics (on rank0)
    return {k: float(v) for k, v in out[0].items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Distributed adversarial eval (DDP) with progress + sanity stats")

    p.add_argument("--data_dir", type=str, default=str(cfg.DATA_PATH / "data_raw"))
    p.add_argument("--run_name", type=str, default="resnet50_baseline")
    p.add_argument("--ckpt_epoch", type=int, required=True)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)

    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strategy", type=str, default="ddp")
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--matmul_precision", type=str, choices=["highest", "high", "medium"], default="high")

    # eps sweeps
    p.add_argument("--fgsm_eps", type=str, default="0.0,0.002,0.004,0.008,0.016")

    # PGD sweeps
    p.add_argument("--pgd_eps", type=str, default="0.0,0.002,0.004,0.008,0.016")
    p.add_argument("--pgd_steps_list", type=str, default="10", help="Comma-separated, e.g. 5,10,20")
    p.add_argument("--pgd_alpha", type=float, default=0.002)
    p.add_argument("--pgd_random_start", action="store_true")

    # examples
    p.add_argument("--save_examples", action="store_true")
    p.add_argument("--save_n_batches", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)

    ckpt_path = resolve_ckpt_path(Path(cfg.OUTPUT_DIR), args.run_name, args.ckpt_epoch)

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
        inference_mode=False,  # allow autograd for adversarial generation
    )

    out_root = Path(cfg.OUTPUT_DIR) / "attacks" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_name": args.run_name,
        "ckpt_epoch": args.ckpt_epoch,
        "ckpt_path": str(ckpt_path),
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
        print(f"\nResolved checkpoint: {ckpt_path}\n")

    # CLEAN
    if trainer.is_global_zero:
        print("Running CLEAN eval...")
    clean_module = AttackEvalModule(ckpt_path=str(ckpt_path), mode="clean")
    clean_metrics = run_test(trainer, dm, clean_module)
    results["clean"] = clean_metrics

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FGSM sweep
    fgsm_eps_list = parse_list_of_floats(args.fgsm_eps)
    if trainer.is_global_zero:
        print("\nRunning FGSM sweep...")
    for eps in tqdm(fgsm_eps_list, disable=not trainer.is_global_zero, desc="FGSM eps"):
        ex_dir = None
        if args.save_examples and eps > 0:
            ex_dir = str(out_root / "examples" / f"fgsm_eps{eps}")

        m = AttackEvalModule(
            ckpt_path=str(ckpt_path),
            mode="fgsm",
            fgsm_cfg=FGSMConfig(eps=eps),
            save_examples_dir=ex_dir,
            save_n_batches=args.save_n_batches if ex_dir else 0,
        )
        metrics = run_test(trainer, dm, m)
        row = {"eps": eps, **metrics}
        results["fgsm"].append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # PGD sweep (eps x steps)
    pgd_eps_list = parse_list_of_floats(args.pgd_eps)
    pgd_steps_list = parse_list_of_ints(args.pgd_steps_list)

    if trainer.is_global_zero:
        print("\nRunning PGD sweep...")
    for steps in tqdm(pgd_steps_list, disable=not trainer.is_global_zero, desc="PGD steps"):
        for eps in tqdm(pgd_eps_list, disable=not trainer.is_global_zero, desc=f"PGD eps (steps={steps})", leave=False):
            pgd_cfg = PGDConfig(
                eps=eps,
                alpha=args.pgd_alpha,
                steps=steps,
                random_start=bool(args.pgd_random_start),
            )

            ex_dir = None
            if args.save_examples and eps > 0:
                ex_dir = str(out_root / "examples" / f"pgd_eps{eps}_steps{steps}_rs{int(args.pgd_random_start)}")

            m = AttackEvalModule(
                ckpt_path=str(ckpt_path),
                mode="pgd",
                pgd_cfg=pgd_cfg,
                save_examples_dir=ex_dir,
                save_n_batches=args.save_n_batches if ex_dir else 0,
            )
            metrics = run_test(trainer, dm, m)
            row = {"eps": eps, **asdict(pgd_cfg), **metrics}
            results["pgd"].append(row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # SAVE (rank0)
    json_path = out_root / f"attack_results_epoch{args.ckpt_epoch}.json"
    csv_path = out_root / f"attack_results_epoch{args.ckpt_epoch}.csv"

    if trainer.is_global_zero:
        json_path.write_text(json.dumps(results, indent=2))

        # Flatten CSV
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "attack", "eps", "steps", "alpha", "random_start",
                "test/acc", "attack/linf_mean", "attack/linf_max", "attack/l2_mean",
                "clean_test/acc", "ckpt_epoch", "ckpt_path"
            ])

            clean_acc = results["clean"].get("test/acc", "")
            # FGSM rows (no steps/alpha)
            for r in results["fgsm"]:
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
                    args.ckpt_epoch,
                    str(ckpt_path),
                ])

            # PGD rows
            for r in results["pgd"]:
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
                    args.ckpt_epoch,
                    str(ckpt_path),
                ])

        print("\n===== ATTACK EVAL SUMMARY (rank0) =====")
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV : {csv_path}")
        print("======================================\n")


if __name__ == "__main__":
    main()