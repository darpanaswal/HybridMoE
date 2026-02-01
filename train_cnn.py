import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from utils import config as cfg
from utils.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger
from data.gtsrb_datamodule import GTSRBConfig, GTSRBDataModule
from models.resnet50_pl import ResNet50Config, ResNet50Classifier
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train ResNet-50 on GTSRB (Lightning, offline pretrained weights)")

    p.add_argument("--data_dir", type=str, default=str(cfg.DATA_PATH / "data_raw"))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)  # 1 CPU => use 0
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_split", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=30)  # early stopping will stop earlier
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument(
        "--pretrained_weights_path",
        type=str,
        default=str(cfg.BASE_DIR / "models" / "resnet50_imagenet1k_v2.pth"),
        help="Local path to pretrained weights state_dict. Required if --pretrained.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", type=str, default="gtsrb-robustness")
    p.add_argument("--run_name", type=str, default="resnet50_baseline")

    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", type=str, default="16-mixed")

    # Early stopping
    p.add_argument("--early_stop_monitor", type=str, default="val/acc")
    p.add_argument("--early_stop_mode", type=str, choices=["min", "max"], default="max")
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0005)

    # PL sanity check
    p.add_argument("--sanity_val_steps", type=int, default=2)

    # Performance/logging
    p.add_argument("--matmul_precision", type=str, choices=["highest", "high", "medium"], default="high")
    p.add_argument("--log_every_n_steps", type=int, default=10)

    return p.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    torch.set_float32_matmul_precision(args.matmul_precision)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm_cfg = GTSRBConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_split=args.val_split,
    )
    dm = GTSRBDataModule(dm_cfg)

    model_cfg = ResNet50Config(
        num_classes=43,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        pretrained_weights_path=args.pretrained_weights_path if args.pretrained else None,
    )
    model = ResNet50Classifier(model_cfg)

    # W&B login (no internet on compute nodes might break; if your cluster blocks it, disable logger)
    import wandb
    wandb.login(key=cfg.wandb_token)
    logger = WandbLogger(project=args.project, name=args.run_name, log_model=False)

    ckpt_dir = out_dir / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:02d}-valacc{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
    )

    callbacks = [
        ckpt_cb,
        EarlyStopping(
            monitor=args.early_stop_monitor,
            mode=args.early_stop_mode,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        num_sanity_val_steps=args.sanity_val_steps,
    )

    trainer.fit(model, datamodule=dm)

    # ---- TEST LOOP (best checkpoint) ----
    test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
    # Lightning returns a list of dicts (one per dataloader). We have one.
    test_metrics = test_results[0] if test_results else {}

    best_path = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    best_score_val = float(best_score.cpu().item()) if best_score is not None else None

    summary = {
        "run_name": args.run_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "best_checkpoint_path": best_path,
        "best_val_metric_name": ckpt_cb.monitor,
        "best_val_metric_score": best_score_val,
        "test_metrics": test_metrics,
        "config": {
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "img_size": args.img_size,
            "val_split": args.val_split,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pretrained": args.pretrained,
            "pretrained_weights_path": args.pretrained_weights_path if args.pretrained else None,
            "devices": args.devices,
            "accelerator": args.accelerator,
            "precision": args.precision,
            "seed": args.seed,
            "early_stop_monitor": args.early_stop_monitor,
            "early_stop_mode": args.early_stop_mode,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "matmul_precision": args.matmul_precision,
        },
    }

    metrics_path = out_dir / "metrics" / args.run_name / "test_metrics.json"
    _save_json(metrics_path, summary)

    print("\n===== FINAL SUMMARY =====")
    print(f"Best checkpoint: {best_path}")
    print(f"Best {ckpt_cb.monitor}: {best_score_val}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print(f"Saved: {metrics_path}")
    print("=========================\n")


if __name__ == "__main__":
    main()