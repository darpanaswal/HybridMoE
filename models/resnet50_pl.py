import torch
import torch.nn as nn
from pathlib import Path
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional, Any
from dataclasses import dataclass
from torchvision.models import resnet50
from torchmetrics.classification import MulticlassAccuracy


@dataclass
class ResNet50Config:
    num_classes: int = 43  # GTSRB
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Offline-pretrained
    pretrained: bool = True
    pretrained_weights_path: Optional[str] = None  # path to .pth state_dict


class ResNet50Classifier(pl.LightningModule):
    """
    ResNet-50 classifier that can load pretrained weights from a LOCAL file.

    IMPORTANT:
    - For Lightning checkpoint loading, __init__ must be compatible with
      hyperparameters stored in the checkpoint. Therefore we accept either:
        (a) cfg=ResNet50Config(...)
        (b) keyword args matching ResNet50Config fields (Lightning passes these)
    """
    def __init__(self, cfg: Optional[ResNet50Config] = None, **kwargs: Any) -> None:
        super().__init__()

        if cfg is None:
            cfg = ResNet50Config(**kwargs)
        self.cfg = cfg

        # Save as flat hyperparameters so Lightning can restore easily
        self.save_hyperparameters(self.cfg.__dict__)

        if self.cfg.pretrained:
            if not self.cfg.pretrained_weights_path:
                raise ValueError(
                    "pretrained=True but pretrained_weights_path is not set. "
                    "Run python -m models.download_resnet first, then pass the path."
                )
            self.backbone = self._build_with_local_pretrained(
                weights_path=Path(self.cfg.pretrained_weights_path),
                num_classes=self.cfg.num_classes,
            )
        else:
            self.backbone = resnet50(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.cfg.num_classes)

        self.train_acc = MulticlassAccuracy(num_classes=self.cfg.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.cfg.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.cfg.num_classes)

    @staticmethod
    def _build_with_local_pretrained(weights_path: Path, num_classes: int) -> nn.Module:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found at: {weights_path}\n"
                f"Create them with: python -m models.download_resnet --output_path {weights_path}"
            )

        model = resnet50(weights=None)
        state = torch.load(weights_path, map_location="cpu")

        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            acc = self.train_acc(preds, y)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        elif stage == "val":
            acc = self.val_acc(preds, y)
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        elif stage == "test":
            acc = self.test_acc(preds, y)
            self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.trainer.max_epochs, 1))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}