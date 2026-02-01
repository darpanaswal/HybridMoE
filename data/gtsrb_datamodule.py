import torch
from pathlib import Path
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import Optional, Tuple
from torchvision.datasets import GTSRB
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split


@dataclass
class GTSRBConfig:
    data_dir: str
    batch_size: int = 256
    num_workers: int = 8
    img_size: int = 224
    val_split: float = 0.1
    pin_memory: bool = True


class GTSRBDataModule(pl.LightningDataModule):
    """
    Offline-safe: expects dataset is pre-downloaded to data_dir.
    """
    def __init__(self, cfg: GTSRBConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_tfms, self.eval_tfms = self._build_transforms(cfg.img_size)

    @staticmethod
    def _build_transforms(img_size: int) -> Tuple[T.Compose, T.Compose]:
        train_tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.1),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        eval_tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return train_tfms, eval_tfms

    def prepare_data(self) -> None:
        """
        No downloads here. Just validate.
        """
        root = Path(self.cfg.data_dir)
        sentinel = root / ".gtsrb_download_complete"
        if not sentinel.exists():
            raise FileNotFoundError(
                f"GTSRB not found / not marked complete in: {root}\n"
                f"Run: python -m data.download_gtsrb --data_dir {root}"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        full_train = GTSRB(root=self.cfg.data_dir, split="train", download=False, transform=self.train_tfms)

        val_size = int(len(full_train) * self.cfg.val_split)
        train_size = len(full_train) - val_size

        self.train_ds, self.val_ds = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # val should not use train augmentations; create separate dataset
        val_base = GTSRB(root=self.cfg.data_dir, split="train", download=False, transform=self.eval_tfms)
        self.val_ds = torch.utils.data.Subset(val_base, indices=self.val_ds.indices)

        self.test_ds = GTSRB(root=self.cfg.data_dir, split="test", download=False, transform=self.eval_tfms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )