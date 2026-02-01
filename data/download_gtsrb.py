import argparse
from pathlib import Path
from utils.config import DATA_PATH
from torchvision.datasets import GTSRB

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Download GTSRB dataset for offline SLURM training")
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_PATH / "data_raw"),
        help="Directory where torchvision will store GTSRB",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GTSRB to: {data_dir}")

    # Download train/test splits
    GTSRB(root=str(data_dir), split="train", download=True)
    GTSRB(root=str(data_dir), split="test", download=True)

    # Sentinel
    (data_dir / ".gtsrb_download_complete").write_text("ok\n")
    print("Done. Wrote sentinel: .gtsrb_download_complete")


if __name__ == "__main__":
    main()