import json
import torch
import argparse
import torchvision
from pathlib import Path
from utils.config import MODEL_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Download ResNet-50 pretrained weights for offline SLURM jobs")
    p.add_argument(
        "--output_path",
        type=str,
        default=str(MODEL_PATH / "resnet50_imagenet1k_v2.pth"),
        help="Where to save the pretrained state_dict",
    )
    p.add_argument(
        "--meta_path",
        type=str,
        default=str(MODEL_PATH / "resnet50_imagenet1k_v2.meta.json"),
        help="Where to save a small JSON metadata file",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_path)
    meta_path = Path(args.meta_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists. Use --force to overwrite.")
    if meta_path.exists() and not args.force:
        raise FileExistsError(f"{meta_path} already exists. Use --force to overwrite.")

    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2  # explicit
    model = resnet50(weights=weights)
    model.eval()

    # Save only the state_dict for portability
    torch.save(model.state_dict(), out_path)

    meta = {
        "arch": "resnet50",
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "transforms": {
            "mean": list(weights.transforms().mean),
            "std": list(weights.transforms().std),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved weights to: {out_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()