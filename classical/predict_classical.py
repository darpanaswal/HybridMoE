import cv2
import joblib
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict
from handcrafted.features.feature_extractor import FeatureExtractorConfig, extract_handcrafted_features


def bgr_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.uint8)


def predict_one(clf: Any, x: np.ndarray) -> Dict[str, object]:
    pred = int(clf.predict(x[None, :])[0])
    out: Dict[str, object] = {"pred": pred}
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x[None, :])[0].astype(np.float32)
        out["proba"] = proba
        out["conf"] = float(proba.max())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Predict with classical model on a single image")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--image_path", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clf = joblib.load(args.model_path)

    bgr = bgr_from_path(args.image_path)
    feat_cfg = FeatureExtractorConfig()
    feat = extract_handcrafted_features(bgr, feat_cfg)["x"]

    out = predict_one(clf, feat)
    print(out)


if __name__ == "__main__":
    main()