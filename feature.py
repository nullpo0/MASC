from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from preprocessing import Preprocessor, ProcessedSample
from fairface import FairFaceFeatureExtractor
from facenet import FaceNetFeatureExtractor
from py_feat import PyFeatFeatureExtractor
from clip_features import CLIPConsistencyExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASC feature extractor -> CSV")
    parser.add_argument("--data-root", type=Path, required=True, help="real/fake 하위 폴더를 가진 루트 경로")
    parser.add_argument("--output", type=Path, default=Path("features.csv"))
    parser.add_argument("--fairface-weights", type=Path, default=Path("res34_fair_align_multi_7_20190809.pt"))
    parser.add_argument("--mask-mode", choices=["blur", "zero"], default="zero")
    parser.add_argument("--device", default=None, help="torch device 문자열(e.g., cuda:0, cpu)")
    parser.add_argument("--preprocess-batch-size", type=int, default=32)
    parser.add_argument("--pyfeat-batch-size", type=int, default=16)
    parser.add_argument("--disable-clip", action="store_true", help="CLIP 피처 비활성화")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 모델 이름")
    parser.add_argument("--clip-strict", action="store_true", help="CLIP 로드 실패 시 종료")
    return parser.parse_args()


def load_samples(pre: Preprocessor, paths: List[Path]) -> List[ProcessedSample]:
    samples: List[ProcessedSample] = []
    for p in paths:
        try:
            samples.append(pre.load_image(p))
        except Exception as ex:  # noqa: BLE001
            print(f"[skip] {p}: {ex}")
    return samples


def sanitize_features(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def save_features_csv(X: np.ndarray, y: np.ndarray, paths: Sequence[Path], out_path: Path) -> None:
    feature_cols = [f"f_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df.insert(0, "label", y.astype(int))
    df.insert(1, "path", [str(p) for p in paths])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[info] saved features to {out_path} (rows={len(df)})")


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else None

    pre = Preprocessor(mask_mode=args.mask_mode, device=device)
    fairface_extractor = FairFaceFeatureExtractor(str(args.fairface_weights), device=device)
    facenet_extractor = FaceNetFeatureExtractor(device=device)
    pyfeat_extractor = PyFeatFeatureExtractor(
        batch_size=args.pyfeat_batch_size,
        device="cuda" if device is None else str(device),
    )
    clip_extractor = None
    if not args.disable_clip:
        try:
            clip_extractor = CLIPConsistencyExtractor(device=device, model_name=args.clip_model, strict=args.clip_strict)
        except Exception as ex:  # noqa: BLE001
            if args.clip_strict:
                raise
            print(f"[warn] CLIP 로드 실패, 피처를 0으로 대체합니다: {ex}")
            clip_extractor = None

    paths = sorted(Preprocessor.iter_image_paths(args.data_root))
    if not paths:
        raise SystemExit(f"이미지가 없습니다: {args.data_root}")

    features: List[np.ndarray] = []
    labels: List[int] = []
    kept_paths: List[Path] = []
    for chunk in pre.iter_batches(paths, args.preprocess_batch_size):
        samples = load_samples(pre, chunk)
        if not samples:
            continue
        try:
            feats = pre.build_feature_matrix(
                samples,
                fairface_extractor,
                facenet_extractor,
                pyfeat_extractor,
                geometry_extractor=None,
                color_extractor=None,
                clip_extractor=clip_extractor,
                freq_extractor=None,
            )
        except Exception as ex:  # noqa: BLE001
            print(f"[error] feature build failed. skipping batch ({len(samples)} imgs). reason: {ex}")
            for s in samples:
                print(f"  - {s.path}")
            continue

        features.append(feats)
        labels.extend([s.label for s in samples])
        kept_paths.extend([s.path for s in samples])

    if not features:
        raise SystemExit("feature가 비었습니다. 전처리/추출 실패 여부를 확인하세요.")

    X = sanitize_features(np.concatenate(features, axis=0))
    y = np.array(labels, dtype=int)
    save_features_csv(X, y, kept_paths, args.output)
    print("[done] feature CSV 생성 완료")


if __name__ == "__main__":
    main()
