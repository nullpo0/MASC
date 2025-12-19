from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASC classifier 추론 (CSV 입력)")
    parser.add_argument("--input-csv", type=Path, default=Path("test_features.csv"))
    parser.add_argument("--classifier", type=Path, default=Path("classifier.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("predictions.csv"))
    parser.add_argument(
        "--ablation",
        type=int,
        default=-1,
        help="제거할 모듈 bitmask: bit0=fairface, bit1=facenet, bit2=pyfeat, bit3=clip (-1=전체 사용)",
    )
    return parser.parse_args()


def load_feature_csv(path: Path) -> Tuple[np.ndarray, np.ndarray | None, List[str] | None]:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    if not feat_cols:
        raise SystemExit(f"no feature columns starting with 'f_' in {path}")
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None
    paths = df["path"].tolist() if "path" in df.columns else None
    return X, y, paths


def sanitize_features(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def apply_ablation(X: np.ndarray, code: int) -> np.ndarray:
    # 피처 순서: fairface(5) + facenet(5) + pyfeat(21) + clip(5)
    dims = {"fairface": 5, "facenet": 5, "pyfeat": 21, "clip": 5}
    order = ["fairface", "facenet", "pyfeat", "clip"]
    start = 0
    slices = {}
    for name in order:
        end = start + dims[name]
        slices[name] = (start, end)
        start = end

    if code == -1:
        return X
    else:
        removes = set()
        for i, name in enumerate(order):
            if code & (1 << i):
                removes.add(name)

    keep_idx = []
    for name in order:
        if name in removes:
            continue
        s, e = slices[name]
        keep_idx.extend(range(s, e))
    if not keep_idx:
        raise SystemExit("모든 피처가 제거되었습니다. ablation 코드를 확인하세요.")
    return X[:, keep_idx]


def main() -> None:
    args = parse_args()

    try:
        ckpt = torch.load(args.classifier, map_location="cpu")
    except Exception:
        ckpt = torch.load(args.classifier, map_location="cpu", weights_only=False)
    clf = ckpt["model"]
    scaler = ckpt.get("scaler", None)

    X, y, paths = load_feature_csv(args.input_csv)
    X = sanitize_features(X)
    X = apply_ablation(X, args.ablation)
    if scaler is not None:
        X = scaler.transform(X)

    proba = clf.predict_proba(X)
    fake_idx = list(clf.classes_).index(1)
    fake_prob = proba[:, fake_idx]
    preds = (fake_prob >= 0.5).astype(int)

    out_df = pd.DataFrame(
        {
            "path": paths if paths is not None else ["" for _ in range(len(preds))],
            "pred": preds,
            "fake_prob": fake_prob,
        }
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(out_df.head())
    print(f"[info] saved predictions to {args.output_csv}")

    if y is not None and len(y) == len(preds):
        tp = int(((preds == 1) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        print(f"[info] correct_fake(tp)={tp}, correct_real(tn)={tn}, total={len(preds)}")
        print(f"[info] fp={fp}, fn={fn}")

        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, fake_prob)
            print(f"[test] acc={acc:.4f}, auc={auc:.4f}")
        except Exception as ex:  # noqa: BLE001
            print(f"[test] AUC 계산 실패: {ex}")
        print(classification_report(y, preds, target_names=["real", "fake"]))
    else:
        n_fake = int((preds == 1).sum())
        n_real = int((preds == 0).sum())
        print(f"[info] predicted fake={n_fake}, real={n_real}, total={len(preds)}")


if __name__ == "__main__":
    main()
