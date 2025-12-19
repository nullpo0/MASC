from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASC classifier 학습 (MLP, CSV 입력)")
    parser.add_argument("--train-csv", type=Path, default=Path("train_features.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("valid_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("classifier.pt"))
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32], help="MLP hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=100, help="훈련 에폭 수")
    parser.add_argument("--max-iter", type=int, default=1, help="MLPClassifier max_iter (에폭당 1로 고정 권장)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--ablation",
        type=int,
        default=-1,
        help="제거할 모듈을 bitmask로 지정 (-1=전체 사용). bit0=fairface, bit1=facenet, bit2=pyfeat, bit3=clip. 예) 1: fairface 제거, 8: clip 제거, 10(=1010b): facenet+clip 제거, 15: 모두 제거",
    )
    parser.add_argument("--save-ablated", action="store_true", help="ablation 적용된 CSV를 추가로 저장")
    parser.add_argument("--ablated-train-csv", type=Path, default=Path("train_ablation.csv"))
    parser.add_argument("--ablated-val-csv", type=Path, default=Path("valid_ablation.csv"))
    return parser.parse_args()


def load_feature_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise SystemExit(f"'label' column missing in {path}")
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    if not feat_cols:
        raise SystemExit(f"no feature columns starting with 'f_' in {path}")
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    paths = df["path"].tolist() if "path" in df.columns else []
    return X, y, paths, df


def sanitize_features(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def apply_ablation(X: np.ndarray, code: int) -> np.ndarray:
    """
    피처 순서: fairface(5) + facenet(5) + pyfeat(21) + clip(5)
    code bitmask: bit0=fairface, bit1=facenet, bit2=pyfeat, bit3=clip (1이면 제거)
    """
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

    X_train, y_train, paths_train, df_train = load_feature_csv(args.train_csv)
    X_val, y_val, paths_val, df_val = load_feature_csv(args.val_csv)

    X_train = sanitize_features(X_train)
    X_val = sanitize_features(X_val)
    X_train = apply_ablation(X_train, args.ablation)
    X_val = apply_ablation(X_val, args.ablation)

    # ablation 적용된 CSV 저장 (옵션)
    if args.save_ablated:
        def save_ablation_csv(X: np.ndarray, y: np.ndarray, paths: List[str], out_path: Path) -> None:
            feat_cols = [f"f_{i}" for i in range(X.shape[1])]
            df_out = pd.DataFrame(X, columns=feat_cols)
            df_out.insert(0, "label", y.astype(int))
            if paths:
                df_out.insert(1, "path", paths)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_path, index=False)
            print(f"[info] saved ablated features to {out_path} (rows={len(df_out)})")

        save_ablation_csv(X_train, y_train, paths_train, args.ablated_train_csv)
        save_ablation_csv(X_val, y_val, paths_val, args.ablated_val_csv)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_sizes),
        max_iter=args.max_iter,
        random_state=args.random_state,
        warm_start=True,
        early_stopping=False,
        verbose=False,
    )

    classes = np.unique(y_train)
    clf.partial_fit(X_train, y_train, classes=classes)

    for epoch in range(1, args.epochs + 1):
        clf.partial_fit(X_train, y_train)
        train_prob = clf.predict_proba(X_train)[:, list(clf.classes_).index(1)]
        val_prob = clf.predict_proba(X_val)[:, list(clf.classes_).index(1)]
        train_loss = log_loss(y_train, train_prob, labels=[0, 1])
        val_loss = log_loss(y_val, val_prob, labels=[0, 1])
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"[val] accuracy={acc:.4f}")
    try:
        class_list = list(clf.classes_)
        fake_idx = class_list.index(1)
        val_prob = clf.predict_proba(X_val)[:, fake_idx]
        auc = roc_auc_score(y_val, val_prob)
        print(f"[val] roc_auc={auc:.4f}")
    except Exception as ex:  # noqa: BLE001
        print(f"[val] roc_auc 계산 실패: {ex}")

    print(classification_report(y_val, val_pred, target_names=["real", "fake"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scaler": scaler, "model": clf}, args.output)
    print(f"saved classifier to {args.output}")


if __name__ == "__main__":
    main()
