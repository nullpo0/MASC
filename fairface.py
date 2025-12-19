from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


class FairFaceFeatureExtractor:
    """
    ResNet-34 기반 FairFace attribute classifier.
    - 입력: ProcessedSample 리스트
    - 출력: JS divergence 기반 feature (6차원)
    """

    def __init__(self, weight_path: str, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet34(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._pair_keys = [
            ("full", "upper"),
            ("full", "middle"),
            ("full", "lower"),
            ("upper", "middle"),
            ("upper", "lower"),
            ("middle", "lower"),
        ]

    def _to_prob_vector(self, logits: np.ndarray) -> np.ndarray:
        race = np.exp(logits[:7]) / np.sum(np.exp(logits[:7]))
        gender = np.exp(logits[7:9]) / np.sum(np.exp(logits[7:9]))
        age = np.exp(logits[9:18]) / np.sum(np.exp(logits[9:18]))
        return np.concatenate([race, gender, age])

    def compute_batch(self, samples) -> np.ndarray:
        regions = ["full", "upper", "middle", "lower"]
        tensors = {r: [] for r in regions}
        for s in samples:
            for r in regions:
                tensors[r].append(self.trans(getattr(s.regions, r)))

        probs: Dict[str, List[np.ndarray]] = {r: [] for r in regions}
        with torch.no_grad():
            for r in regions:
                batch = torch.stack(tensors[r]).to(self.device)
                out = self.model(batch).cpu().numpy()  # [B, 18]
                for logits in out:
                    probs[r].append(self._to_prob_vector(logits))

        feats = []
        for i in range(len(samples)):
            row = []
            for a, b in self._pair_keys:
                row.append(_js_divergence(probs[a][i], probs[b][i]))
            feats.append(row)
        return np.array(feats, dtype=float)
