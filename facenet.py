from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


class FaceNetFeatureExtractor:
    """
    FaceNet 임베딩으로 full vs 부분/좌우 영역 유사도 계산.
    출력: [B, 5] (full-upper, full-middle, full-lower, full-left, full-right cosine similarity)
    """

    def __init__(self, device: torch.device | None = None, pretrained: str | None = "vggface2") -> None:
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        self.trans = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return float((a * b).sum().item())

    def compute_batch(self, samples) -> np.ndarray:
        regions = ["full", "upper", "middle", "lower", "left", "right"]
        tensors = {r: [] for r in regions}
        for s in samples:
            for r in regions:
                img = getattr(s.regions, r, None)
                if img is None:
                    raise RuntimeError(f"missing region '{r}' for sample {s.path}")
                tensors[r].append(self.trans(img))

        embs = {}
        with torch.no_grad():
            for r in regions:
                batch = torch.stack(tensors[r]).to(self.device)
                embs[r] = self.model(batch)

        feats = []
        for i in range(len(samples)):
            full = embs["full"][i]
            feats.append(
                [
                    self._cos(full, embs["upper"][i]),
                    self._cos(full, embs["middle"][i]),
                    self._cos(full, embs["lower"][i]),
                    self._cos(full, embs["left"][i]),
                    self._cos(full, embs["right"][i]),
                ]
            )
        return np.array(feats, dtype=float)
