from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class CLIPConsistencyExtractor:
    """
    CLIP 기반 전역/부분 일관성과 텍스트 프롬프트 점수 추출.
    반환 차원: 5(부분 코사인) + 3(텍스트 real/fake 로짓, fake-real 차) = 8
    """

    def __init__(self, device: Optional[torch.device] = None, model_name: str = "ViT-B/32", strict: bool = False) -> None:
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.available = True
        self.strict = strict
        try:
            import clip  # type: ignore

            self.clip = clip
            self.model, self.preprocess = clip.load(model_name, device=self.device, download=False)
            self.model.eval()
        except Exception as ex:  # noqa: BLE001
            if strict:
                raise
            warnings.warn(
                f"CLIP 로드 실패, 해당 피처는 0으로 채워집니다: {ex}. "
                "pip install git+https://github.com/openai/CLIP.git 후 재실행하거나 --disable-clip 사용"
            )
            self.available = False
            self.model = None
            self.preprocess = None
        self._text_tokens = None

    def _encode_text(self) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        if self._text_tokens is None:
            prompts = ["a real photo of a person", "an AI generated fake face"]
            self._text_tokens = self.clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(self._text_tokens)

    def _encode_images(self, images: List[Image.Image]) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        tensors = [self.preprocess(img).unsqueeze(0) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            embs = self.model.encode_image(batch)
        return embs

    def compute_batch(self, samples) -> np.ndarray:
        if not self.available or self.model is None or self.preprocess is None:
            return np.zeros((len(samples), 8), dtype=float)

        regions = ["full", "upper", "middle", "lower", "left", "right"]
        image_lists = {r: [] for r in regions}
        for s in samples:
            for r in regions:
                img = getattr(s.regions, r, None)
                if img is None:
                    raise RuntimeError(f"missing region '{r}' for sample {s.path}")
                image_lists[r].append(img)

        embs = {}
        for r in regions:
            embs[r] = self._encode_images(image_lists[r])
            embs[r] = embs[r] / embs[r].norm(dim=-1, keepdim=True)

        text_embs = self._encode_text()
        if text_embs is not None:
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        feats = []
        for i in range(len(samples)):
            full = embs["full"][i]
            sims = []
            for r in ["upper", "middle", "lower", "left", "right"]:
                sims.append(float((full * embs[r][i]).sum().item()))

            text_feats = [0.0, 0.0, 0.0]
            if text_embs is not None:
                logits = (full @ text_embs.T).softmax(dim=-1).cpu().numpy()  # [2]
                real_prob = float(logits[0])
                fake_prob = float(logits[1])
                text_feats = [real_prob, fake_prob, fake_prob - real_prob]

            feats.append(np.array(sims + text_feats, dtype=float))

        return np.vstack(feats)
