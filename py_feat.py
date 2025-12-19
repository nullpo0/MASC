from __future__ import annotations

import contextlib
import io
import os
import tempfile
from typing import Dict, List

import numpy as np
from PIL import Image

# scipy 1.13에서 simps -> simpson 변경 대응
try:
    import scipy.integrate as _scipy_integrate

    if not hasattr(_scipy_integrate, "simps") and hasattr(_scipy_integrate, "simpson"):
        _scipy_integrate.simps = _scipy_integrate.simpson  # type: ignore[attr-defined]
except Exception:
    pass

from feat import Detector

UPPER_AUS = [1, 2, 4, 5, 6, 7, 43]
LOWER_AUS = [9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28]


class PyFeatFeatureExtractor:
    """
    py-feat AU 추출 후 상/하/감정 기반 hand-crafted feature 생성.
    기본 15차원 + 감정 그룹 6차원 = 21차원.
    """

    def __init__(self, batch_size: int = 16, device: str = "cuda") -> None:
        self.detector = Detector(device=device)
        self.batch_size = batch_size

    def _extract_aus(self, pil_imgs: List[Image.Image]) -> List[Dict[str, float]]:
        tmp_files: List[str] = []
        for img in pil_imgs:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp_files.append(tmp.name)
            tmp.close()

        lookups: List[Dict[str, float]] = []
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fex = self.detector.detect_image(tmp_files, verbose=False, batch_size=self.batch_size)
            if len(fex) == 0:
                raise RuntimeError("py-feat가 얼굴을 찾지 못했습니다.")
            au_df = fex.aus() if callable(getattr(fex, "aus", None)) else fex.aus
            if au_df is None or len(au_df) == 0:
                raise RuntimeError("py-feat AU 추출 결과가 비어 있습니다.")

            for _, row in au_df.iterrows():
                lookup: Dict[str, float] = {}
                for lab, val in row.items():
                    digits = "".join(ch for ch in str(lab) if ch.isdigit())
                    if digits:
                        lookup[digits.lstrip("0") or "0"] = float(val)
                lookups.append(lookup)
        finally:
            for f in tmp_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
        return lookups

    @staticmethod
    def _get(lookup: Dict[str, float], key: int) -> float:
        return float(lookup.get(str(key), 0.0))

    def compute_batch(self, samples) -> np.ndarray:
        eps = 1e-6
        lookups = self._extract_aus([s.regions.full for s in samples])

        if len(lookups) > len(samples):
            lookups = lookups[: len(samples)]
        elif len(lookups) < len(samples):
            zero_lookup: Dict[str, float] = {}
            lookups.extend([zero_lookup] * (len(samples) - len(lookups)))

        feats_all = []
        for lookup in lookups:
            upper_vals = np.array([self._get(lookup, k) for k in UPPER_AUS], dtype=float)
            lower_vals = np.array([self._get(lookup, k) for k in LOWER_AUS], dtype=float)

            upper_mean = upper_vals.mean()
            lower_mean = lower_vals.mean()
            upper_energy = float(np.sqrt((upper_vals**2).sum()))
            lower_energy = float(np.sqrt((lower_vals**2).sum()))

            feats = {}
            feats["upper_mean_minus_lower_mean"] = upper_mean - lower_mean
            feats["upper_mean_div_lower_mean"] = upper_mean / (lower_mean + eps)
            feats["upper_energy_minus_lower_energy"] = upper_energy - lower_energy
            feats["upper_energy_div_lower_energy"] = upper_energy / (lower_energy + eps)

            au6 = self._get(lookup, 6)
            au12 = self._get(lookup, 12)
            feats["duchenne_diff_6_minus_12"] = au6 - au12
            feats["duchenne_ratio_6_div_12"] = au6 / (au12 + eps)
            feats["duchenne_sum_6_plus_12"] = au6 + au12

            upper_sad = self._get(lookup, 1) + self._get(lookup, 4)
            lower_sad = self._get(lookup, 15)
            feats["sad_diff_upper_minus_lower"] = upper_sad - lower_sad
            feats["sad_ratio_upper_div_lower"] = upper_sad / (lower_sad + eps)

            upper_surprise = self._get(lookup, 1) + self._get(lookup, 2) + self._get(lookup, 5)
            lower_surprise = self._get(lookup, 25) + self._get(lookup, 26)
            feats["surprise_diff_upper_minus_lower"] = upper_surprise - lower_surprise
            feats["surprise_ratio_upper_div_lower"] = upper_surprise / (lower_surprise + eps)

            au9 = self._get(lookup, 9)
            au10 = self._get(lookup, 10)
            feats["disgust_diff_9_minus_10"] = au9 - au10
            feats["disgust_ratio_9_div_10"] = au9 / (au10 + eps)
            feats["disgust_sum_9_plus_10"] = au9 + au10

            tension = self._get(lookup, 23) + self._get(lookup, 24)
            smile = au12
            feats["smile_tension_ratio_23_24_div_12"] = tension / (smile + eps)
            feats["smile_tension_diff_23_24_minus_12"] = tension - smile

            # 감정 그룹 스코어
            feats["emotion_happy"] = au6 + au12
            feats["emotion_surprise"] = upper_surprise + lower_surprise
            feats["emotion_sadness"] = self._get(lookup, 1) + self._get(lookup, 4) + self._get(lookup, 15)
            feats["emotion_anger"] = self._get(lookup, 4) + self._get(lookup, 5) + self._get(lookup, 7) + self._get(lookup, 23)
            feats["emotion_disgust"] = au9 + au10
            feats["emotion_fear"] = self._get(lookup, 1) + self._get(lookup, 2) + self._get(lookup, 4) + self._get(lookup, 5) + self._get(lookup, 7) + self._get(lookup, 20)

            feats_all.append(np.array(list(feats.values()), dtype=float))

        return np.stack(feats_all, axis=0)
