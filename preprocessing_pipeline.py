from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import contextlib
import io
import os
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import InceptionResnetV1

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class FaceCrops:
    full: Image.Image
    upper: Image.Image
    middle: Image.Image
    lower: Image.Image
    left: Image.Image
    right: Image.Image


@dataclass
class Sample:
    path: Path
    label: int  # 0=real, 1=fake
    crops: FaceCrops


# -----------------------------
# Cropper (stub; replace with 실제 crop/마스킹 로직)
# -----------------------------

class Cropper:
    def __init__(self, image_size: int = 224, mask_mode: str = "zero") -> None:
        self.image_size = image_size
        self.mask_mode = mask_mode
        import torch
        from facenet_pytorch import MTCNN

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=False, device=device)

    def _masked_region(self, face_img: Image.Image, box: Sequence[float]) -> Image.Image:
        from PIL import ImageDraw, ImageFilter

        if self.mask_mode == "blur":
            background = face_img.filter(ImageFilter.GaussianBlur(radius=8.0))
        else:
            background = Image.new("RGB", face_img.size, (0, 0, 0))

        mask = Image.new("L", face_img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(box, fill=255)
        return Image.composite(face_img, background, mask)

    def __call__(self, img: Image.Image) -> FaceCrops:
        """
        얼굴 검출 후 full/upper/middle/lower/left/right 생성.
        """
        import numpy as np

        boxes, _, landmarks = self.mtcnn.detect(img, landmarks=True)
        if boxes is None or len(boxes) == 0 or landmarks is None:
            raise RuntimeError("MTCNN에서 얼굴을 찾지 못했습니다.")

        x1, y1, x2, y2 = boxes[0]
        w_img, h_img = img.size
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w_img)
        y2 = min(int(y2), h_img)

        face = img.crop((x1, y1, x2, y2)).convert("RGB")
        face_resized = face.resize((self.image_size, self.image_size))

        lm = landmarks[0]  # [left_eye, right_eye, nose, mouth_left, mouth_right]
        scale_x = self.image_size / (x2 - x1)
        scale_y = self.image_size / (y2 - y1)
        lm_local = (lm - np.array([x1, y1])) * np.array([scale_x, scale_y])

        eye_pts = lm_local[:2]
        nose_pt = lm_local[2]
        mouth_pts = lm_local[3:]

        eye_min = eye_pts.min(axis=0)
        eye_max = eye_pts.max(axis=0)
        eye_span = np.linalg.norm(eye_pts[0] - eye_pts[1]) + 1e-6
        eye_pad = eye_span * 0.25
        upper_box = [
            eye_min[0] - eye_pad * 2,
            eye_min[1] - eye_pad * 4,
            eye_max[0] + eye_pad * 2,
            eye_max[1] + eye_pad,
        ]

        nose_w = eye_span * 0.6
        nose_h = eye_span * 0.6
        middle_box = [
            nose_pt[0] - nose_w * 2,
            nose_pt[1] - nose_h / 2,
            nose_pt[0] + nose_w * 2,
            nose_pt[1] + nose_h / 2,
        ]

        mouth_min = mouth_pts.min(axis=0)
        mouth_max = mouth_pts.max(axis=0)
        mouth_pad = eye_span * 0.25
        lower_box = [
            mouth_min[0] - mouth_pad,
            mouth_min[1] - mouth_pad,
            mouth_max[0] + mouth_pad,
            mouth_max[1] + mouth_pad * 2,
        ]

        upper_img = self._masked_region(face_resized, upper_box)
        middle_img = self._masked_region(face_resized, middle_box)
        lower_img = self._masked_region(face_resized, lower_box)

        w = face_resized.width
        h = face_resized.height
        left_box = (0, 0, w // 2, h)
        right_box = (w // 2, 0, w, h)
        left_img = self._masked_region(face_resized, left_box)
        right_img = self._masked_region(face_resized, right_box)

        return FaceCrops(
            full=face_resized,
            upper=upper_img,
            middle=middle_img,
            lower=lower_img,
            left=left_img,
            right=right_img,
        )


# -----------------------------
# Feature extractor 인터페이스 (stub)
# -----------------------------

class FeatureExtractor:
    name: str
    dim: int

    def compute(self, samples: Sequence[Sample]) -> np.ndarray:
        raise NotImplementedError


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


class FairFaceExtractor(FeatureExtractor):
    """
    full 대비 upper/middle/lower/left/right 확률 분포 JS divergence 5차원.
    """

    def __init__(self, weight_path: Path | str = "res34_fair_align_multi_7_20190809.pt", device: torch.device | None = None) -> None:
        self.name = "fairface"
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
        self.dim = 5

    def _to_prob_vector(self, logits: np.ndarray) -> np.ndarray:
        race = np.exp(logits[:7]) / np.sum(np.exp(logits[:7]))
        gender = np.exp(logits[7:9]) / np.sum(np.exp(logits[7:9]))
        age = np.exp(logits[9:18]) / np.sum(np.exp(logits[9:18]))
        return np.concatenate([race, gender, age])

    def compute(self, samples: Sequence[Sample]) -> np.ndarray:
        regions = ["full", "upper", "middle", "lower", "left", "right"]
        tensors: Dict[str, List[torch.Tensor]] = {r: [] for r in regions}
        for s in samples:
            for r in regions:
                tensors[r].append(self.trans(getattr(s.crops, r)))

        probs: Dict[str, List[np.ndarray]] = {r: [] for r in regions}
        with torch.no_grad():
            for r in regions:
                batch = torch.stack(tensors[r]).to(self.device)
                out = self.model(batch).cpu().numpy()  # [B, 18]
                for logits in out:
                    probs[r].append(self._to_prob_vector(logits))

        feats = []
        for i in range(len(samples)):
            full_vec = probs["full"][i]
            row = []
            for r in ["upper", "middle", "lower", "left", "right"]:
                row.append(_js_divergence(full_vec, probs[r][i]))
            feats.append(row)
        return np.array(feats, dtype=float)


class FaceNetExtractor(FeatureExtractor):
    """
    FaceNet 임베딩으로 full 대비 upper/middle/lower/left/right 코사인 유사도 5차원.
    """

    def __init__(self, device: torch.device | None = None, pretrained: str | None = "vggface2") -> None:
        self.name = "facenet"
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        self.trans = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.dim = 5

    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return float((a * b).sum().item())

    def compute(self, samples: Sequence[Sample]) -> np.ndarray:
        regions = ["full", "upper", "middle", "lower", "left", "right"]
        tensors: Dict[str, List[torch.Tensor]] = {r: [] for r in regions}
        for s in samples:
            for r in regions:
                tensors[r].append(self.trans(getattr(s.crops, r)))

        embs: Dict[str, torch.Tensor] = {}
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


UPPER_AUS = [1, 2, 4, 5, 6, 7, 43]
LOWER_AUS = [9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28]

def _ensure_simps() -> None:
    """
    scipy 1.13 이후 simps 제거 대응.
    feat import 전에 호출해서 simps alias를 보장.
    """
    try:
        import scipy.integrate as _scipy_integrate  # type: ignore

        if not hasattr(_scipy_integrate, "simps") and hasattr(_scipy_integrate, "simpson"):
            _scipy_integrate.simps = _scipy_integrate.simpson  # type: ignore[attr-defined]
    except Exception:
        pass


class PyFeatExtractor(FeatureExtractor):
    """
    py-feat AU 기반 hand-crafted feature (21차원).
    """

    def __init__(self, batch_size: int = 16, device: str = "cuda") -> None:
        self.name = "pyfeat"
        _ensure_simps()
        from feat import Detector  # type: ignore

        self.detector = Detector(device=device)
        self.batch_size = batch_size
        self.dim = 21

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

    def compute(self, samples: Sequence[Sample]) -> np.ndarray:
        eps = 1e-6
        lookups = self._extract_aus([s.crops.full for s in samples])

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

            # 감정 그룹
            feats["emotion_happy"] = au6 + au12
            feats["emotion_surprise"] = upper_surprise + lower_surprise
            feats["emotion_sadness"] = self._get(lookup, 1) + self._get(lookup, 4) + self._get(lookup, 15)
            feats["emotion_anger"] = self._get(lookup, 4) + self._get(lookup, 5) + self._get(lookup, 7) + self._get(lookup, 23)
            feats["emotion_disgust"] = au9 + au10
            feats["emotion_fear"] = self._get(lookup, 1) + self._get(lookup, 2) + self._get(lookup, 4) + self._get(lookup, 5) + self._get(lookup, 7) + self._get(lookup, 20)

            feats_all.append(np.array(list(feats.values()), dtype=float))

        return np.stack(feats_all, axis=0)


class CLIPExtractor(FeatureExtractor):
    """
    CLIP 이미지 임베딩으로 full 대비 upper/middle/lower/left/right 코사인 5차원.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        model_name: str = "ViT-B/32",
        strict: bool = False,
    ) -> None:
        self.name = "clip"
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim = 5
        self.available = True
        import clip  # type: ignore

        self.clip = clip
        # 일부 버전은 download 인자를 지원하지 않음
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # type: ignore[union-attr]

    def _encode(self, imgs: List[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess(img).unsqueeze(0) for img in imgs]
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            embs = self.model.encode_image(batch)  # type: ignore[union-attr]
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs

    def compute(self, samples: Sequence[Sample]) -> np.ndarray:
        regions = ["full", "upper", "middle", "lower", "left", "right"]
        image_lists: Dict[str, List[Image.Image]] = {r: [] for r in regions}
        for s in samples:
            for r in regions:
                image_lists[r].append(getattr(s.crops, r))

        embs: Dict[str, torch.Tensor] = {}
        for r in regions:
            embs[r] = self._encode(image_lists[r])

        feats = []
        for i in range(len(samples)):
            full = embs["full"][i]
            sims = []
            for r in ["upper", "middle", "lower", "left", "right"]:
                sims.append(float((full * embs[r][i]).sum().item()))
            feats.append(sims)
        return np.array(feats, dtype=float)


# -----------------------------
# 파이프라인 유틸
# -----------------------------

def list_image_paths(root: Path, exts: Tuple[str, ...]) -> List[Tuple[Path, int]]:
    paths: List[Tuple[Path, int]] = []
    label_map = {"real": 0, "fake": 1}
    for cls_name, label in label_map.items():
        cls_root = root / cls_name
        if not cls_root.exists():
            continue
        for p in cls_root.rglob("*"):
            if p.suffix.lower() in exts and p.is_file():
                paths.append((p, label))
    return sorted(paths, key=lambda x: str(x[0]))


def iter_batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def save_features_csv(X: np.ndarray, y: np.ndarray, paths: Sequence[Path], out_path: Path) -> None:
    feature_cols = [f"f_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df.insert(0, "label", y.astype(int))
    df.insert(1, "path", [str(p) for p in paths])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[info] saved {len(df)} rows to {out_path}")


# -----------------------------
# 메인 파이프라인
# -----------------------------

def run_split(
    split_root: Path,
    batch_size: int,
    cropper: Cropper,
    extractors: List[FeatureExtractor],
    exts: Tuple[str, ...],
    out_csv: Path,
) -> None:
    path_labels = list_image_paths(split_root, exts)
    if not path_labels:
        print(f"[warn] no images under {split_root}, skip")
        return

    features: List[np.ndarray] = []
    labels: List[int] = []
    kept_paths: List[Path] = []

    iterator = iter_batches(path_labels, batch_size)
    if tqdm:
        iterator = tqdm(list(iterator), desc=f"{split_root.name} batches")

    for batch in iterator:
        samples: List[Sample] = []
        for path, label in batch:
            try:
                img = Image.open(path).convert("RGB")
                crops = cropper(img)
                samples.append(Sample(path=path, label=label, crops=crops))
            except Exception as ex:  # noqa: BLE001
                print(f"[skip] {path}: {ex}")
                continue

        if not samples:
            continue

        feats_each: List[np.ndarray] = []
        for ext in extractors:
            feats_each.append(ext.compute(samples))

        # truncate to min length to avoid mismatch
        min_len = min(f.shape[0] for f in feats_each)
        feats_each = [f[:min_len] for f in feats_each]
        concat = np.concatenate(feats_each, axis=1)
        features.append(concat)
        labels.extend([s.label for s in samples[:min_len]])
        kept_paths.extend([s.path for s in samples[:min_len]])

    if not features:
        print(f"[warn] {split_root}: no features generated")
        return

    X = np.concatenate(features, axis=0)
    y = np.array(labels, dtype=int)
    save_features_csv(X, y, kept_paths, out_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing pipeline (crop -> fairface/facenet/pyfeat/clip -> CSV)")
    parser.add_argument("--data-root", type=Path, default=Path("CelebDF"), help="root containing train/valid/test")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="output directory for CSVs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None, help="torch device 문자열 (e.g., cuda:0, cpu)")
    parser.add_argument("--fairface-weights", type=Path, default=Path("res34_fair_align_multi_7_20190809.pt"))
    # enable/disable modules
    parser.add_argument("--disable-fairface", action="store_true")
    parser.add_argument("--disable-facenet", action="store_true")
    parser.add_argument("--disable-pyfeat", action="store_true")
    parser.add_argument("--disable-clip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exts = (".jpg", ".jpeg", ".png", ".bmp")

    cropper = Cropper(image_size=args.image_size, mask_mode="zero")
    device = torch.device(args.device) if args.device else None

    extractors: List[FeatureExtractor] = []
    if not args.disable_fairface:
        extractors.append(FairFaceExtractor(weight_path=args.fairface_weights, device=device))
    if not args.disable_facenet:
        extractors.append(FaceNetExtractor(device=device))
    if not args.disable_pyfeat:
        extractors.append(PyFeatExtractor(device=device))
    if not args.disable_clip:
        extractors.append(CLIPExtractor(device=device))

    if not extractors:
        raise SystemExit("No feature extractors enabled.")

    for split in ["train", "valid", "test"]:
        split_root = args.data_root / split
        out_csv = args.out_dir / f"{split}_features.csv"
        run_split(split_root, args.batch_size, cropper, extractors, exts, out_csv)

    print("[done] preprocessing pipeline finished")


if __name__ == "__main__":
    main()
