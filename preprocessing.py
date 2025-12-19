from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN


MASK_MODES = {"blur", "zero"}


@dataclass
class FaceRegions:
    full: Image.Image
    upper: Image.Image
    middle: Image.Image
    lower: Image.Image
    left: Image.Image
    right: Image.Image
    landmarks: np.ndarray  # (5, 2) in resized face coordinates


@dataclass
class ProcessedSample:
    path: Path
    label: int
    regions: FaceRegions


class Preprocessor:
    """
    얼굴 탐지 후 resize, 상/중/하/좌/우 마스킹된 패치 및 랜드마크 생성.
    """

    def __init__(
        self,
        mask_mode: str = "zero",
        image_size: int = 224,
        device: Optional[torch.device] = None,
    ) -> None:
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {sorted(MASK_MODES)}")
        self.mask_mode = mask_mode
        self.image_size = image_size
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

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

    def _detect_regions(self, image: Image.Image) -> FaceRegions:
        boxes, _, landmarks = self.mtcnn.detect(image, landmarks=True)
        if boxes is None or len(boxes) == 0 or landmarks is None:
            raise RuntimeError("MTCNN에서 얼굴을 찾지 못했습니다.")

        x1, y1, x2, y2 = boxes[0]
        w_img, h_img = image.size
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w_img)
        y2 = min(int(y2), h_img)

        face = image.crop((x1, y1, x2, y2)).convert("RGB")
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

        # 좌/우 절반
        w = face_resized.width
        h = face_resized.height
        left_img = face_resized.crop((0, 0, w // 2, h))
        right_img = face_resized.crop((w // 2, 0, w, h))

        return FaceRegions(
            full=face_resized,
            upper=upper_img,
            middle=middle_img,
            lower=lower_img,
            left=left_img,
            right=right_img,
            landmarks=lm_local.astype(np.float32),
        )

    def load_image(self, path: Path, label_map: Optional[Dict[str, int]] = None) -> ProcessedSample:
        label_map = label_map or {"real": 0, "fake": 1}
        img = Image.open(path).convert("RGB")
        regions = self._detect_regions(img)

        label = None
        for part in reversed(path.parts):
            if part in label_map:
                label = label_map[part]
                break
        if label is None:
            raise ValueError(f"레이블을 추론할 수 없습니다: {path}")

        return ProcessedSample(path=path, label=label, regions=regions)

    @staticmethod
    def iter_image_paths(dataset_root: Path, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
        return [p for p in dataset_root.rglob("*") if p.suffix.lower() in exts]

    @staticmethod
    def iter_batches(items: Sequence, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]
