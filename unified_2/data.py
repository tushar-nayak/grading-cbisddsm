from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, filters, measure, morphology


@dataclass
class ManifestSample:
    patient_id: str
    sample_id: str
    breast_side: str
    birads_label: int
    cc_image_path: str
    mlo_image_path: str


@dataclass
class ViewData:
    image_path: str
    raw_image: np.ndarray
    preprocessed_image: np.ndarray
    breast_mask: np.ndarray


class ManifestLoader:
    def __init__(self, manifest_csv: str):
        self.manifest_csv = Path(manifest_csv)
        self.frame = pd.read_csv(self.manifest_csv)

    def load_samples(self, limit: int | None = None) -> List[ManifestSample]:
        rows = self.frame.iloc[:limit] if limit else self.frame
        samples = []
        for idx, row in rows.iterrows():
            cc_stem = Path(str(row["cc_image_path"])).stem
            patient_id = str(row.get("patient_id", "")).strip() or f"case_{idx:05d}"
            sample_id = str(row.get("sample_id", "")).strip() or f"{patient_id}_{str(row['breast_side']).strip().upper()}_{cc_stem}"
            samples.append(
                ManifestSample(
                    patient_id=patient_id,
                    sample_id=sample_id,
                    breast_side=str(row["breast_side"]),
                    birads_label=int(row["birads_label"]),
                    cc_image_path=str(row["cc_image_path"]),
                    mlo_image_path=str(row["mlo_image_path"]),
                )
            )
        return samples


def load_grayscale_image(image_path: str, flip_right: bool = False) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if flip_right:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(image, dtype=np.float32)
    if arr.max() > 0:
        arr /= 255.0
    return arr


def preprocess_mammogram(image: np.ndarray) -> np.ndarray:
    enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
    blurred = filters.gaussian(enhanced, sigma=1.0, preserve_range=True)
    blurred = blurred.astype(np.float32)
    if blurred.max() > blurred.min():
        blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min())
    return blurred.astype(np.float32)


def extract_breast_mask(image: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    binary = image > threshold
    binary = morphology.binary_opening(binary, morphology.disk(5))
    binary = morphology.binary_closing(binary, morphology.disk(9))
    labels = measure.label(binary)
    if labels.max() == 0:
        return binary.astype(np.uint8)
    regions = measure.regionprops(labels)
    largest = max(regions, key=lambda r: r.area)
    return (labels == largest.label).astype(np.uint8)


def load_view_data(image_path: str, breast_side: str) -> ViewData:
    flip_right = breast_side.strip().upper() == "RIGHT"
    raw = load_grayscale_image(image_path, flip_right=flip_right)
    preprocessed = preprocess_mammogram(raw)
    breast_mask = extract_breast_mask(preprocessed)
    return ViewData(
        image_path=image_path,
        raw_image=raw,
        preprocessed_image=preprocessed,
        breast_mask=breast_mask,
    )
