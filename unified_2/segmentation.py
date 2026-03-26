from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology


@dataclass
class SegmentationResult:
    mask: np.ndarray
    center_yx: tuple[int, int]
    area: int
    score: float


def _largest_component(binary: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, int, float]:
    labels = measure.label(binary)
    if labels.max() == 0:
        return np.zeros_like(binary, dtype=np.uint8), 0, 0.0
    best_mask = np.zeros_like(binary, dtype=np.uint8)
    best_area = 0
    best_score = -1.0
    for region in measure.regionprops(labels, intensity_image=image):
        if region.area < 20:
            continue
        score = float(region.mean_intensity * np.sqrt(region.area))
        if score > best_score:
            best_score = score
            best_area = int(region.area)
            best_mask = (labels == region.label).astype(np.uint8)
    return best_mask, best_area, max(best_score, 0.0)


def segment_lesion(image: np.ndarray, breast_mask: np.ndarray) -> SegmentationResult:
    masked = image * breast_mask
    smooth = gaussian_filter(masked, sigma=1.2)
    background = gaussian_filter(masked, sigma=9.0)
    heatmap = np.clip(smooth - background, 0.0, 1.0)

    active = heatmap[breast_mask > 0]
    if active.size == 0:
        empty = np.zeros_like(breast_mask, dtype=np.uint8)
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
        return SegmentationResult(empty, (cy, cx), 0, 0.0)

    threshold = float(active.mean() + 0.75 * active.std())
    binary = (heatmap > threshold) & (breast_mask > 0)
    binary = morphology.binary_closing(binary, morphology.disk(5))
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = morphology.remove_small_objects(binary, min_size=20)
    mask, area, score = _largest_component(binary, image)

    if area == 0:
        fallback = (masked > np.quantile(active, 0.995)) & (breast_mask > 0)
        fallback = morphology.binary_dilation(fallback, morphology.disk(4))
        mask, area, score = _largest_component(fallback, image)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        ys, xs = np.where(breast_mask > 0)
    cy = int(np.mean(ys)) if len(ys) else image.shape[0] // 2
    cx = int(np.mean(xs)) if len(xs) else image.shape[1] // 2
    return SegmentationResult(mask.astype(np.uint8), (cy, cx), area, float(score))
