from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology


@dataclass
class DetectionResult:
    bbox_xywh: list[int] | None
    confidence: float
    heatmap: np.ndarray


def detect_lesion_bbox(image: np.ndarray, breast_mask: np.ndarray) -> DetectionResult:
    masked = image * breast_mask
    local = gaussian_filter(masked, sigma=1.0)
    broad = gaussian_filter(masked, sigma=12.0)
    heatmap = np.clip(local - broad, 0.0, 1.0)
    active = heatmap[breast_mask > 0]
    if active.size == 0:
        return DetectionResult(None, 0.0, heatmap)

    threshold = float(active.mean() + 1.0 * active.std())
    binary = (heatmap > threshold) & (breast_mask > 0)
    binary = morphology.binary_dilation(binary, morphology.disk(6))
    binary = morphology.remove_small_objects(binary, min_size=20)
    labels = measure.label(binary)
    if labels.max() == 0:
        return DetectionResult(None, 0.0, heatmap)

    region = max(measure.regionprops(labels, intensity_image=heatmap), key=lambda r: r.area)
    min_row, min_col, max_row, max_col = region.bbox
    x, y = int(min_col), int(min_row)
    w, h = int(max_col - min_col), int(max_row - min_row)
    confidence = float(min(1.0, region.mean_intensity * 2.5 + np.sqrt(max(region.area, 1.0)) / 1000.0))
    return DetectionResult([x, y, w, h], confidence, heatmap)
