from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology

from localizer import predict_localization


@dataclass
class DetectionResult:
    bbox_xywh: list[int] | None
    confidence: float
    heatmap: np.ndarray


def _heuristic_detect(image: np.ndarray, breast_mask: np.ndarray) -> DetectionResult:
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


def detect_lesion_bbox(image: np.ndarray, breast_mask: np.ndarray) -> DetectionResult:
    localization = predict_localization(image, breast_mask)
    if localization is not None:
        ys, xs = np.where(localization.binary_mask > 0)
        if len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            return DetectionResult(bbox_xywh=bbox, confidence=float(localization.confidence), heatmap=localization.probability_map)
        return DetectionResult(None, float(localization.confidence), localization.probability_map)
    return _heuristic_detect(image, breast_mask)
