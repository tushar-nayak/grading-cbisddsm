from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import laplace


@dataclass
class CorrespondenceResult:
    cc_center_norm: tuple[float, float]
    mlo_center_norm: tuple[float, float]
    fused_center_norm: tuple[float, float]
    centroid_distance_px: float
    cc_weight: float
    mlo_weight: float

    def to_dict(self):
        return asdict(self)


def assess_image_quality(image: np.ndarray) -> float:
    lap_var = float(laplace(image).var())
    contrast = float(image.std())
    quality = min(1.0, (lap_var / 0.005) * 0.6 + (contrast / 0.3) * 0.4)
    return float(np.clip(quality, 0.05, 1.0))


def compute_cross_view_correspondence(cc_mask: np.ndarray, mlo_mask: np.ndarray, cc_image: np.ndarray, mlo_image: np.ndarray) -> CorrespondenceResult:
    cc_ys, cc_xs = np.where(cc_mask > 0)
    mlo_ys, mlo_xs = np.where(mlo_mask > 0)

    h = max(cc_mask.shape[0], mlo_mask.shape[0])
    w = max(cc_mask.shape[1], mlo_mask.shape[1])

    cc_cy = float(np.mean(cc_ys)) if len(cc_ys) else cc_mask.shape[0] / 2.0
    cc_cx = float(np.mean(cc_xs)) if len(cc_xs) else cc_mask.shape[1] / 2.0
    mlo_cy = float(np.mean(mlo_ys)) if len(mlo_ys) else mlo_mask.shape[0] / 2.0
    mlo_cx = float(np.mean(mlo_xs)) if len(mlo_xs) else mlo_mask.shape[1] / 2.0

    cc_center_norm = (cc_cx / cc_mask.shape[1], cc_cy / cc_mask.shape[0])
    mlo_center_norm = (mlo_cx / mlo_mask.shape[1], mlo_cy / mlo_mask.shape[0])

    cc_q = assess_image_quality(cc_image)
    mlo_q = assess_image_quality(mlo_image)
    total_q = cc_q + mlo_q
    cc_weight = cc_q / total_q
    mlo_weight = mlo_q / total_q

    fused_center_norm = (
        cc_weight * cc_center_norm[0] + mlo_weight * mlo_center_norm[0],
        cc_weight * cc_center_norm[1] + mlo_weight * mlo_center_norm[1],
    )
    dist = float(np.sqrt((cc_center_norm[0] - mlo_center_norm[0]) ** 2 + (cc_center_norm[1] - mlo_center_norm[1]) ** 2) * max(h, w))
    return CorrespondenceResult(cc_center_norm, mlo_center_norm, fused_center_norm, dist, float(cc_weight), float(mlo_weight))
