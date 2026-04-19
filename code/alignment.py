from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift
from skimage.transform import resize


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    affine_matrix: list[list[float]]

    def to_dict(self):
        return {"affine_matrix": self.affine_matrix}


def _center_from_mask(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    return float(np.mean(xs)), float(np.mean(ys))


def align_mlo_to_cc(cc_image: np.ndarray, mlo_image: np.ndarray, cc_mask: np.ndarray, mlo_mask: np.ndarray) -> AlignmentResult:
    target_h, target_w = cc_image.shape
    mlo_resized = resize(mlo_image, (target_h, target_w), preserve_range=True, anti_aliasing=True).astype(np.float32)
    mlo_mask_resized = resize(mlo_mask.astype(np.float32), (target_h, target_w), preserve_range=True, order=0, anti_aliasing=False) > 0.5

    cc_cx, cc_cy = _center_from_mask(cc_mask)
    mlo_cx, mlo_cy = _center_from_mask(mlo_mask_resized)
    dx = cc_cx - mlo_cx
    dy = cc_cy - mlo_cy

    aligned = shift(mlo_resized, shift=(dy, dx), order=1, mode="constant", cval=0.0).astype(np.float32)
    matrix = [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]]
    return AlignmentResult(aligned_image=aligned, affine_matrix=matrix)
