import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from .alignment import align_mlo_to_cc
    from .classification import BiradsClassifier
    from .correspondence import compute_cross_view_correspondence
    from .data import ManifestLoader, load_view_data
    from .detection import detect_lesion_bbox
    from .segmentation import segment_lesion
except ImportError:
    from alignment import align_mlo_to_cc
    from classification import BiradsClassifier
    from correspondence import compute_cross_view_correspondence
    from data import ManifestLoader, load_view_data
    from detection import detect_lesion_bbox
    from segmentation import segment_lesion


@dataclass
class ViewStageResult:
    image_path: str
    bbox_xywh: list[int] | None
    detection_confidence: float
    lesion_center_yx: tuple[int, int]
    lesion_area: int
    segmentation_score: float


@dataclass
class SampleStageResult:
    patient_id: str
    sample_id: str
    breast_side: str
    true_birads: int
    predicted_birads: int
    classifier_source: str
    cc: ViewStageResult
    mlo: ViewStageResult
    correspondence: dict
    alignment: dict


class UnifiedMammoPipeline:
    def __init__(self, manifest_csv: str, classifier_checkpoint: str | None = None, output_dir: str | None = None, device: str | None = None):
        self.loader = ManifestLoader(manifest_csv)
        self.classifier = BiradsClassifier(classifier_checkpoint, device=device)
        base_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "trial_run"
        self.output_dir = base_dir
        self.preview_dir = base_dir / "previews"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def _run_view(self, view_data):
        with ThreadPoolExecutor(max_workers=2) as executor:
            seg_future = executor.submit(segment_lesion, view_data.preprocessed_image, view_data.breast_mask)
            det_future = executor.submit(detect_lesion_bbox, view_data.preprocessed_image, view_data.breast_mask)
            return seg_future.result(), det_future.result()

    def _overlay(self, image: np.ndarray, seg, det) -> Image.Image:
        base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([base, base, base], axis=-1)
        rgb[seg.mask > 0] = np.array([255, 120, 0], dtype=np.uint8)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        cy, cx = seg.center_yx
        draw.ellipse((cx - 20, cy - 20, cx + 20, cy + 20), outline=(0, 255, 255), width=4)
        if det.bbox_xywh is not None:
            x, y, w, h = det.bbox_xywh
            draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=5)
        return pil

    def _save_preview(self, sample_id: str, cc_data, mlo_data, cc_seg, mlo_seg, cc_det, mlo_det, aligned_mlo: np.ndarray):
        cc_overlay = self._overlay(cc_data.preprocessed_image, cc_seg, cc_det).resize((640, 640))
        mlo_overlay = self._overlay(mlo_data.preprocessed_image, mlo_seg, mlo_det).resize((640, 640))
        aligned_u8 = np.clip(aligned_mlo * 255.0, 0, 255).astype(np.uint8)
        aligned_rgb = Image.fromarray(np.stack([aligned_u8, aligned_u8, aligned_u8], axis=-1)).resize((640, 640))
        fused = Image.blend(cc_overlay.convert("L"), aligned_rgb.convert("L"), 0.5).convert("RGB")
        canvas = Image.new("RGB", (1280, 1280), color=(0, 0, 0))
        canvas.paste(cc_overlay, (0, 0))
        canvas.paste(mlo_overlay, (640, 0))
        canvas.paste(aligned_rgb, (0, 640))
        canvas.paste(fused, (640, 640))
        out_path = self.preview_dir / f"{sample_id}.png"
        canvas.save(out_path)

    def run(self, limit: int | None = None):
        results = []
        for sample in self.loader.load_samples(limit=limit):
            cc_data = load_view_data(sample.cc_image_path, sample.breast_side)
            mlo_data = load_view_data(sample.mlo_image_path, sample.breast_side)
            cc_seg, cc_det = self._run_view(cc_data)
            mlo_seg, mlo_det = self._run_view(mlo_data)

            corr = compute_cross_view_correspondence(
                cc_seg.mask,
                mlo_seg.mask,
                cc_data.preprocessed_image,
                mlo_data.preprocessed_image,
            )
            alignment = align_mlo_to_cc(
                cc_data.preprocessed_image,
                mlo_data.preprocessed_image,
                cc_seg.mask,
                mlo_seg.mask,
            )
            classification = self.classifier.predict(
                cc_data.preprocessed_image,
                alignment.aligned_image,
                cc_seg.mask,
                mlo_seg.mask,
                cc_det.bbox_xywh,
                mlo_det.bbox_xywh,
            )

            result = SampleStageResult(
                patient_id=sample.patient_id,
                sample_id=sample.sample_id,
                breast_side=sample.breast_side,
                true_birads=sample.birads_label,
                predicted_birads=classification.predicted_birads,
                classifier_source=classification.source,
                cc=ViewStageResult(
                    image_path=sample.cc_image_path,
                    bbox_xywh=cc_det.bbox_xywh,
                    detection_confidence=cc_det.confidence,
                    lesion_center_yx=cc_seg.center_yx,
                    lesion_area=cc_seg.area,
                    segmentation_score=cc_seg.score,
                ),
                mlo=ViewStageResult(
                    image_path=sample.mlo_image_path,
                    bbox_xywh=mlo_det.bbox_xywh,
                    detection_confidence=mlo_det.confidence,
                    lesion_center_yx=mlo_seg.center_yx,
                    lesion_area=mlo_seg.area,
                    segmentation_score=mlo_seg.score,
                ),
                correspondence=corr.to_dict(),
                alignment=alignment.to_dict(),
            )
            results.append((result, classification.ordinal_scores))
            self._save_preview(sample.sample_id, cc_data, mlo_data, cc_seg, mlo_seg, cc_det, mlo_det, alignment.aligned_image)

        payload = []
        for result, ordinal_scores in results:
            item = asdict(result)
            item["ordinal_scores"] = ordinal_scores
            payload.append(item)
        out_path = self.output_dir / "trial_results.json"
        out_path.write_text(json.dumps(payload, indent=2))
        return out_path, payload
