import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models, transforms


def normalize_path(path: Optional[str]) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve())


@dataclass
class ViewInferenceResult:
    view: str
    bbox: Optional[List[int]]
    bbox_confidence: Optional[float]
    segmentation_path: Optional[str]
    segmentation_pixels: int
    coverage_ratio: float


@dataclass
class SampleInferenceResult:
    sample_id: str
    patient_id: str
    breast_side: str
    label: Optional[int]
    cc_result: ViewInferenceResult
    mlo_result: ViewInferenceResult
    alignment_theta: Optional[List[List[float]]]
    prediction: Optional[int]
    ordinal_scores: Optional[List[float]]


class RawSpatialTransformer(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels * 2, 16, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2),
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, cc_view: torch.Tensor, mlo_view: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([cc_view, mlo_view], dim=1)
        xs = self.localization(x).view(-1, 32 * 7 * 7)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        theta_constrained = theta.clone()
        theta_constrained[:, 0, 0] = torch.clamp(theta[:, 0, 0], min=0.8, max=1.2)
        theta_constrained[:, 1, 1] = torch.clamp(theta[:, 1, 1], min=0.8, max=1.2)

        grid = F.affine_grid(theta_constrained, mlo_view.size(), align_corners=True)
        aligned_mlo = F.grid_sample(mlo_view, grid, align_corners=True)
        return aligned_mlo, theta_constrained


class CNNCrossAttentionGrader(nn.Module):
    def __init__(self, num_classes: int = 6, backbone_weights=None):
        super().__init__()
        resnet = models.resnet50(weights=backbone_weights)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        embed_dim = 2048
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes - 1),
        )

    def forward(self, dual_tensor: torch.Tensor) -> torch.Tensor:
        cc_img = dual_tensor[:, 0:1, :, :]
        mlo_img = dual_tensor[:, 1:2, :, :]
        cc_feat = self.feature_extractor(cc_img)
        mlo_feat = self.feature_extractor(mlo_img)
        batch_size, channels, height, width = cc_feat.shape
        cc_seq = cc_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        mlo_seq = mlo_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        attended_mlo_seq, _ = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq)
        attended_mlo_feat = attended_mlo_seq.permute(0, 2, 1).view(batch_size, channels, height, width)
        cc_pooled = self.global_pool(cc_feat).view(batch_size, -1)
        mlo_pooled = self.global_pool(attended_mlo_feat).view(batch_size, -1)
        fused = torch.cat([cc_pooled, mlo_pooled], dim=1)
        return self.classifier(fused)


class DetectionBackend:
    def predict(self, patient_id: str, view: str, image_path: str) -> Tuple[Optional[List[int]], Optional[float]]:
        return None, None


class JsonDetectionBackend(DetectionBackend):
    def __init__(self, json_path: Path):
        payload = json.loads(Path(json_path).read_text())
        detections = payload.get("detections", payload)
        self.by_path: Dict[str, dict] = {}
        self.by_patient_view: Dict[Tuple[str, str], dict] = {}
        for item in detections:
            image_path = normalize_path(item.get("img_path"))
            if image_path:
                self.by_path[image_path] = item
            patient_id = str(item.get("patient_id", "")).strip()
            view = str(item.get("view", "")).strip().upper()
            if patient_id and view:
                self.by_patient_view[(patient_id, view)] = item

    def predict(self, patient_id: str, view: str, image_path: str) -> Tuple[Optional[List[int]], Optional[float]]:
        item = self.by_path.get(normalize_path(image_path))
        if item is None:
            item = self.by_patient_view.get((str(patient_id).strip(), str(view).strip().upper()))
        if item is None:
            return None, None
        bbox = item.get("bbox")
        confidence = item.get("confidence")
        return bbox, float(confidence) if confidence is not None else None


class SegmentationBackend:
    def segment(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        return None, None


class SummarySegmentationBackend(SegmentationBackend):
    def __init__(self, summary_csv: Path, export_root: Optional[Path] = None, allow_ground_truth_fallback: bool = True):
        frame = pd.read_csv(summary_csv)
        self.by_path: Dict[str, dict] = {}
        self.export_root = Path(export_root).expanduser().resolve() if export_root else None
        self.allow_ground_truth_fallback = allow_ground_truth_fallback
        for _, row in frame.iterrows():
            image_path = normalize_path(row.get("img_path"))
            if image_path:
                self.by_path[image_path] = row.to_dict()

    def _resolve_mask_path(self, row: dict) -> Optional[Path]:
        parent_folder = row.get("parent_folder")
        original_name = row.get("original_name")
        if self.export_root and parent_folder and original_name:
            candidate = self.export_root / str(parent_folder) / f"{original_name}_segmented.png"
            if candidate.exists():
                return candidate
        if self.allow_ground_truth_fallback and row.get("mask_path"):
            candidate = Path(str(row["mask_path"])).expanduser()
            if candidate.exists():
                return candidate.resolve()
        return None

    def segment(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        row = self.by_path.get(normalize_path(image_path))
        if row is None:
            return None, None
        mask_path = self._resolve_mask_path(row)
        if mask_path is None:
            return None, None
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        binary_mask = (mask > 127).astype(np.float32)
        return binary_mask, str(mask_path)


class UnifiedMammographyPipeline:
    def __init__(
        self,
        manifest_csv: Path,
        alignment_weights: Optional[Path] = None,
        grader_weights: Optional[Path] = None,
        detection_backend: Optional[DetectionBackend] = None,
        segmentation_backend: Optional[SegmentationBackend] = None,
        device: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.records = pd.read_csv(self.manifest_csv).to_dict(orient="records")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = Path(output_dir) if output_dir else self.manifest_csv.parent / "output" / "unified_pipeline"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.detection_backend = detection_backend or DetectionBackend()
        self.segmentation_backend = segmentation_backend or SegmentationBackend()
        self.alignment_model = self._load_alignment_model(alignment_weights)
        self.grader_model = self._load_grader_model(grader_weights)

    def _load_alignment_model(self, weights_path: Optional[Path]) -> RawSpatialTransformer:
        model = RawSpatialTransformer().to(self.device)
        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_grader_model(self, weights_path: Optional[Path]) -> Optional[CNNCrossAttentionGrader]:
        if not weights_path:
            return None
        model = CNNCrossAttentionGrader(backbone_weights=None).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_view_tensor(self, image_path: str, breast_side: str) -> Tuple[torch.Tensor, np.ndarray]:
        image = Image.open(image_path).convert("L")
        if str(breast_side).strip().upper() == "RIGHT":
            image = TF.hflip(image)
        tensor = self.transform(image)
        raw = np.array(image, dtype=np.float32) / 255.0
        return tensor, raw

    def _make_guidance_mask(
        self,
        raw_image: np.ndarray,
        bbox: Optional[List[int]],
        segmentation_mask: Optional[np.ndarray],
    ) -> torch.Tensor:
        mask = np.zeros(raw_image.shape, dtype=np.float32)
        if segmentation_mask is not None and segmentation_mask.shape == raw_image.shape:
            mask = np.maximum(mask, segmentation_mask.astype(np.float32))
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(raw_image.shape[1], x + w)
            y2 = min(raw_image.shape[0], y + h)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        return transforms.ToTensor()(mask_image)

    def _prepare_guided_tensor(
        self,
        image_tensor: torch.Tensor,
        raw_image: np.ndarray,
        bbox: Optional[List[int]],
        segmentation_mask: Optional[np.ndarray],
    ) -> torch.Tensor:
        guidance_mask = self._make_guidance_mask(raw_image, bbox, segmentation_mask)
        guidance_mask = transforms.Resize((224, 224))(guidance_mask)
        emphasis = 1.0 + (0.35 * guidance_mask)
        return image_tensor * emphasis

    def _run_parallel_vision(self, patient_id: str, view: str, image_path: str) -> Tuple[Tuple[Optional[List[int]], Optional[float]], Tuple[Optional[np.ndarray], Optional[str]]]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            detection_future = executor.submit(self.detection_backend.predict, patient_id, view, image_path)
            segmentation_future = executor.submit(self.segmentation_backend.segment, image_path)
            return detection_future.result(), segmentation_future.result()

    def _infer_single(self, row: dict) -> SampleInferenceResult:
        patient_id = str(row.get("patient_id", ""))
        sample_id = str(row.get("sample_id", f"{patient_id}_{row['breast_side']}"))
        breast_side = str(row.get("breast_side", ""))
        label = int(row["birads_label"]) if "birads_label" in row and not pd.isna(row["birads_label"]) else None

        cc_path = str(row["cc_image_path"])
        mlo_path = str(row["mlo_image_path"])
        (cc_bbox, cc_conf), (cc_seg, cc_seg_path) = self._run_parallel_vision(patient_id, "CC", cc_path)
        (mlo_bbox, mlo_conf), (mlo_seg, mlo_seg_path) = self._run_parallel_vision(patient_id, "MLO", mlo_path)

        cc_tensor, cc_raw = self._load_view_tensor(cc_path, breast_side)
        mlo_tensor, mlo_raw = self._load_view_tensor(mlo_path, breast_side)
        cc_tensor = self._prepare_guided_tensor(cc_tensor, cc_raw, cc_bbox, cc_seg).unsqueeze(0).to(self.device)
        mlo_tensor = self._prepare_guided_tensor(mlo_tensor, mlo_raw, mlo_bbox, mlo_seg).unsqueeze(0).to(self.device)

        with torch.no_grad():
            aligned_mlo, theta = self.alignment_model(cc_tensor, mlo_tensor)

        prediction = None
        ordinal_scores = None
        if self.grader_model is not None:
            with torch.no_grad():
                logits = self.grader_model(torch.cat([cc_tensor, aligned_mlo], dim=1))
                scores = torch.sigmoid(logits).squeeze(0).cpu().tolist()
                prediction = int((torch.tensor(scores) > 0.5).sum().item())
                ordinal_scores = [float(x) for x in scores]

        cc_pixels = int(cc_seg.sum()) if cc_seg is not None else 0
        mlo_pixels = int(mlo_seg.sum()) if mlo_seg is not None else 0
        cc_total = int(cc_seg.size) if cc_seg is not None else 0
        mlo_total = int(mlo_seg.size) if mlo_seg is not None else 0

        return SampleInferenceResult(
            sample_id=sample_id,
            patient_id=patient_id,
            breast_side=breast_side,
            label=label,
            cc_result=ViewInferenceResult(
                view="CC",
                bbox=cc_bbox,
                bbox_confidence=cc_conf,
                segmentation_path=cc_seg_path,
                segmentation_pixels=cc_pixels,
                coverage_ratio=float(cc_pixels / cc_total) if cc_total else 0.0,
            ),
            mlo_result=ViewInferenceResult(
                view="MLO",
                bbox=mlo_bbox,
                bbox_confidence=mlo_conf,
                segmentation_path=mlo_seg_path,
                segmentation_pixels=mlo_pixels,
                coverage_ratio=float(mlo_pixels / mlo_total) if mlo_total else 0.0,
            ),
            alignment_theta=theta.squeeze(0).cpu().tolist(),
            prediction=prediction,
            ordinal_scores=ordinal_scores,
        )

    def run(self, limit: Optional[int] = None) -> List[SampleInferenceResult]:
        results: List[SampleInferenceResult] = []
        rows = self.records[:limit] if limit else self.records
        for row in rows:
            results.append(self._infer_single(row))
        return results

    def save_results(self, results: List[SampleInferenceResult]) -> Path:
        output_path = self.output_dir / "unified_pipeline_results.json"
        serialized = [asdict(item) for item in results]
        output_path.write_text(json.dumps(serialized, indent=2))
        return output_path


def build_detection_backend(args) -> DetectionBackend:
    if args.detection_json:
        return JsonDetectionBackend(Path(args.detection_json))
    return DetectionBackend()


def build_segmentation_backend(args) -> SegmentationBackend:
    if args.segmentation_summary:
        export_root = Path(args.segmentation_export_root) if args.segmentation_export_root else None
        return SummarySegmentationBackend(
            summary_csv=Path(args.segmentation_summary),
            export_root=export_root,
            allow_ground_truth_fallback=not args.disable_mask_fallback,
        )
    return SegmentationBackend()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified mammography pipeline: detection + segmentation -> alignment -> BI-RADS grading")
    parser.add_argument("--manifest-csv", default="dicom_clean_train.csv")
    parser.add_argument("--alignment-weights", default=None)
    parser.add_argument("--grader-weights", default=None)
    parser.add_argument("--detection-json", default=None)
    parser.add_argument("--segmentation-summary", default=None)
    parser.add_argument("--segmentation-export-root", default=None)
    parser.add_argument("--disable-mask-fallback", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = UnifiedMammographyPipeline(
        manifest_csv=Path(args.manifest_csv),
        alignment_weights=Path(args.alignment_weights) if args.alignment_weights else None,
        grader_weights=Path(args.grader_weights) if args.grader_weights else None,
        detection_backend=build_detection_backend(args),
        segmentation_backend=build_segmentation_backend(args),
        device=args.device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    results = pipeline.run(limit=args.limit)
    output_path = pipeline.save_results(results)
    print(f"Saved {len(results)} unified pipeline results to {output_path}")


if __name__ == "__main__":
    main()
