import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import shift
from skimage.transform import resize

from unified_mammo_pipeline.alignment import align_mlo_to_cc
from unified_mammo_pipeline.classification import BiradsClassifier
from unified_mammo_pipeline.correspondence import compute_cross_view_correspondence
from unified_mammo_pipeline.data import extract_breast_mask, load_grayscale_image, preprocess_mammogram
from unified_mammo_pipeline.detection import detect_lesion_bbox
from unified_mammo_pipeline.segmentation import segment_lesion


def classification_accuracy(y_true: list[int], y_pred: list[int]) -> float | None:
    if not y_true:
        return None
    return float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true))


def weighted_f1(y_true: list[int], y_pred: list[int]) -> float | None:
    if not y_true:
        return None
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    score = 0.0
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        score += f1 * support
    return float(score / total)


def confusion_matrix_list(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    if not y_true:
        return []
    labels = sorted(set(y_true) | set(y_pred))
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        matrix[idx[yt]][idx[yp]] += 1
    return matrix


CSV_FILES = [
    "mass_case_description_train_set.csv",
    "calc_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_test_set.csv",
]


def parse_args():
    base_dir = Path("/home/sofa/host_dir/spatial_alignment/dataset/raw/cbisddsm-kaggle")
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Full unified pipeline on the CBIS-DDSM Kaggle dataset")
    parser.add_argument("--dataset-base", default=str(base_dir))
    parser.add_argument("--classifier-checkpoint", default=str(root_dir / "output" / "cnn_attentional_weights.pth"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "kaggle_full_run"))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-long-side", type=int, default=1536)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_uid_index(jpeg_root: Path):
    index = {}
    for folder in jpeg_root.iterdir():
        if folder.is_dir():
            index[folder.name] = sorted(folder.glob("*.jpg"))
    return index


def resolve_from_dicom_path(path_str: str, uid_index: dict[str, list[Path]], prefer_prefix: str) -> Path | None:
    parts = str(path_str).strip().replace("\\", "/").split("/")
    for part in parts:
        if part in uid_index:
            files = uid_index[part]
            preferred = [f for f in files if f.name.startswith(prefer_prefix)]
            if preferred:
                return preferred[0]
            if files:
                return files[0]
    return None


def build_kaggle_manifest(dataset_base: Path, output_dir: Path) -> pd.DataFrame:
    csv_root = dataset_base / "csv"
    jpeg_root = dataset_base / "jpeg"
    uid_index = build_uid_index(jpeg_root)

    rows = []
    for name in CSV_FILES:
        csv_path = csv_root / name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            image_path = resolve_from_dicom_path(row.get("image file path", ""), uid_index, prefer_prefix="1-")
            roi_mask_path = resolve_from_dicom_path(row.get("ROI mask file path", ""), uid_index, prefer_prefix="2-")
            if image_path is None:
                continue
            rows.append(
                {
                    "patient_id": str(row.get("patient_id", "")).strip(),
                    "breast_side": str(row.get("left or right breast", "")).strip().upper(),
                    "view": str(row.get("image view", "")).strip().upper(),
                    "birads": int(row.get("assessment", 0)),
                    "pathology": str(row.get("pathology", "")).strip(),
                    "abnormality_type": str(row.get("abnormality type", "")).strip(),
                    "image_path": str(image_path),
                    "roi_mask_path": str(roi_mask_path) if roi_mask_path else "",
                    "source_csv": name,
                }
            )

    df = pd.DataFrame(rows)
    grouped = {}
    for _, row in df.iterrows():
        key = (row["patient_id"], row["breast_side"], row["view"])
        grouped.setdefault(key, {"image_path": row["image_path"], "roi_mask_paths": [], "birads": [], "pathology": [], "abnormality_type": []})
        if row["roi_mask_path"]:
            grouped[key]["roi_mask_paths"].append(row["roi_mask_path"])
        grouped[key]["birads"].append(int(row["birads"]))
        if row["pathology"]:
            grouped[key]["pathology"].append(row["pathology"])
        if row["abnormality_type"]:
            grouped[key]["abnormality_type"].append(row["abnormality_type"])

    paired_rows = []
    patient_side_keys = sorted({(patient_id, side) for patient_id, side, _ in grouped.keys()})
    for patient_id, side in patient_side_keys:
        cc = grouped.get((patient_id, side, "CC"))
        mlo = grouped.get((patient_id, side, "MLO"))
        if not cc or not mlo:
            continue
        sample_id = f"{patient_id}_{side}"
        paired_rows.append(
            {
                "patient_id": patient_id,
                "sample_id": sample_id,
                "breast_side": side,
                "birads_label": int(max(cc["birads"] + mlo["birads"])),
                "cc_image_path": cc["image_path"],
                "mlo_image_path": mlo["image_path"],
                "cc_roi_mask_paths": json.dumps(sorted(set(cc["roi_mask_paths"]))),
                "mlo_roi_mask_paths": json.dumps(sorted(set(mlo["roi_mask_paths"]))),
                "pathology": ";".join(sorted(set(cc["pathology"] + mlo["pathology"]))),
                "abnormality_type": ";".join(sorted(set(cc["abnormality_type"] + mlo["abnormality_type"]))),
            }
        )

    manifest = pd.DataFrame(paired_rows)
    manifest_path = output_dir / "paired_kaggle_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def resize_long_side(image: np.ndarray, max_long_side: int) -> np.ndarray:
    h, w = image.shape
    long_side = max(h, w)
    if long_side <= max_long_side:
        return image.astype(np.float32)
    scale = max_long_side / long_side
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    resized = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32) / 255.0


def load_roi_union(mask_paths_json: str, breast_side: str, target_shape: tuple[int, int], max_long_side: int) -> np.ndarray:
    mask_paths = json.loads(mask_paths_json) if mask_paths_json else []
    union = np.zeros(target_shape, dtype=np.uint8)
    for mask_path in mask_paths:
        mask_img = load_grayscale_image(mask_path, flip_right=breast_side == "RIGHT")
        mask_img = resize_long_side(mask_img, max_long_side)
        if mask_img.shape != target_shape:
            mask_img = resize(mask_img, target_shape, preserve_range=True, order=0, anti_aliasing=False)
        union = np.maximum(union, (mask_img > 0.5).astype(np.uint8))
    return union


def bbox_to_mask(bbox: list[int] | None, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if bbox is None:
        return mask
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(shape[1], x + w)
    y2 = min(shape[0], y + h)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(a, b).sum()) / denom)


def iou_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    precision = float(tp / pred.sum()) if pred.sum() else 0.0
    recall = float(tp / target.sum()) if target.sum() else 0.0
    return precision, recall


def center_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    ay, ax = np.where(mask_a > 0)
    by, bx = np.where(mask_b > 0)
    if len(ay) == 0 or len(by) == 0:
        return float("nan")
    return float(np.sqrt((ax.mean() - bx.mean()) ** 2 + (ay.mean() - by.mean()) ** 2))


def overlay_detection(image: np.ndarray, bbox: list[int] | None, out_path: Path):
    base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(np.stack([base, base, base], axis=-1))
    draw = ImageDraw.Draw(pil)
    if bbox is not None:
        x, y, w, h = bbox
        draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=4)
    pil.save(out_path)


def overlay_segmentation(image: np.ndarray, mask: np.ndarray, out_path: Path):
    base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[mask > 0] = np.array([255, 120, 0], dtype=np.uint8)
    Image.fromarray(rgb).save(out_path)


def save_alignment_overlay(cc_image: np.ndarray, aligned_mlo: np.ndarray, out_path: Path):
    cc = Image.fromarray(np.clip(cc_image * 255.0, 0, 255).astype(np.uint8)).convert("L")
    mlo = Image.fromarray(np.clip(aligned_mlo * 255.0, 0, 255).astype(np.uint8)).convert("L")
    Image.blend(cc, mlo, 0.5).convert("RGB").save(out_path)


def main():
    args = parse_args()
    dataset_base = Path(args.dataset_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["manifests", "detection", "segmentation", "alignment", "correspondence", "classification", "metrics"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    manifest = build_kaggle_manifest(dataset_base, output_dir / "manifests")
    if args.limit:
        manifest = manifest.iloc[: args.limit].copy()

    classifier = BiradsClassifier(args.classifier_checkpoint, device=args.device)
    per_sample = []
    y_true = []
    y_pred = []

    for _, row in manifest.iterrows():
        side = str(row["breast_side"]).strip().upper()
        sample_id = str(row["sample_id"])

        cc_raw = resize_long_side(load_grayscale_image(row["cc_image_path"], flip_right=side == "RIGHT"), args.max_long_side)
        mlo_raw = resize_long_side(load_grayscale_image(row["mlo_image_path"], flip_right=side == "RIGHT"), args.max_long_side)
        cc_gt_mask = load_roi_union(row["cc_roi_mask_paths"], side, cc_raw.shape, args.max_long_side)
        mlo_gt_mask = load_roi_union(row["mlo_roi_mask_paths"], side, mlo_raw.shape, args.max_long_side)

        cc_img = preprocess_mammogram(cc_raw)
        mlo_img = preprocess_mammogram(mlo_raw)
        cc_breast = extract_breast_mask(cc_img)
        mlo_breast = extract_breast_mask(mlo_img)

        cc_seg = segment_lesion(cc_img, cc_breast)
        cc_det = detect_lesion_bbox(cc_img, cc_breast)
        mlo_seg = segment_lesion(mlo_img, mlo_breast)
        mlo_det = detect_lesion_bbox(mlo_img, mlo_breast)

        correspondence = compute_cross_view_correspondence(cc_seg.mask, mlo_seg.mask, cc_img, mlo_img)
        alignment = align_mlo_to_cc(cc_img, mlo_img, cc_seg.mask, mlo_seg.mask)

        mlo_seg_resized = resize(mlo_seg.mask.astype(np.float32), cc_seg.mask.shape, preserve_range=True, order=0, anti_aliasing=False) > 0.5
        aligned_mlo_mask = shift(mlo_seg_resized.astype(np.float32), shift=(alignment.affine_matrix[1][2], alignment.affine_matrix[0][2]), order=0, mode="constant", cval=0.0) > 0.5

        cls = classifier.predict(cc_img, alignment.aligned_image, cc_seg.mask, aligned_mlo_mask.astype(np.uint8), cc_det.bbox_xywh, mlo_det.bbox_xywh)

        y_true.append(int(row["birads_label"]))
        y_pred.append(int(cls.predicted_birads))

        cc_det_mask = bbox_to_mask(cc_det.bbox_xywh, cc_seg.mask.shape)
        mlo_det_mask = bbox_to_mask(mlo_det.bbox_xywh, mlo_seg.mask.shape)
        cc_precision, cc_recall = precision_recall(cc_seg.mask, cc_gt_mask)
        mlo_precision, mlo_recall = precision_recall(mlo_seg.mask, mlo_gt_mask)

        sample_metrics = {
            "sample_id": sample_id,
            "patient_id": row["patient_id"],
            "true_birads": int(row["birads_label"]),
            "predicted_birads": int(cls.predicted_birads),
            "classifier_source": cls.source,
            "cc_seg_dice_vs_roi": dice_score(cc_seg.mask, cc_gt_mask),
            "cc_seg_iou_vs_roi": iou_score(cc_seg.mask, cc_gt_mask),
            "cc_seg_precision_vs_roi": cc_precision,
            "cc_seg_recall_vs_roi": cc_recall,
            "mlo_seg_dice_vs_roi": dice_score(mlo_seg.mask, mlo_gt_mask),
            "mlo_seg_iou_vs_roi": iou_score(mlo_seg.mask, mlo_gt_mask),
            "mlo_seg_precision_vs_roi": mlo_precision,
            "mlo_seg_recall_vs_roi": mlo_recall,
            "cc_det_seg_iou": iou_score(cc_det_mask, cc_seg.mask),
            "mlo_det_seg_iou": iou_score(mlo_det_mask, mlo_seg.mask),
            "cc_detection_confidence": float(cc_det.confidence),
            "mlo_detection_confidence": float(mlo_det.confidence),
            "correspondence_centroid_distance_px": float(correspondence.centroid_distance_px),
            "alignment_dice_before": dice_score(cc_seg.mask, mlo_seg_resized),
            "alignment_dice_after": dice_score(cc_seg.mask, aligned_mlo_mask),
            "alignment_centroid_distance_before": center_distance(cc_seg.mask, mlo_seg_resized),
            "alignment_centroid_distance_after": center_distance(cc_seg.mask, aligned_mlo_mask),
        }
        per_sample.append(sample_metrics)

        overlay_detection(cc_img, cc_det.bbox_xywh, output_dir / "detection" / f"{sample_id}_CC_detection.png")
        overlay_detection(mlo_img, mlo_det.bbox_xywh, output_dir / "detection" / f"{sample_id}_MLO_detection.png")
        overlay_segmentation(cc_img, cc_seg.mask, output_dir / "segmentation" / f"{sample_id}_CC_segmentation.png")
        overlay_segmentation(mlo_img, mlo_seg.mask, output_dir / "segmentation" / f"{sample_id}_MLO_segmentation.png")
        save_alignment_overlay(cc_img, alignment.aligned_image, output_dir / "alignment" / f"{sample_id}_alignment_overlay.png")
        (output_dir / "correspondence" / f"{sample_id}_correspondence.json").write_text(json.dumps(correspondence.to_dict(), indent=2))
        (output_dir / "classification" / f"{sample_id}_classification.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "true_birads": int(row["birads_label"]),
                    "predicted_birads": int(cls.predicted_birads),
                    "ordinal_scores": cls.ordinal_scores,
                    "source": cls.source,
                },
                indent=2,
            )
        )

    per_sample_df = pd.DataFrame(per_sample)
    per_sample_df.to_csv(output_dir / "metrics" / "per_sample_metrics.csv", index=False)

    aggregate = {
        "dataset_base": str(dataset_base),
        "num_samples": int(len(per_sample_df)),
        "classification_accuracy": classification_accuracy(y_true, y_pred),
        "classification_f1_weighted": weighted_f1(y_true, y_pred),
        "classification_confusion_matrix": confusion_matrix_list(y_true, y_pred),
        "mean_cc_seg_dice_vs_roi": float(per_sample_df["cc_seg_dice_vs_roi"].mean()) if not per_sample_df.empty else None,
        "mean_mlo_seg_dice_vs_roi": float(per_sample_df["mlo_seg_dice_vs_roi"].mean()) if not per_sample_df.empty else None,
        "mean_cc_det_seg_iou": float(per_sample_df["cc_det_seg_iou"].mean()) if not per_sample_df.empty else None,
        "mean_mlo_det_seg_iou": float(per_sample_df["mlo_det_seg_iou"].mean()) if not per_sample_df.empty else None,
        "mean_alignment_dice_before": float(per_sample_df["alignment_dice_before"].mean()) if not per_sample_df.empty else None,
        "mean_alignment_dice_after": float(per_sample_df["alignment_dice_after"].mean()) if not per_sample_df.empty else None,
        "mean_alignment_centroid_distance_before": float(per_sample_df["alignment_centroid_distance_before"].dropna().mean()) if not per_sample_df.empty else None,
        "mean_alignment_centroid_distance_after": float(per_sample_df["alignment_centroid_distance_after"].dropna().mean()) if not per_sample_df.empty else None,
    }
    (output_dir / "metrics" / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
