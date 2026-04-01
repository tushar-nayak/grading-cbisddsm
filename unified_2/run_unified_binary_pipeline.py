import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.ndimage import shift
from skimage.transform import resize

try:
    from unified_mammo_pipeline.alignment import align_mlo_to_cc
    from unified_mammo_pipeline.classification import BiradsClassifier, birads_to_binary_label
    from unified_mammo_pipeline.correspondence import compute_cross_view_correspondence
    from unified_mammo_pipeline.data import extract_breast_mask, load_grayscale_image, preprocess_mammogram
    from unified_mammo_pipeline.detection import detect_lesion_bbox
    from unified_mammo_pipeline.segmentation import segment_lesion
    from unified_mammo_pipeline.run_kaggle_full_pipeline import (
        bbox_to_mask,
        build_kaggle_manifest,
        center_distance,
        default_classifier_checkpoint,
        dice_score,
        iou_score,
        load_roi_union,
        overlay_detection,
        overlay_gt_comparison,
        overlay_segmentation,
        precision_recall,
        resize_long_side,
        save_alignment_overlay,
    )
except ImportError:
    from alignment import align_mlo_to_cc
    from classification import BiradsClassifier, birads_to_binary_label
    from correspondence import compute_cross_view_correspondence
    from data import extract_breast_mask, load_grayscale_image, preprocess_mammogram
    from detection import detect_lesion_bbox
    from segmentation import segment_lesion
    from run_kaggle_full_pipeline import (
        bbox_to_mask,
        build_kaggle_manifest,
        center_distance,
        default_classifier_checkpoint,
        dice_score,
        iou_score,
        load_roi_union,
        overlay_detection,
        overlay_gt_comparison,
        overlay_segmentation,
        precision_recall,
        resize_long_side,
        save_alignment_overlay,
    )


LABEL_NAMES = {
    0: 'birads_0_1_2_3',
    1: 'birads_4_5',
}


def parse_args():
    base_dir = Path('/home/sofa/host_dir/spatial_alignment/dataset/raw/cbisddsm-kaggle')
    parser = argparse.ArgumentParser(
        description='Unified mammography pipeline with segmentation, detection, alignment, and binary BI-RADS classification'
    )
    parser.add_argument('--dataset-base', default=str(base_dir))
    parser.add_argument('--manifest-csv', default=None)
    parser.add_argument('--classifier-checkpoint', default=default_classifier_checkpoint())
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'binary_full_run'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-long-side', type=int, default=1536)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_binary_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    labels = [0, 1]
    matrix = [[0, 0], [0, 0]]
    for yt, yp in zip(y_true, y_pred):
        matrix[int(yt)][int(yp)] += 1

    total = len(y_true)
    correct = matrix[0][0] + matrix[1][1]
    class_metrics = {}
    recalls = []
    f1s = []
    weighted_f1_sum = 0.0

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[row][label] for row in labels if row != label)
        fn = sum(matrix[label][col] for col in labels if col != label)
        tn = total - tp - fp - fn
        support = sum(matrix[label])
        predicted_count = matrix[0][label] + matrix[1][label]

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        npv = safe_div(tn, tn + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) else 0.0
        prevalence = safe_div(support, total)

        recalls.append(recall)
        f1s.append(f1)
        weighted_f1_sum += f1 * support
        class_metrics[str(label)] = {
            'label_name': LABEL_NAMES[label],
            'support': int(support),
            'predicted_count': int(predicted_count),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'npv': npv,
            'f1': f1,
            'prevalence': prevalence,
        }

    return {
        'labels': labels,
        'label_names': [LABEL_NAMES[label] for label in labels],
        'confusion_matrix': matrix,
        'num_samples': int(total),
        'accuracy': safe_div(correct, total),
        'balanced_accuracy': float(sum(recalls) / len(recalls)) if recalls else 0.0,
        'macro_f1': float(sum(f1s) / len(f1s)) if f1s else 0.0,
        'weighted_f1': safe_div(weighted_f1_sum, total),
        'class_metrics': class_metrics,
    }


def ensure_binary_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if 'binary_label' in manifest.columns:
        return manifest
    manifest = manifest.copy()
    manifest['binary_label'] = manifest['birads_label'].map(birads_to_binary_label)
    return manifest


def main():
    args = parse_args()
    dataset_base = Path(args.dataset_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ['manifests', 'detection', 'segmentation', 'gt_comparison', 'alignment', 'correspondence', 'classification', 'metrics']:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    if args.manifest_csv:
        manifest = ensure_binary_manifest(pd.read_csv(args.manifest_csv))
        if args.limit:
            manifest = manifest.iloc[: args.limit].copy()
        (output_dir / 'manifests' / 'paired_kaggle_manifest.csv').write_text(manifest.to_csv(index=False))
    else:
        manifest = build_kaggle_manifest(dataset_base, output_dir / 'manifests', pair_limit=args.limit)
        if args.limit:
            manifest = manifest.iloc[: args.limit].copy()

    classifier = BiradsClassifier(args.classifier_checkpoint, device=args.device)
    per_sample = []
    y_true = []
    y_pred = []

    for _, row in manifest.iterrows():
        side = str(row['breast_side']).strip().upper()
        sample_id = str(row['sample_id'])

        cc_raw_original = load_grayscale_image(row['cc_image_path'], flip_right=side == 'RIGHT')
        mlo_raw_original = load_grayscale_image(row['mlo_image_path'], flip_right=side == 'RIGHT')
        cc_raw = resize_long_side(cc_raw_original, args.max_long_side)
        mlo_raw = resize_long_side(mlo_raw_original, args.max_long_side)
        cc_gt_mask = load_roi_union(row['cc_roi_entries'], side, cc_raw, cc_raw_original.shape)
        mlo_gt_mask = load_roi_union(row['mlo_roi_entries'], side, mlo_raw, mlo_raw_original.shape)

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

        mlo_seg_resized = resize(
            mlo_seg.mask.astype(float),
            cc_seg.mask.shape,
            preserve_range=True,
            order=0,
            anti_aliasing=False,
        ) > 0.5
        aligned_mlo_mask = shift(
            mlo_seg_resized.astype(float),
            shift=(alignment.affine_matrix[1][2], alignment.affine_matrix[0][2]),
            order=0,
            mode='constant',
            cval=0.0,
        ) > 0.5

        cls = classifier.predict(
            cc_img,
            alignment.aligned_image,
            cc_seg.mask,
            aligned_mlo_mask.astype('uint8'),
            cc_det.bbox_xywh,
            mlo_det.bbox_xywh,
        )

        true_binary = int(row['binary_label'])
        pred_binary = int(cls.predicted_binary_label)
        y_true.append(true_binary)
        y_pred.append(pred_binary)

        cc_det_mask = bbox_to_mask(cc_det.bbox_xywh, cc_seg.mask.shape)
        mlo_det_mask = bbox_to_mask(mlo_det.bbox_xywh, mlo_seg.mask.shape)
        cc_precision, cc_recall = precision_recall(cc_seg.mask, cc_gt_mask)
        mlo_precision, mlo_recall = precision_recall(mlo_seg.mask, mlo_gt_mask)

        sample_metrics = {
            'sample_id': sample_id,
            'patient_id': row['patient_id'],
            'true_birads': int(row['birads_label']),
            'true_binary_label': true_binary,
            'predicted_birads': int(cls.predicted_birads),
            'predicted_binary_label': pred_binary,
            'predicted_binary_name': LABEL_NAMES[pred_binary],
            'classifier_source': cls.source,
            'cc_seg_dice_vs_roi': dice_score(cc_seg.mask, cc_gt_mask),
            'cc_seg_iou_vs_roi': iou_score(cc_seg.mask, cc_gt_mask),
            'cc_seg_precision_vs_roi': cc_precision,
            'cc_seg_recall_vs_roi': cc_recall,
            'mlo_seg_dice_vs_roi': dice_score(mlo_seg.mask, mlo_gt_mask),
            'mlo_seg_iou_vs_roi': iou_score(mlo_seg.mask, mlo_gt_mask),
            'mlo_seg_precision_vs_roi': mlo_precision,
            'mlo_seg_recall_vs_roi': mlo_recall,
            'cc_det_seg_iou': iou_score(cc_det_mask, cc_seg.mask),
            'mlo_det_seg_iou': iou_score(mlo_det_mask, mlo_seg.mask),
            'cc_detection_confidence': float(cc_det.confidence),
            'mlo_detection_confidence': float(mlo_det.confidence),
            'correspondence_centroid_distance_px': float(correspondence.centroid_distance_px),
            'alignment_dice_before': dice_score(cc_seg.mask, mlo_seg_resized),
            'alignment_dice_after': dice_score(cc_seg.mask, aligned_mlo_mask),
            'alignment_centroid_distance_before': center_distance(cc_seg.mask, mlo_seg_resized),
            'alignment_centroid_distance_after': center_distance(cc_seg.mask, aligned_mlo_mask),
            'ordinal_scores': json.dumps(cls.ordinal_scores),
        }
        per_sample.append(sample_metrics)

        overlay_detection(cc_img, cc_det.bbox_xywh, output_dir / 'detection' / f'{sample_id}_CC_detection.png')
        overlay_detection(mlo_img, mlo_det.bbox_xywh, output_dir / 'detection' / f'{sample_id}_MLO_detection.png')
        overlay_segmentation(cc_img, cc_seg.mask, output_dir / 'segmentation' / f'{sample_id}_CC_segmentation.png')
        overlay_segmentation(mlo_img, mlo_seg.mask, output_dir / 'segmentation' / f'{sample_id}_MLO_segmentation.png')
        overlay_gt_comparison(cc_img, cc_seg.mask, cc_gt_mask, output_dir / 'gt_comparison' / f'{sample_id}_CC_gt_comparison.png')
        overlay_gt_comparison(mlo_img, mlo_seg.mask, mlo_gt_mask, output_dir / 'gt_comparison' / f'{sample_id}_MLO_gt_comparison.png')
        save_alignment_overlay(cc_img, alignment.aligned_image, output_dir / 'alignment' / f'{sample_id}_alignment_overlay.png')
        (output_dir / 'correspondence' / f'{sample_id}_correspondence.json').write_text(json.dumps(correspondence.to_dict(), indent=2))
        (output_dir / 'classification' / f'{sample_id}_classification.json').write_text(
            json.dumps(
                {
                    'sample_id': sample_id,
                    'true_birads': int(row['birads_label']),
                    'true_binary_label': true_binary,
                    'true_binary_name': LABEL_NAMES[true_binary],
                    'predicted_birads': int(cls.predicted_birads),
                    'predicted_binary_label': pred_binary,
                    'predicted_binary_name': LABEL_NAMES[pred_binary],
                    'ordinal_scores': cls.ordinal_scores,
                    'source': cls.source,
                },
                indent=2,
            )
        )

    per_sample_df = pd.DataFrame(per_sample)
    per_sample_df.to_csv(output_dir / 'metrics' / 'per_sample_metrics.csv', index=False)

    cls_metrics = compute_binary_classification_metrics(y_true, y_pred)
    aggregate = {
        'dataset_base': str(dataset_base),
        'classification_target': 'binary_birads_0_3_vs_4_5',
        'classification_report': cls_metrics,
        'mean_cc_seg_dice_vs_roi': float(per_sample_df['cc_seg_dice_vs_roi'].mean()) if not per_sample_df.empty else None,
        'mean_mlo_seg_dice_vs_roi': float(per_sample_df['mlo_seg_dice_vs_roi'].mean()) if not per_sample_df.empty else None,
        'mean_cc_det_seg_iou': float(per_sample_df['cc_det_seg_iou'].mean()) if not per_sample_df.empty else None,
        'mean_mlo_det_seg_iou': float(per_sample_df['mlo_det_seg_iou'].mean()) if not per_sample_df.empty else None,
        'mean_alignment_dice_before': float(per_sample_df['alignment_dice_before'].mean()) if not per_sample_df.empty else None,
        'mean_alignment_dice_after': float(per_sample_df['alignment_dice_after'].mean()) if not per_sample_df.empty else None,
        'mean_alignment_centroid_distance_before': float(per_sample_df['alignment_centroid_distance_before'].dropna().mean()) if not per_sample_df.empty else None,
        'mean_alignment_centroid_distance_after': float(per_sample_df['alignment_centroid_distance_after'].dropna().mean()) if not per_sample_df.empty else None,
    }
    (output_dir / 'metrics' / 'aggregate_metrics.json').write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
