import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from data import extract_breast_mask, load_grayscale_image, preprocess_mammogram
from run_kaggle_full_pipeline import build_kaggle_manifest

warnings.filterwarnings("ignore")

LABEL_NAMES = {
    0: "birads_0_1_2_3",
    1: "birads_4_5",
}


@dataclass
class SampleRecord:
    sample_id: str
    patient_id: str
    side: str
    cc_image_path: str
    mlo_image_path: str
    birads_label: int
    binary_label: int


def birads_to_binary_label(birads: int) -> int:
    return 1 if int(birads) >= 4 else 0


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_binary_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if "binary_label" in manifest.columns:
        return manifest.copy()
    out = manifest.copy()
    out["binary_label"] = out["birads_label"].map(birads_to_binary_label)
    return out


def build_records(manifest: pd.DataFrame) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for _, row in manifest.iterrows():
        birads_label = int(row["birads_label"])
        records.append(
            SampleRecord(
                sample_id=str(row["sample_id"]),
                patient_id=str(row.get("patient_id", row["sample_id"])),
                side=str(row["breast_side"]).strip().upper(),
                cc_image_path=str(row["cc_image_path"]),
                mlo_image_path=str(row["mlo_image_path"]),
                birads_label=birads_label,
                binary_label=int(row.get("binary_label", birads_to_binary_label(birads_label))),
            )
        )
    return records


def split_records_by_patient(
    records: list[SampleRecord],
    seed: int,
    val_ratio: float,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if len(records) <= 1:
        return records, []

    patient_ids = sorted({r.patient_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    val_count = max(1, int(round(len(patient_ids) * val_ratio)))
    val_count = min(val_count, len(patient_ids) - 1)
    val_patients = set(patient_ids[:val_count])

    train_records = [r for r in records if r.patient_id not in val_patients]
    val_records = [r for r in records if r.patient_id in val_patients]

    if len(train_records) == 0 or len(val_records) == 0:
        rng.shuffle(records)
        val_count = max(1, int(round(len(records) * val_ratio)))
        val_count = min(val_count, len(records) - 1)
        val_records = records[:val_count]
        train_records = records[val_count:]

    return train_records, val_records


def resize_image(image: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = cv2.resize(image.astype(np.float32), (size, size), interpolation=interp)
    return out.astype(np.float32)


def crop_to_breast_bbox(image: np.ndarray, breast_mask: np.ndarray, pad_frac: float = 0.03) -> np.ndarray:
    ys, xs = np.where(breast_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        ys, xs = np.where(image > 0)
        if len(ys) == 0 or len(xs) == 0:
            return image

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    h, w = image.shape
    pad_y = max(4, int((y1 - y0 + 1) * pad_frac))
    pad_x = max(4, int((x1 - x0 + 1) * pad_frac))

    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y + 1)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x + 1)

    return image[y0:y1, x0:x1]


def percentile_normalize(image: np.ndarray, lower_q: float = 1.0, upper_q: float = 99.0) -> np.ndarray:
    active = image[image > 0]
    if active.size == 0:
        return image.astype(np.float32)

    lo = np.percentile(active, lower_q)
    hi = np.percentile(active, upper_q)
    if hi <= lo:
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    out = clahe.apply(img_u8).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def prepare_channel(image: np.ndarray, input_size: int, use_clahe: bool = True) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    else:
        image = np.zeros_like(image, dtype=np.float32)

    breast_mask = extract_breast_mask(image)
    image = crop_to_breast_bbox(image, breast_mask)
    image = percentile_normalize(image)
    if use_clahe:
        image = apply_clahe(image)
    image = resize_image(image, input_size, nearest=False)
    image = (image - 0.5) / 0.5
    return image.astype(np.float32)


class BinaryCrossAttentionDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        input_size: int,
        augment: bool,
    ):
        self.records = records
        self.input_size = input_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _load_pair(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
        flip_right = record.side == "RIGHT"
        cc_raw = load_grayscale_image(record.cc_image_path, flip_right=flip_right)
        mlo_raw = load_grayscale_image(record.mlo_image_path, flip_right=flip_right)

        cc_img = preprocess_mammogram(cc_raw)
        mlo_img = preprocess_mammogram(mlo_raw)

        cc_channel = prepare_channel(cc_img, self.input_size, use_clahe=True)
        mlo_channel = prepare_channel(mlo_img, self.input_size, use_clahe=True)
        return cc_channel, mlo_channel

    def _augment_pair(self, cc: np.ndarray, mlo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return cc, mlo

        if random.random() < 0.5:
            cc = np.fliplr(cc).copy()
            mlo = np.fliplr(mlo).copy()

        if random.random() < 0.3:
            gain = random.uniform(0.95, 1.05)
            bias = random.uniform(-0.03, 0.03)
            cc = np.clip(cc * gain + bias, -1.0, 1.0)
            mlo = np.clip(mlo * gain + bias, -1.0, 1.0)

        if random.random() < 0.25:
            noise_cc = np.random.normal(0.0, 0.01, cc.shape).astype(np.float32)
            noise_mlo = np.random.normal(0.0, 0.01, mlo.shape).astype(np.float32)
            cc = np.clip(cc + noise_cc, -1.0, 1.0)
            mlo = np.clip(mlo + noise_mlo, -1.0, 1.0)

        return cc, mlo

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        cc_channel, mlo_channel = self._load_pair(record)
        cc_channel, mlo_channel = self._augment_pair(cc_channel, mlo_channel)

        stacked = np.stack([cc_channel, mlo_channel], axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(stacked),
            "label": torch.tensor(float(record.binary_label), dtype=torch.float32),
            "sample_id": record.sample_id,
            "patient_id": record.patient_id,
            "birads_label": torch.tensor(record.birads_label, dtype=torch.int64),
        }


class SymmetricCrossAttentionBinaryGrader(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2
        resnet = resnet50(weights=weights)
        old_weight = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = old_weight.mean(dim=1, keepdim=True)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        embed_dim = 2048

        self.cc_to_mlo_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.mlo_to_cc_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, 1),
        )

    def forward(self, dual_tensor: torch.Tensor) -> torch.Tensor:
        cc_img = dual_tensor[:, 0:1, :, :]
        mlo_img = dual_tensor[:, 1:2, :, :]

        cc_feat = self.feature_extractor(cc_img)
        mlo_feat = self.feature_extractor(mlo_img)

        batch_size, channels, height, width = cc_feat.shape

        cc_seq = cc_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        mlo_seq = mlo_feat.view(batch_size, channels, -1).permute(0, 2, 1)

        cc_attended_seq, _ = self.cc_to_mlo_attention(query=cc_seq, key=mlo_seq, value=mlo_seq)
        mlo_attended_seq, _ = self.mlo_to_cc_attention(query=mlo_seq, key=cc_seq, value=cc_seq)

        cc_attended_feat = cc_attended_seq.permute(0, 2, 1).view(batch_size, channels, height, width)
        mlo_attended_feat = mlo_attended_seq.permute(0, 2, 1).view(batch_size, channels, height, width)

        cc_pooled = self.global_pool(cc_feat).view(batch_size, -1)
        mlo_pooled = self.global_pool(mlo_feat).view(batch_size, -1)
        cc_attended_pooled = self.global_pool(cc_attended_feat).view(batch_size, -1)
        mlo_attended_pooled = self.global_pool(mlo_attended_feat).view(batch_size, -1)

        fused = torch.cat([cc_pooled, mlo_pooled, cc_attended_pooled, mlo_attended_pooled], dim=1)
        logits = self.classifier(fused).squeeze(1)
        return logits


def compute_binary_classification_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict:
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
            "label_name": LABEL_NAMES[label],
            "support": int(support),
            "predicted_count": int(predicted_count),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "npv": npv,
            "f1": f1,
            "prevalence": prevalence,
        }

    roc_auc = None
    if len(set(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None

    return {
        "labels": labels,
        "label_names": [LABEL_NAMES[label] for label in labels],
        "confusion_matrix": matrix,
        "num_samples": int(total),
        "accuracy": safe_div(correct, total),
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
        "weighted_f1": safe_div(weighted_f1_sum, total),
        "roc_auc": roc_auc,
        "class_metrics": class_metrics,
    }


def threshold_sweep(y_true: list[int], y_prob: list[float]) -> dict:
    thresholds = [round(x, 2) for x in np.arange(0.2, 0.81, 0.02)]
    best_threshold = 0.5
    best_f1 = -1.0
    by_threshold = {}

    for thr in thresholds:
        y_pred = [1 if p >= thr else 0 for p in y_prob]
        score = f1_score(y_true, y_pred, zero_division=0)
        by_threshold[f"{thr:.2f}"] = float(score)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(thr)

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "f1_by_threshold": by_threshold,
    }


def evaluate(model, loader, device, criterion, threshold: float | None = None) -> dict:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_prob: list[float] = []
    sample_rows: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            labels = y.detach().cpu().numpy().astype(int).tolist()
            birads_labels = batch["birads_label"].cpu().numpy().astype(int).tolist()
            sample_ids = batch["sample_id"]
            patient_ids = batch["patient_id"]

            losses.append(float(loss.item()))
            y_true.extend(labels)
            y_prob.extend([float(p) for p in probs])

            for sid, pid, true_bin, birads_label, prob in zip(sample_ids, patient_ids, labels, birads_labels, probs):
                sample_rows.append(
                    {
                        "sample_id": sid,
                        "patient_id": pid,
                        "true_birads": int(birads_label),
                        "true_binary_label": int(true_bin),
                        "probability_malignant": float(prob),
                    }
                )

    sweep = threshold_sweep(y_true, y_prob)
    chosen_threshold = float(threshold if threshold is not None else sweep["best_threshold"])
    y_pred = [1 if p >= chosen_threshold else 0 for p in y_prob]
    metrics = compute_binary_classification_metrics(y_true, y_pred, y_prob)

    if len(set(y_true)) > 1:
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        pr_curve = {
            "precision": [float(x) for x in precision.tolist()],
            "recall": [float(x) for x in recall.tolist()],
            "thresholds": [float(x) for x in pr_thresholds.tolist()],
        }
    else:
        pr_curve = None

    for row, pred in zip(sample_rows, y_pred):
        row["predicted_binary_label"] = int(pred)
        row["predicted_binary_name"] = LABEL_NAMES[int(pred)]
        row["threshold_used"] = chosen_threshold

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "threshold": chosen_threshold,
        "threshold_search": sweep,
        "classification_report": metrics,
        "per_sample": sample_rows,
        "pr_curve": pr_curve,
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1] / "dataset" / "raw" / "cbisddsm-kaggle"
    parser = argparse.ArgumentParser(
        description="Train a binary BI-RADS classifier using ResNet50 + symmetric cross-attention on CC/MLO pairs, without segmentation or detection."
    )
    parser.add_argument("--dataset-base", default=str(base_dir))
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--output-dir", default=str(Path("output") / "binary_cross_attention_only"))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-full", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), output_dir / "run_config.json")

    if args.manifest_csv:
        manifest = pd.read_csv(args.manifest_csv)
        manifest = ensure_binary_manifest(manifest)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
        manifest = ensure_binary_manifest(build_kaggle_manifest(Path(args.dataset_base), output_dir / "manifests", pair_limit=args.limit))
        if args.limit:
            manifest = manifest.iloc[: args.limit].copy()

    records = build_records(manifest)
    train_records, val_records = split_records_by_patient(records, seed=args.seed, val_ratio=args.val_ratio)

    print(f"Total records: {len(records)}", flush=True)
    print(f"Train records: {len(train_records)}", flush=True)
    print(f"Val records: {len(val_records)}", flush=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)
    torch.backends.cudnn.benchmark = True
    train_ds = BinaryCrossAttentionDataset(train_records, input_size=args.input_size, augment=not args.no_augment)
    val_ds = BinaryCrossAttentionDataset(val_records, input_size=args.input_size, augment=False)

    pin = device.type == "cuda"
    persistent = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    model = SymmetricCrossAttentionBinaryGrader().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    pos_count = sum(r.binary_label for r in train_records)
    neg_count = max(0, len(train_records) - pos_count)
    pos_weight_value = max(1.0, neg_count / max(pos_count, 1))
    print(f"pos_weight={pos_weight_value:.4f}", flush=True)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=device.type == "cuda")

    history: list[dict] = []
    best_macro_f1 = -1.0
    best_threshold = 0.5
    best_checkpoint = checkpoints_dir / "best_cross_attention_only.pth"
    last_checkpoint = checkpoints_dir / "last_cross_attention_only.pth"

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            print("Unfreezing backbone.", flush=True)
            for param in model.feature_extractor.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_full, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - epoch), eta_min=1e-6
            )

        model.train()
        train_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch in progress:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device, criterion, threshold=None)
        scheduler.step()

        macro_f1 = val_metrics["classification_report"]["macro_f1"]
        best_threshold = float(val_metrics["threshold"])
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": val_metrics["loss"],
            "val_threshold": val_metrics["threshold"],
            "val_accuracy": val_metrics["classification_report"]["accuracy"],
            "val_balanced_accuracy": val_metrics["classification_report"]["balanced_accuracy"],
            "val_macro_f1": macro_f1,
            "val_weighted_f1": val_metrics["classification_report"]["weighted_f1"],
            "val_roc_auc": val_metrics["classification_report"]["roc_auc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

        checkpoint_payload = {
            "model_state": model.state_dict(),
            "history": history,
            "best_macro_f1": best_macro_f1,
            "best_threshold": best_threshold,
            "input_size": args.input_size,
            "label_names": LABEL_NAMES,
        }
        torch.save(checkpoint_payload, last_checkpoint)

        save_json(val_metrics, metrics_dir / f"val_epoch_{epoch + 1:02d}.json")
        pd.DataFrame(val_metrics["per_sample"]).to_csv(metrics_dir / f"val_epoch_{epoch + 1:02d}_per_sample.csv", index=False)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "history": history,
                    "best_macro_f1": best_macro_f1,
                    "best_threshold": best_threshold,
                    "input_size": args.input_size,
                    "label_names": LABEL_NAMES,
                },
                best_checkpoint,
            )
            save_json(val_metrics, metrics_dir / "best_val_metrics.json")
            pd.DataFrame(val_metrics["per_sample"]).to_csv(metrics_dir / "best_val_per_sample.csv", index=False)

    save_json({"history": history, "best_macro_f1": best_macro_f1}, metrics_dir / "training_history.json")

    print(f"saved_best_checkpoint={best_checkpoint}", flush=True)
    print(f"saved_last_checkpoint={last_checkpoint}", flush=True)
    print(f"best_macro_f1={best_macro_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
