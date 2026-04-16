import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import load_grayscale_image
from localizer import AttentionUNet, default_localizer_checkpoint
from run_kaggle_full_pipeline import build_kaggle_manifest, load_roi_union

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SampleRecord:
    sample_id: str
    view: str
    image_path: str
    breast_side: str
    roi_entries: str


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1e-6,
        bce_weight: float = 0.35,
        ft_weight: float = 0.65,
        pos_weight: float = 8.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.ft_weight = ft_weight
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        pf = probs.reshape(-1)
        tf = targets.reshape(-1)

        tp = (pf * tf).sum()
        fp = (pf * (1.0 - tf)).sum()
        fn = ((1.0 - pf) * tf).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        focal_tversky = (1.0 - tversky) ** (1.0 / self.gamma)

        pos_weight = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        return self.ft_weight * focal_tversky + self.bce_weight * bce


def compute_binary_dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def build_records(manifest: pd.DataFrame) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for _, row in manifest.iterrows():
        for view in ["cc", "mlo"]:
            records.append(
                SampleRecord(
                    sample_id=str(row["sample_id"]),
                    view=view.upper(),
                    image_path=str(row[f"{view}_image_path"]),
                    breast_side=str(row["breast_side"]).strip().upper(),
                    roi_entries=str(row[f"{view}_roi_entries"]),
                )
            )
    return records


def split_records(records: list[SampleRecord], seed: int = 42) -> tuple[list[SampleRecord], list[SampleRecord]]:
    grouped: dict[str, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.sample_id, []).append(record)

    sample_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    val_count = max(1, int(round(0.1 * len(sample_ids))))
    val_ids = set(sample_ids[:val_count])

    train_records = [record for record in records if record.sample_id not in val_ids]
    val_records = [record for record in records if record.sample_id in val_ids]
    return train_records, val_records


class ViewLocalizerDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        cache_dir: str | Path,
        input_size: int = 512,
        augment: bool = False,
        lesion_pad: int = 60,
        cache_version: str = "v7_smallunet_context_crop_curriculum",
        context_scale_range: tuple[float, float] = (2.5, 4.0),
        shift_frac: float = 0.18,
    ):
        self.records = records
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.input_size = input_size
        self.augment = augment
        self.lesion_pad = lesion_pad
        self.cache_version = cache_version
        self.context_scale_range = context_scale_range
        self.shift_frac = shift_frac

    def __len__(self) -> int:
        return len(self.records)

    def _cache_key(self, record: SampleRecord) -> str:
        payload = "||".join(
            [
                self.cache_version,
                record.sample_id,
                record.view,
                record.image_path,
                record.breast_side,
                record.roi_entries,
                str(self.input_size),
                str(int(self.augment)),
                str(self.context_scale_range),
                str(self.shift_frac),
            ]
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, record: SampleRecord) -> Path:
        return self.cache_dir / f"{self._cache_key(record)}.npz"

    def _load_raw_image(self, image_path: str, breast_side: str) -> np.ndarray:
        flip_right = breast_side == "RIGHT"
        raw = load_grayscale_image(image_path, flip_right=flip_right).astype(np.float32)

        if raw.max() > raw.min():
            raw = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            raw = np.zeros_like(raw, dtype=np.float32)

        return raw

    def _extract_breast_mask(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        active = img[img > 0]
        if active.size == 0:
            return np.zeros_like(img, dtype=np.uint8)

        thr = max(0.02, float(np.percentile(active, 5.0)))
        mask = (img > thr).astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask.astype(np.uint8)

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8)

    def _percentile_normalize(
        self,
        image: np.ndarray,
        lower_q: float = 1.0,
        upper_q: float = 99.0,
    ) -> np.ndarray:
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

    def _apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 3.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        out = clahe.apply(img_u8).astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _mask_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return None
        return int(ymin := ys.min()), int(ymax := ys.max()), int(xmin := xs.min()), int(xmax := xs.max())

    def _crop_with_box(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))
        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    def _context_crop_from_gt(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        breast_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lesion_box = self._mask_bbox(gt_mask)
        if lesion_box is None:
            ys, xs = np.where(breast_mask > 0)
            if len(ys) == 0 or len(xs) == 0:
                return image, gt_mask
            return self._crop_with_box(
                image,
                gt_mask,
                int(ys.min()),
                int(ys.max()) + 1,
                int(xs.min()),
                int(xs.max()) + 1,
            )

        ly0, ly1, lx0, lx1 = lesion_box
        lesion_h = max(ly1 - ly0 + 1, 8)
        lesion_w = max(lx1 - lx0 + 1, 8)
        cy = 0.5 * (ly0 + ly1)
        cx = 0.5 * (lx0 + lx1)

        scale = random.uniform(*self.context_scale_range) if self.augment else np.mean(self.context_scale_range)
        crop_h = max(int(round(lesion_h * scale)), 96)
        crop_w = max(int(round(lesion_w * scale)), 96)

        if self.augment:
            shift_y = random.uniform(-self.shift_frac, self.shift_frac) * crop_h
            shift_x = random.uniform(-self.shift_frac, self.shift_frac) * crop_w
            cy += shift_y
            cx += shift_x

        y0 = int(round(cy - crop_h / 2))
        y1 = int(round(cy + crop_h / 2))
        x0 = int(round(cx - crop_w / 2))
        x1 = int(round(cx + crop_w / 2))

        bys, bxs = np.where(breast_mask > 0)
        if len(bys) > 0 and len(bxs) > 0:
            breast_y0, breast_y1 = int(bys.min()), int(bys.max()) + 1
            breast_x0, breast_x1 = int(bxs.min()), int(bxs.max()) + 1

            # keep crop mostly inside the breast extent
            y0 = max(y0, breast_y0)
            x0 = max(x0, breast_x0)
            y1 = min(y1, breast_y1)
            x1 = min(x1, breast_x1)

            # if clipping made it too small, expand back a bit
            if (y1 - y0) < 64:
                pad = (64 - (y1 - y0)) // 2 + 2
                y0 = max(breast_y0, y0 - pad)
                y1 = min(breast_y1, y1 + pad)
            if (x1 - x0) < 64:
                pad = (64 - (x1 - x0)) // 2 + 2
                x0 = max(breast_x0, x0 - pad)
                x1 = min(breast_x1, x1 + pad)

        return self._crop_with_box(image, gt_mask, y0, y1, x0, x1)

    def _resize_pair(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_r = cv2.resize(image.astype(np.float32), (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask.astype(np.uint8), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        return image_r.astype(np.float32), (mask_r > 0).astype(np.uint8)

    def _build_cached_sample(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
        raw_img = self._load_raw_image(record.image_path, record.breast_side)

        gt_mask = load_roi_union(
            record.roi_entries,
            record.breast_side,
            raw_img,
            raw_img.shape,
        ).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        if gt_mask.shape != raw_img.shape:
            gt_mask = cv2.resize(
                gt_mask,
                (raw_img.shape[1], raw_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)

        breast_mask = self._extract_breast_mask(raw_img)

        cropped_img, cropped_gt = self._context_crop_from_gt(raw_img, gt_mask, breast_mask)

        cropped_img = self._percentile_normalize(cropped_img)
        cropped_img = self._apply_clahe(cropped_img, clip_limit=3.0, tile_grid_size=(8, 8))

        image_f, mask_f = self._resize_pair(cropped_img, cropped_gt)

        if mask_f.sum() < 8:
            # safer fallback: slightly tighter crop around lesion without jitter
            old_augment = self.augment
            self.augment = False
            fallback_img, fallback_gt = self._context_crop_from_gt(raw_img, gt_mask, breast_mask)
            self.augment = old_augment

            fallback_img = self._percentile_normalize(fallback_img)
            fallback_img = self._apply_clahe(fallback_img, clip_limit=3.0, tile_grid_size=(8, 8))
            image_f, mask_f = self._resize_pair(fallback_img, fallback_gt)

        return image_f.astype(np.float32), mask_f.astype(np.uint8)

    def _load_or_create_cached_sample(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
        cache_path = self._cache_path(record)
        if cache_path.exists():
            arr = np.load(cache_path)
            image = arr["image"].astype(np.float32)
            mask = arr["mask"].astype(np.uint8)
            return image, mask

        image, mask = self._build_cached_sample(record)
        np.savez_compressed(cache_path, image=image, mask=mask)
        return image, mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() < 0.35:
            gain = random.uniform(0.90, 1.10)
            bias = random.uniform(-0.04, 0.04)
            image = np.clip(image * gain + bias, 0.0, 1.0)

        if random.random() < 0.25:
            noise = np.random.normal(0.0, 0.012, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        if random.random() < 0.20:
            sigma = random.uniform(0.3, 0.8)
            k = int(2 * round(3 * sigma) + 1)
            image = cv2.GaussianBlur(image, (k, k), sigmaX=sigma)

        return image, mask

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image, mask = self._load_or_create_cached_sample(record)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask.astype(np.float32), dtype=torch.float32).unsqueeze(0)
        return {"image": image, "mask": mask}

def warmup_cache(dataset: ViewLocalizerDataset, desc: str) -> None:
    for i in tqdm(range(len(dataset)), desc=desc):
        record = dataset.records[i]
        dataset._load_or_create_cached_sample(record)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []

    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    dice_by_threshold = {thr: [] for thr in thresholds}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits)

            for thr in thresholds:
                preds = (probs >= thr).float()
                dice = compute_binary_dice(preds, masks).item()
                dice_by_threshold[thr].append(float(dice))

    mean_dice_by_threshold = {
        str(thr): (float(np.mean(vals)) if vals else 0.0)
        for thr, vals in dice_by_threshold.items()
    }

    best_thr = max(thresholds, key=lambda t: np.mean(dice_by_threshold[t]) if dice_by_threshold[t] else -1.0)
    best_dice = float(np.mean(dice_by_threshold[best_thr])) if dice_by_threshold[best_thr] else 0.0

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "dice": best_dice,
        "best_threshold": float(best_thr),
        "dice_by_threshold": mean_dice_by_threshold,
    }


def parse_args():
    root = Path(__file__).resolve().parents[1]
    base_dir = root / "dataset" / "raw" / "cbisddsm-kaggle"
    parser = argparse.ArgumentParser(description="Train a lightweight lesion localizer on the CBIS-DDSM Kaggle manifest")

    parser.add_argument("--dataset-base", default=str(base_dir))
    parser.add_argument("--manifest-dir", default=str(Path(__file__).resolve().parent / "localizer_training_manifest"))
    parser.add_argument("--checkpoint", default=default_localizer_checkpoint())
    parser.add_argument("--cache-dir", default=str(root / "output" / "localizer_cache"))

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-cache", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--lesion-pad", type=int, default=60)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    return parser.parse_args()


def main():
    print("Starting main...")
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("Building manifest...")
    manifest = build_kaggle_manifest(Path(args.dataset_base), manifest_dir)

    records = build_records(manifest)
    train_records, val_records = split_records(records, seed=args.seed)

    print(f"Total records: {len(records)}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")

    print("Getting device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    print("Building datasets...")
    train_ds = ViewLocalizerDataset(
        train_records,
        cache_dir=Path(args.cache_dir) / "train",
        input_size=args.input_size,
        augment=not args.no_augment,
        lesion_pad=args.lesion_pad,
    )
    val_ds = ViewLocalizerDataset(
        val_records,
        cache_dir=Path(args.cache_dir) / "val",
        input_size=args.input_size,
        augment=False,
        lesion_pad=args.lesion_pad,
    )

    if args.warmup_cache:
        print("Warming cache for training set...")
        warmup_cache(train_ds, desc="Caching train")
        print("Warming cache for validation set...")
        warmup_cache(val_ds, desc="Caching val")

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

    print("Building model...")
    model = AttentionUNet().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    criterion = FocalTverskyLoss()
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_dice = -1.0
    history = []

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_path.with_name(checkpoint_path.stem + "_last" + checkpoint_path.suffix)

    print("Training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        train_losses = []

        progress = tqdm(train_loader, desc=f"Train {epoch + 1}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device, criterion=criterion)

        current_lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "best_threshold": val_metrics["best_threshold"],
            "dice_by_threshold": val_metrics["dice_by_threshold"],
            "lr": current_lr,
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        torch.save(
            {
                "model_state": model.state_dict(),
                "history": history,
                "best_val_dice": best_val_dice,
                "input_size": args.input_size,
                "best_threshold": float(val_metrics["best_threshold"]),
            },
            last_checkpoint_path,
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "history": history,
                    "best_val_dice": best_val_dice,
                    "input_size": args.input_size,
                    "best_threshold": float(val_metrics["best_threshold"]),
                },
                checkpoint_path,
            )

        scheduler.step()


        metrics_path = checkpoint_path.with_suffix(".json")
        metrics_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_val_dice": best_val_dice,
                "best_threshold": history[-1]["best_threshold"] if history else None,
            },
            indent=2,
        )
    )

    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_last_checkpoint={last_checkpoint_path}")
    print(f"best_val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()
