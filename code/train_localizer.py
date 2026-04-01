import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from data import extract_breast_mask, load_grayscale_image, preprocess_mammogram
from localizer import SmallUNet, default_localizer_checkpoint
from run_kaggle_full_pipeline import build_kaggle_manifest, load_roi_union, resize_long_side


@dataclass
class SampleRecord:
    sample_id: str
    view: str
    image_path: str
    breast_side: str
    roi_entries: str


class ViewLocalizerDataset(Dataset):
    def __init__(self, records: list[SampleRecord], max_long_side: int = 1536, input_size: int = 256):
        self.records = records
        self.max_long_side = max_long_side
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        flip_right = record.breast_side == "RIGHT"
        raw_orig = load_grayscale_image(record.image_path, flip_right=flip_right)
        raw = resize_long_side(raw_orig, self.max_long_side)
        image = preprocess_mammogram(raw)
        breast_mask = extract_breast_mask(image)
        gt_mask = load_roi_union(record.roi_entries, record.breast_side, raw, raw_orig.shape)

        masked = image * breast_mask
        pil = Image.fromarray(np.clip(masked * 255.0, 0, 255).astype(np.uint8))
        img_small = np.array(pil.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        if img_small.max() > img_small.min():
            img_small = (img_small - img_small.min()) / (img_small.max() - img_small.min())
        mask_small = np.array(
            Image.fromarray((gt_mask > 0).astype(np.uint8) * 255).resize((self.input_size, self.input_size), Image.Resampling.NEAREST),
            dtype=np.float32,
        ) / 255.0
        return {
            "image": torch.from_numpy(img_small).unsqueeze(0),
            "mask": torch.from_numpy(mask_small).unsqueeze(0),
        }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    dices = []
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss(logits, masks)
            losses.append(float(loss.item()))
            preds = (torch.sigmoid(logits) >= 0.35).float()
            inter = (preds * masks).sum(dim=(1, 2, 3))
            denom = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean().item()
            dices.append(float(dice))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
    }


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a lightweight lesion localizer on the CBIS-DDSM Kaggle manifest")
    parser.add_argument("--dataset-base", default=str(root / "dataset" / "raw" / "cbisddsm-kaggle"))
    parser.add_argument("--manifest-dir", default=str(Path(__file__).resolve().parent / "localizer_training_manifest"))
    parser.add_argument("--checkpoint", default=default_localizer_checkpoint())
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--max-long-side", type=int, default=1536)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_kaggle_manifest(Path(args.dataset_base), manifest_dir)
    records = build_records(manifest)
    train_records, val_records = split_records(records, seed=args.seed)

    train_ds = ViewLocalizerDataset(train_records, max_long_side=args.max_long_side, input_size=args.input_size)
    val_ds = ViewLocalizerDataset(val_records, max_long_side=args.max_long_side, input_size=args.input_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_dice = -1.0
    history = []
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
        }
        history.append(record)
        print(json.dumps(record))

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "history": history,
                    "best_val_dice": best_val_dice,
                    "input_size": args.input_size,
                    "max_long_side": args.max_long_side,
                },
                checkpoint_path,
            )

    metrics_path = checkpoint_path.with_suffix('.json')
    metrics_path.write_text(json.dumps({"history": history, "best_val_dice": best_val_dice}, indent=2))
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"best_val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()
