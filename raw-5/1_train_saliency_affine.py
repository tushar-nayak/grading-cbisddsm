import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class DualViewMammogramDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        cc_path = str(row["cc_image_path"])
        mlo_path = str(row["mlo_image_path"])
        label = int(row["birads_label"])
        side = str(row["breast_side"]).strip().upper()
        sample_id = str(row["sample_id"]) if "sample_id" in row else self._build_sample_id(row)
        patient_id = str(row["patient_id"]) if "patient_id" in row and pd.notna(row["patient_id"]) else ""

        cc_image = Image.open(cc_path).convert("L")
        mlo_image = Image.open(mlo_path).convert("L")

        if side == "RIGHT":
            cc_image = TF.hflip(cc_image)
            mlo_image = TF.hflip(mlo_image)

        cc_tensor = self.transform(cc_image)
        mlo_tensor = self.transform(mlo_image)

        return (
            cc_tensor,
            mlo_tensor,
            torch.tensor(label, dtype=torch.long),
            cc_path,
            mlo_path,
            sample_id,
            patient_id,
        )

    @staticmethod
    def _build_sample_id(row):
        key = f"{row['cc_image_path']}::{row['mlo_image_path']}::{row['breast_side']}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


class RawSpatialTransformer(nn.Module):
    def __init__(self, input_channels=1):
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

    def forward(self, cc_view, mlo_view):
        x = torch.cat([cc_view, mlo_view], dim=1)
        xs = self.localization(x).view(-1, 32 * 7 * 7)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        theta_constrained = theta.clone()
        theta_constrained[:, 0, 0] = torch.clamp(theta[:, 0, 0], min=0.8, max=1.2)
        theta_constrained[:, 1, 1] = torch.clamp(theta[:, 1, 1], min=0.8, max=1.2)

        grid = F.affine_grid(theta_constrained, mlo_view.size(), align_corners=True)
        aligned_mlo = F.grid_sample(mlo_view, grid, align_corners=True)
        return aligned_mlo, theta_constrained


class SaliencyWeightedNCC(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.threshold = float(os.getenv("SALIENCY_THRESHOLD", "0.3"))
        self.weight = float(os.getenv("SALIENCY_WEIGHT", "10.0"))

    def forward(self, I, J):
        I_mean = torch.mean(I, dim=[2, 3], keepdim=True)
        J_mean = torch.mean(J, dim=[2, 3], keepdim=True)
        I_centered, J_centered = I - I_mean, J - J_mean

        cross = I_centered * J_centered
        I_var, J_var = I_centered ** 2, J_centered ** 2

        I_norm = (I - I.amin(dim=(2, 3), keepdim=True)) / (
            I.amax(dim=(2, 3), keepdim=True) - I.amin(dim=(2, 3), keepdim=True) + self.eps
        )
        saliency_mask = (I_norm > self.threshold).float()
        weight_map = 1.0 + (saliency_mask * self.weight)

        w_cross = torch.sum(cross * weight_map, dim=[2, 3])
        w_I_var = torch.sum(I_var * weight_map, dim=[2, 3])
        w_J_var = torch.sum(J_var * weight_map, dim=[2, 3])

        ncc = w_cross / (torch.sqrt(w_I_var * w_J_var) + self.eps)
        return 1 - torch.mean(ncc)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split_indices(labels, seed, train_ratio=0.7, val_ratio=0.15):
    rng = random.Random(seed)
    by_class = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        n_items = len(class_indices)
        if n_items < 3:
            train_idx.extend(class_indices)
            continue

        n_train = max(1, int(round(n_items * train_ratio)))
        n_train = min(n_train, n_items - 2)
        remaining = n_items - n_train

        n_val = int(round(n_items * val_ratio))
        n_val = max(1, min(n_val, remaining - 1))
        n_test = n_items - n_train - n_val

        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        train_idx.extend(class_indices[:n_train])
        val_idx.extend(class_indices[n_train:n_train + n_val])
        test_idx.extend(class_indices[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def build_split_lookup(train_idx, val_idx, test_idx):
    return {
        "train": set(train_idx),
        "val": set(val_idx),
        "test": set(test_idx),
    }


def split_name_for_index(idx, split_lookup):
    if idx in split_lookup["train"]:
        return "train"
    if idx in split_lookup["val"]:
        return "val"
    if idx in split_lookup["test"]:
        return "test"
    raise KeyError(f"Missing split assignment for index {idx}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = Path(os.getenv("PIPELINE_CSV_PATH", str(BASE_DIR / "dicom_clean_train.csv")))
    DEFAULT_OUT_DIR = BASE_DIR / "output"
    SEED = int(os.getenv("PIPELINE_SEED", "42"))
    NUM_EPOCHS = int(os.getenv("STN_EPOCHS", "5"))

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.getenv("RUN_OUT_DIR", str(DEFAULT_OUT_DIR)))
    tensor_dir = out_dir / "fused_saliency_tensors"
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    dataset = DualViewMammogramDataset(csv_file=CSV_PATH)
    labels = dataset.data_frame["birads_label"].astype(int).tolist()
    train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=SEED)
    split_lookup = build_split_lookup(train_idx, val_idx, test_idx)

    train_dataset = Subset(dataset, train_idx)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = RawSpatialTransformer().to(device)
    criterion = SaliencyWeightedNCC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("--- Training Saliency-Weighted Affine STN ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for cc, mlo, _, _, _, _, _ in train_loader:
            cc, mlo = cc.to(device), mlo.to(device)
            optimizer.zero_grad()
            aligned_mlo, _ = model(cc, mlo)
            loss = criterion(cc, aligned_mlo)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Saliency NCC Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    save_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    meta = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(save_loader):
            cc, mlo, labels, cc_paths, mlo_paths, sample_ids, patient_ids = batch
            cc, mlo = cc.to(device), mlo.to(device)
            aligned_mlo, theta = model(cc, mlo)
            fused = torch.cat([cc, aligned_mlo], dim=1)

            for i in range(len(labels)):
                global_idx = batch_idx * save_loader.batch_size + i
                sample_id = str(sample_ids[i])
                tensor_path = tensor_dir / f"{sample_id}_saliency.pt"
                torch.save(fused[i].cpu(), tensor_path)
                meta.append({
                    "original_cc_path": cc_paths[i],
                    "original_mlo_path": mlo_paths[i],
                    "fused_tensor_path": str(tensor_path),
                    "sample_id": sample_id,
                    "patient_id": str(patient_ids[i]),
                    "split": split_name_for_index(global_idx, split_lookup),
                    "birads_label": labels[i].item(),
                    "transformation_matrix": theta[i].cpu().numpy().tolist(),
                })

    with open(out_dir / "saliency_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    print("Saliency tensors saved.")
