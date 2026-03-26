import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


class SaliencyTensorDataset(Dataset):
    def __init__(self, records, augment=False):
        self.data = records
        self.augment = augment
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tensor = torch.load(item["fused_tensor_path"])
        label = int(item["birads_label"])
        if self.augment:
            tensor = self.transforms(tensor)
        return tensor, label


class CNNCrossAttentionGrader(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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

    def forward(self, dual_tensor):
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


def ordinal_loss(predictions, targets):
    num_classes = predictions.size(1) + 1
    levels = torch.arange(num_classes - 1).to(predictions.device)
    binary_labels = (targets.view(-1, 1) > levels).float()
    return nn.BCEWithLogitsLoss()(predictions, binary_labels)


def get_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for tensors, labels in loader:
            logits = model(tensors.to(device))
            preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=range(6))
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return cm, f1_w


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_OUT_DIR = BASE_DIR / "output"
    SEED = int(os.getenv("PIPELINE_SEED", "42"))
    NUM_EPOCHS = int(os.getenv("GRADER_EPOCHS", "20"))

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.getenv("RUN_OUT_DIR", str(DEFAULT_OUT_DIR)))
    meta_path = out_dir / "saliency_metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot find {meta_path}.")

    use_augment = os.getenv("DATA_AUGMENT", "True") == "True"
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    if not metadata or "split" not in metadata[0]:
        raise ValueError("Metadata is missing split assignments. Regenerate tensors with script 1.")

    train_records = [item for item in metadata if item["split"] == "train"]
    val_records = [item for item in metadata if item["split"] == "val"]
    test_records = [item for item in metadata if item["split"] == "test"]

    if not train_records or not val_records or not test_records:
        raise ValueError("Train/val/test splits are incomplete in saliency metadata.")

    train_ds = SaliencyTensorDataset(train_records, augment=use_augment)
    val_ds = SaliencyTensorDataset(val_records, augment=False)
    test_ds = SaliencyTensorDataset(test_records, augment=False)

    train_labels = [int(item["birads_label"]) for item in train_records]
    class_sample_count = np.array([train_labels.count(t) for t in range(6)])
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    samples_weight = np.array([1.0 / class_sample_count[t] for t in train_labels])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, drop_last=len(train_ds) >= 16)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = CNNCrossAttentionGrader().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print(f"--- Training CNN (Augment: {use_augment}) ---")
    best_val_f1 = -1.0
    best_model_path = out_dir / "cnn_attentional_weights.pth"

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for tensors, labels in train_loader:
            optimizer.zero_grad()
            logits = model(tensors.to(device))
            loss = ordinal_loss(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        _, val_f1 = get_metrics(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {train_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    cm, f1_w = get_metrics(model, test_loader, device)

    metrics_dict = {
        "F1_Weighted": float(f1_w),
        "Best_Val_F1": float(best_val_f1),
        "Confusion_Matrix": cm.tolist(),
        "Seed": SEED,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print("Run data saved successfully.")
