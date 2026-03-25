import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import models, transforms
import json
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

# ==========================================
# 1. Dataset Loader 
# ==========================================
class SaliencyTensorDataset(Dataset):
    def __init__(self, metadata_json, augment=False):
        with open(metadata_json, 'r') as f:
            self.data = json.load(f)
        self.augment = augment
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10)
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tensor = torch.load(item['fused_tensor_path']) 
        label = int(item['birads_label'])
        if self.augment: 
            tensor = self.transforms(tensor)
        return tensor, label

# ==========================================
# 2. The CNN Cross-Attention Grader
# ==========================================
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
            nn.Linear(512, num_classes - 1)
        )

    def forward(self, dual_tensor):
        cc_img = dual_tensor[:, 0:1, :, :] 
        mlo_img = dual_tensor[:, 1:2, :, :]
        cc_feat = self.feature_extractor(cc_img)   
        mlo_feat = self.feature_extractor(mlo_img) 
        B, C, H, W = cc_feat.shape
        cc_seq = cc_feat.view(B, C, -1).permute(0, 2, 1) 
        mlo_seq = mlo_feat.view(B, C, -1).permute(0, 2, 1)
        attended_mlo_seq, _ = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq)
        attended_mlo_feat = attended_mlo_seq.permute(0, 2, 1).view(B, C, H, W)
        cc_pooled = self.global_pool(cc_feat).view(B, -1)
        mlo_pooled = self.global_pool(attended_mlo_feat).view(B, -1)
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
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return cm, f1_w

# ==========================================
# 4. Training Loop (FIXED FOR ABLATION)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Listen for current experiment folder
    out_dir = os.getenv("RUN_OUT_DIR", "/home/sofa/host_dir/spatial_alignment/raw-4/output")
    meta_path = os.path.join(out_dir, "saliency_metadata.json")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find {meta_path}.")
    
    # Check if we should use augmentation
    use_augment = os.getenv("DATA_AUGMENT", "True") == "True"
    
    full_dataset = SaliencyTensorDataset(meta_path, augment=use_augment)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    # Balanced Sampling
    train_labels = [int(full_dataset.data[i]['birads_label']) for i in train_ds.indices]
    class_sample_count = np.array([train_labels.count(t) for t in range(6)])
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    samples_weight = np.array([1. / class_sample_count[t] for t in train_labels])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = CNNCrossAttentionGrader().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 

    print(f"--- Training CNN (Augment: {use_augment}) ---")
    for epoch in range(20):
        model.train()
        train_loss = 0
        for tensors, labels in train_loader:
            optimizer.zero_grad()
            logits = model(tensors.to(device))
            loss = ordinal_loss(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        _, val_f1 = get_metrics(model, val_loader, device)
        print(f"Epoch {epoch+1}/20 | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    print("\n--- Final Evaluation ---")
    cm, f1_w = get_metrics(model, test_loader, device)
    
    # SAVE WEIGHTS AND METRICS FOR RUNNER
    torch.save(model.state_dict(), os.path.join(out_dir, "cnn_attentional_weights.pth"))
    
    metrics_dict = {"F1_Weighted": float(f1_w), "Confusion_Matrix": cm.tolist()}
    with open(os.path.join(out_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print("Run data saved successfully.")