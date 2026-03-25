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
        tensor = torch.load(item['fused_tensor_path']) # Shape: [2, 224, 224]
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
        
        # 1. The CNN Feature Extractor (ResNet-50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the first layer to accept 1-channel grayscale (Mammogram)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Strip the classification head and global pooling layer.
        # We want the raw spatial feature maps (Shape will be [Batch, 2048, 7, 7])
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. The Cross-Attention Module
        # ResNet50 outputs 2048 channels.
        embed_dim = 2048
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        
        # 3. The Ordinal Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), # CC + Attended MLO
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Strong dropout to prevent overfitting
            nn.Linear(512, num_classes - 1)
        )

    def forward(self, dual_tensor):
        # dual_tensor shape: [Batch, 2, 224, 224]
        cc_img = dual_tensor[:, 0:1, :, :] 
        mlo_img = dual_tensor[:, 1:2, :, :]
        
        # 1. Extract Spatial Feature Maps using CNN
        cc_feat = self.feature_extractor(cc_img)   
        mlo_feat = self.feature_extractor(mlo_img) 
        
        B, C, H, W = cc_feat.shape
        
        # 2. Prepare for Attention (Flatten spatial dimensions 7x7 -> 49 sequence length)
        cc_seq = cc_feat.view(B, C, -1).permute(0, 2, 1) 
        mlo_seq = mlo_feat.view(B, C, -1).permute(0, 2, 1)
        
        # 3. Cross-Attention
        attended_mlo_seq, _ = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq)
        
        # 4. Reshape back to image dimensions and Pool
        attended_mlo_feat = attended_mlo_seq.permute(0, 2, 1).view(B, C, H, W)
        
        cc_pooled = self.global_pool(cc_feat).view(B, -1)
        mlo_pooled = self.global_pool(attended_mlo_feat).view(B, -1)
        
        # 5. Fuse and Classify
        fused = torch.cat([cc_pooled, mlo_pooled], dim=1)
        return self.classifier(fused)

# ==========================================
# 3. Ordinal Loss & Metrics
# ==========================================
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
    f1_g = f1_score(all_labels, all_preds, average=None, labels=range(6), zero_division=0)
    return cm, f1_w, f1_g

# ==========================================
# 4. Training Loop
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CORRECTED PATH FOR YOUR ENVIRONMENT
    meta_path = "/home/host_dir/spatial_alignment/raw-4/output/saliency_metadata.json"
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find {meta_path}. Did you run 1_train_saliency_affine.py first?")
    
    # 1. Dataset & Splits
    full_dataset = SaliencyTensorDataset(meta_path, augment=True)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    # 2. Balanced Sampling
    train_labels = [int(full_dataset.data[i]['birads_label']) for i in train_ds.indices]
    class_sample_count = np.array([train_labels.count(t) for t in range(6)])
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    samples_weight = np.array([1. / class_sample_count[t] for t in train_labels])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))

    # Batch size set to 16. If you get CUDA Out of Memory, drop this to 8 or 4.
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # 3. Initialize Model
    model = CNNCrossAttentionGrader().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 

    print("--- Training CNN ResNet50 with Cross-Attention ---")
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
            
        _, val_f1, _ = get_metrics(model, val_loader, device)
        print(f"Epoch {epoch+1}/20 | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    print("\n--- Final Evaluation (Blind Test Set) ---")
    cm, f1_w, f1_grades = get_metrics(model, test_loader, device)
    print("\nConfusion Matrix:\n", cm)
    print("\nOverall Weighted F1 Score:", f1_w)
    
    # Save the weights so you can use them in visualization later if you want
    torch.save(model.state_dict(), "/home/host_dir/spatial_alignment/raw-4/output/cnn_attentional_weights.pth")
    print("Model weights saved.")