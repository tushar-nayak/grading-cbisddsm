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
# 1. Dataset Loader (Identical to before)
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
# 2. Dual-Stream Vision Transformer (ViT)
# ==========================================
class DualViewViT(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # 1. Load Pre-trained Vision Transformer (ViT-Base with 16x16 patches)
        weights = models.ViT_B_16_Weights.DEFAULT
        self.vit = models.vit_b_16(weights=weights)
        
        # We don't need the ViT's final classification head, just the sequence encoder
        self.vit.heads = nn.Identity() 
        
        # 2. Cross-Attention Module
        # ViT-Base has an embedding dimension of 768
        embed_dim = 768 
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        
        # 3. Final Ordinal Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), # CLS Token from CC + Attended CLS Token from MLO
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes - 1)
        )

    def extract_vit_sequence(self, x):
        """Helper function to get the 197 patch tokens instead of just the final class prediction."""
        # Convert 1-channel grayscale to 3-channel to utilize pre-trained RGB weights
        x = x.repeat(1, 3, 1, 1) 
        
        # Process patches and add the CLS (Class) token
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Run through the Transformer Encoder [Output Shape: Batch, 197 patches, 768 features]
        return self.vit.encoder(x)

    def forward(self, dual_tensor):
        # Split the input tensor [Batch, 2, 224, 224]
        cc_img = dual_tensor[:, 0:1, :, :] 
        mlo_img = dual_tensor[:, 1:2, :, :]
        
        # 1. Extract Sequences [Batch, 197, 768]
        cc_seq = self.extract_vit_sequence(cc_img)
        mlo_seq = self.extract_vit_sequence(mlo_img)
        
        # 2. Cross Attention: CC patches query the MLO patches
        # "If I see a spiculated patch in CC, does an MLO patch confirm it?"
        attended_mlo_seq, _ = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq)
        
        # 3. Extract the CLS Tokens (The 0th token summarizes the entire image sequence)
        cc_cls = cc_seq[:, 0, :]               # [Batch, 768]
        mlo_attended_cls = attended_mlo_seq[:, 0, :] # [Batch, 768]
        
        # 4. Fuse and Classify
        fused = torch.cat([cc_cls, mlo_attended_cls], dim=1) # [Batch, 1536]
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
    meta_path = "/home/sofa/host_dir/spatial_alignment/output/saliency_metadata.json"
    
    # 1. Dataset & Splits
    full_dataset = SaliencyTensorDataset(meta_path, augment=True)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    # 2. Balanced Sampling
    #train_labels = [int(full_dataset.data[i]['birads_label']) for i in train_ds.indices]
    #class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in range(6)])
    train_labels = [int(full_dataset.data[i]['birads_label']) for i in train_ds.indices]
    class_sample_count = np.array([train_labels.count(t) for t in range(6)])
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    samples_weight = np.array([1. / class_sample_count[t] for t in train_labels])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))

    # NOTE: ViT takes more VRAM. If you hit OOM errors, drop batch_size to 8.
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # 3. Initialize Model
    model = DualViewViT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5) # Slightly lower LR for ViT

    print("--- Training Dual-View ViT with Cross-Attention ---")
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