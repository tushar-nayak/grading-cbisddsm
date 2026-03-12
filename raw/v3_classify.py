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
# 1. Dataset with Tensor Augmentation
# ==========================================
class FusedTensorDataset(Dataset):
    def __init__(self, metadata_json, augment=False):
        with open(metadata_json, 'r') as f:
            self.data = json.load(f)
        self.augment = augment
        # Augmentations for tensors of shape [2, 224, 224]
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Adding small rotations can help with breast tissue variance
            transforms.RandomRotation(10)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tensor = torch.load(item['fused_tensor_path'])
        label = int(item['birads_label'])
        
        if self.augment:
            tensor = self.transforms(tensor)
            
        return tensor, label

# ==========================================
# 2. Improved Model (MobileNetV3 + Transfer Learning)
# ==========================================
class MammogramGrader(nn.Module):
    def __init__(self, num_classes=6):
        super(MammogramGrader, self).__init__()
        # Use the correct naming for weights based on your torchvision version
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        self.backbone = models.mobilenet_v3_large(weights=weights)
        
        # Modify first conv layer to accept 2 channels (CC + Aligned MLO)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            2, original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        # Ordinal Regression Head
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Increased dropout for better regularization
            nn.Linear(128, num_classes - 1) 
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 3. Ordinal Loss & Metric Functions
# ==========================================
def ordinal_loss(predictions, targets):
    num_classes = predictions.size(1) + 1
    levels = torch.arange(num_classes - 1).to(predictions.device)
    binary_labels = (targets.view(-1, 1) > levels).float()
    return nn.BCEWithLogitsLoss()(predictions, binary_labels)

def get_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for tensors, labels in loader:
            tensors = tensors.to(device)
            logits = model(tensors)
            # Sum threshold crossings to get discrete grade (0-5)
            preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds, labels=range(6))
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_grade = f1_score(all_labels, all_preds, average=None, labels=range(6), zero_division=0)
    
    return cm, f1_weighted, f1_per_grade

# ==========================================
# 4. Training Loop
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    meta_path = "/home/sofa/host_dir/spatial_alignment/output/alignment_metadata_v2.json"
    output_model = "/home/sofa/host_dir/spatial_alignment/output/v5_balanced_grader.pth"

    # 1. Setup Dataset and 70/15/15 Split
    full_dataset = FusedTensorDataset(meta_path, augment=True)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 2. Balanced Sampling (Force the model to see minority classes)
    train_labels = [int(full_dataset.data[i]['birads_label']) for i in train_ds.indices]
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in range(6)])
    
    # Avoid division by zero for classes with 0 samples
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # 3. Model & Optimizer
    model = MammogramGrader().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print(f"Starting Balanced Training on {train_size} samples...")
    

    for epoch in range(15):
        model.train()
        train_loss = 0
        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = ordinal_loss(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        _, val_f1, _ = get_metrics(model, val_loader, device)
        print(f"Epoch {epoch+1}/15 | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    # 4. Final Evaluation
    print("\n--- Final Evaluation on Blind Test Set ---")
    cm, f1_w, f1_grades = get_metrics(model, test_loader, device)
    
    print("\nConfusion Matrix (Rows: Truth, Cols: Predicted):")
    print(cm)
    
    
    
    print("\nGrade-Wise F1 Scores (BI-RADS 0-5):")
    for i, score in enumerate(f1_grades):
        print(f"  BI-RADS {i}: {score:.4f}")
    
    print(f"\nOverall Weighted F1 Score: {f1_w:.4f}")
    
    torch.save(model.state_dict(), output_model)
    print(f"\nModel saved to {output_model}")