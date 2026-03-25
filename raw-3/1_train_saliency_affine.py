import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. Dataset Loader
# ==========================================
class DualViewMammogramDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        cc_path = str(row['cc_image_path'])
        mlo_path = str(row['mlo_image_path'])
        label = int(row['birads_label'])

        cc_image = self.transform(Image.open(cc_path).convert('L'))
        mlo_image = self.transform(Image.open(mlo_path).convert('L'))

        return cc_image, mlo_image, torch.tensor(label, dtype=torch.long), cc_path, mlo_path

# ==========================================
# 2. Saliency-Weighted Affine STN
# ==========================================
class RawSpatialTransformer(nn.Module):
    def __init__(self, input_channels=1):
        super(RawSpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels * 2, 16, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((7, 7)) 
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, cc_view, mlo_view):
        x = torch.cat([cc_view, mlo_view], dim=1)
        xs = self.localization(x).view(-1, 32 * 7 * 7)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, mlo_view.size(), align_corners=True)
        aligned_mlo = F.grid_sample(mlo_view, grid, align_corners=True)
        return aligned_mlo, theta

class SaliencyWeightedNCC(nn.Module):
    """Novelty: Forces STN to prioritize alignment of dense breast tissue."""
    def __init__(self, eps=1e-5, threshold=0.3, weight=10.0):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.weight = weight

    def forward(self, I, J):
        I_mean = torch.mean(I, dim=[2, 3], keepdim=True)
        J_mean = torch.mean(J, dim=[2, 3], keepdim=True)
        I_centered, J_centered = I - I_mean, J - J_mean
        
        cross = I_centered * J_centered
        I_var, J_var = I_centered ** 2, J_centered ** 2
        
        # Isolate bright pixels (dense tissue/lesions)
        I_norm = (I - I.amin(dim=(2,3), keepdim=True)) / (I.amax(dim=(2,3), keepdim=True) - I.amin(dim=(2,3), keepdim=True) + self.eps)
        saliency_mask = (I_norm > self.threshold).float()
        
        # Apply multiplier to dense areas
        weight_map = 1.0 + (saliency_mask * self.weight)
        
        w_cross = torch.sum(cross * weight_map, dim=[2, 3])
        w_I_var = torch.sum(I_var * weight_map, dim=[2, 3])
        w_J_var = torch.sum(J_var * weight_map, dim=[2, 3])
        
        ncc = w_cross / (torch.sqrt(w_I_var * w_J_var) + self.eps)
        return 1 - torch.mean(ncc)

# ==========================================
# 3. Training & Tensor Generation
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = '/home/sofa/host_dir/spatial_alignment/dicom_clean_train.csv' 
    out_dir = '/home/sofa/host_dir/spatial_alignment/output'
    tensor_dir = os.path.join(out_dir, 'fused_saliency_tensors')
    os.makedirs(tensor_dir, exist_ok=True)

    dataset = DualViewMammogramDataset(csv_file=csv_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = RawSpatialTransformer().to(device)
    criterion = SaliencyWeightedNCC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("--- Training Saliency-Weighted Affine STN ---")
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        for cc, mlo, _, _, _ in dataloader:
            cc, mlo = cc.to(device), mlo.to(device)
            optimizer.zero_grad()
            aligned_mlo, _ = model(cc, mlo)
            loss = criterion(cc, aligned_mlo)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 | Saliency NCC Loss: {total_loss/len(dataloader):.4f}")

    # Generate Final Tensors
    model.eval()
    save_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    meta = []
    
    with torch.no_grad():
        for cc, mlo, labels, cc_p, mlo_p in save_loader:
            cc, mlo = cc.to(device), mlo.to(device)
            aligned_mlo, theta = model(cc, mlo)
            fused = torch.cat([cc, aligned_mlo], dim=1) # [Batch, 2, 224, 224]
            
            for i in range(len(labels)):
                fname = os.path.basename(cc_p[i]).split('.')[0]
                t_path = os.path.join(tensor_dir, f"{fname}_saliency.pt")
                torch.save(fused[i].cpu(), t_path)
                meta.append({
                    "original_cc_path": cc_p[i],
                    "original_mlo_path": mlo_p[i],
                    "fused_tensor_path": t_path,
                    "birads_label": labels[i].item(),
                    "transformation_matrix": theta[i].cpu().numpy().tolist()
                })

    with open(os.path.join(out_dir, 'saliency_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    print("Saliency Tensors Saved.")