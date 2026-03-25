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
# 1. Dual-View CSV Data Loader 
# ==========================================
class DualViewMammogramDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.data_frame = pd.read_csv(csv_file)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]) 
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        cc_path = str(row['cc_image_path'])
        mlo_path = str(row['mlo_image_path'])
        label = int(row['birads_label'])

        cc_image = Image.open(cc_path).convert('L')
        mlo_image = Image.open(mlo_path).convert('L')

        if self.transform:
            cc_image = self.transform(cc_image)
            mlo_image = self.transform(mlo_image)

        return cc_image, mlo_image, torch.tensor(label, dtype=torch.long), cc_path, mlo_path

# ==========================================
# 2. Deformable Spatial Transformer (U-Net)
# ==========================================
class DeformableSTN(nn.Module):
    def __init__(self, in_channels=2):
        super(DeformableSTN, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        
        # Decoder
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        
        # Predicts Dense Displacement Field (DDF)
        self.flow = nn.Conv2d(16, 2, 3, padding=1)
        
        # Initialize to Identity (Zero displacement)
        self.flow.weight.data.normal_(mean=0.0, std=1e-5)
        self.flow.bias.data.zero_()

    def forward(self, cc_view, mlo_view):
        x = torch.cat([cc_view, mlo_view], dim=1)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        d2 = F.relu(self.dec2(e3))
        d1 = F.relu(self.dec1(d2))
        return self.flow(d1)

class DenseSpatialTransformer(nn.Module):
    def __init__(self, size=(224, 224)):
        super(DenseSpatialTransformer, self).__init__()
        H, W = size
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack([xx, yy]).unsqueeze(0).float()
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0
        self.register_buffer('base_grid', grid)

    def forward(self, moving_image, displacement_field):
        new_grid = self.base_grid + displacement_field
        new_grid = new_grid.permute(0, 2, 3, 1) # Expected shape for grid_sample: (B, H, W, 2)
        return F.grid_sample(moving_image, new_grid, align_corners=True, padding_mode='border')

# ==========================================
# 3. Physics-Informed Advanced Loss
# ==========================================
class PhysicsInformedRegistrationLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1):
        super(PhysicsInformedRegistrationLoss, self).__init__()
        self.lambda_smooth = lambda_smooth

    def ncc_loss(self, I, J):
        I_centered = I - I.mean(dim=[2, 3], keepdim=True)
        J_centered = J - J.mean(dim=[2, 3], keepdim=True)
        cross_corr = torch.sum(I_centered * J_centered, dim=[2, 3])
        I_var = torch.sum(I_centered**2, dim=[2, 3])
        J_var = torch.sum(J_centered**2, dim=[2, 3])
        ncc = cross_corr / (torch.sqrt(I_var * J_var) + 1e-5)
        return -torch.mean(ncc)

    def physics_smoothness_loss(self, displacement_field):
        dy = torch.abs(displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :])
        dx = torch.abs(displacement_field[:, :, :, 1:] - displacement_field[:, :, :, :-1])
        return torch.mean(dx**2) + torch.mean(dy**2)

    def forward(self, warped_mlo, target_cc, displacement_field):
        sim_loss = self.ncc_loss(warped_mlo, target_cc)
        reg_loss = self.physics_smoothness_loss(displacement_field)
        return sim_loss + (self.lambda_smooth * reg_loss), sim_loss, reg_loss

# ==========================================
# 4. Training Loop & Execution
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    csv_file_path = '/home/sofa/host_dir/spatial_alignment/dicom_clean_train.csv' 
    output_dir = '/home/sofa/host_dir/spatial_alignment/output'
    tensor_dir = os.path.join(output_dir, 'fused_tensors')
    model_save_path = os.path.join(output_dir, 'stn_deformable_weights.pth')
    os.makedirs(tensor_dir, exist_ok=True)
    
    # Dataloader
    dataset = DualViewMammogramDataset(csv_file=csv_file_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Initialize Deformable Models
    network = DeformableSTN(in_channels=2).to(device)
    warper = DenseSpatialTransformer(size=(224, 224)).to(device)
    criterion = PhysicsInformedRegistrationLoss(lambda_smooth=55.0).to(device) # High smoothness for medical tissue
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    
    # --- PHASE 1: TRAINING ---
    epochs = 50 
    print("\n--- Starting Deformable STN Training ---")
    
    network.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (cc_view, mlo_view, _, _, _) in enumerate(dataloader):
            cc_view, mlo_view = cc_view.to(device), mlo_view.to(device)
            optimizer.zero_grad() 
            
            # Predict Dense Displacement Field and Warp
            displacement_field = network(cc_view, mlo_view)
            aligned_mlo = warper(mlo_view, displacement_field)
            
            # Calculate physics loss
            loss, sim_loss, reg_loss = criterion(aligned_mlo, cc_view, displacement_field)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Avg Total Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(network.state_dict(), model_save_path)
    print(f"\nTrained STN weights saved to: {model_save_path}")

    # --- PHASE 2: GENERATE FINAL DATASET ---
    print("\n--- Generating Aligned Dataset & Metadata ---")
    network.eval() 
    
    save_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    alignment_results = []

    with torch.no_grad():
        for batch_idx, (cc_view, mlo_view, labels, cc_paths, mlo_paths) in enumerate(save_loader):
            cc_view, mlo_view = cc_view.to(device), mlo_view.to(device)
            
            displacement_field = network(cc_view, mlo_view)
            aligned_mlo = warper(mlo_view, displacement_field)
            fused_features = torch.cat([cc_view, aligned_mlo], dim=1)
            
            for i in range(len(labels)):
                base_filename = os.path.basename(cc_paths[i]).split('.')[0]
                tensor_save_path = os.path.join(tensor_dir, f"{base_filename}_fused.pt")
                
                torch.save(fused_features[i].cpu(), tensor_save_path)
                
                # NOTE: We no longer save 'transformation_matrix' because a Dense Displacement Field 
                # is an enormous pixel-wise tensor. We only save metadata paths for the classifier.
                alignment_results.append({
                    "original_cc_path": cc_paths[i],
                    "original_mlo_path": mlo_paths[i],
                    "fused_tensor_path": tensor_save_path,
                    "birads_label": labels[i].item()
                })
                
    json_output_path = os.path.join(output_dir, 'alignment_metadata_v2.json')
    with open(json_output_path, 'w') as f:
        json.dump(alignment_results, f, indent=4)
        
    print("Pipeline complete! Deformable Tensors and Metadata saved.")