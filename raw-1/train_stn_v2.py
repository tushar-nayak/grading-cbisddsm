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
# 2. Raw Spatial Transformer Network (STN)
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
        
        # Initialize with the identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, cc_view, mlo_view):
        x = torch.cat([cc_view, mlo_view], dim=1)
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 7 * 7)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, mlo_view.size(), align_corners=True)
        aligned_mlo = F.grid_sample(mlo_view, grid, align_corners=True)
        return aligned_mlo, theta

# ==========================================
# 3. Training Loop & Execution
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    csv_file_path = '/home/sofa/host_dir/spatial_alignment/dicom_clean_train.csv' 
    output_dir = '/home/sofa/host_dir/spatial_alignment/output'
    tensor_dir = os.path.join(output_dir, 'fused_tensors')
    model_save_path = os.path.join(output_dir, 'stn_weights_v1.pth')
    
    os.makedirs(tensor_dir, exist_ok=True)
    
    # Dataloader
    dataset = DualViewMammogramDataset(csv_file=csv_file_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    print(f"Dataset loaded. Total records: {len(dataset)}")

    # Initialize Model, Optimizer, and Loss Function
    stn_model = RawSpatialTransformer(input_channels=1).to(device)
    optimizer = optim.Adam(stn_model.parameters(), lr=1e-4)
    
    # Mean Squared Error will measure how well the aligned MLO matches the CC view
    criterion = nn.MSELoss() 

    # --- PHASE 1: TRAINING ---
    epochs = 5 # Start small to verify learning
    print("\n--- Starting STN Training ---")
    
    stn_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (cc_view, mlo_view, _, _, _) in enumerate(dataloader):
            cc_view = cc_view.to(device)
            mlo_view = mlo_view.to(device)
            
            optimizer.zero_grad() # Clear old gradients
            
            # Forward pass
            aligned_mlo, _ = stn_model(cc_view, mlo_view)
            
            # Calculate alignment loss
            loss = criterion(aligned_mlo, cc_view)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Alignment Loss (MSE): {avg_loss:.4f}")

    # Save the trained weights
    torch.save(stn_model.state_dict(), model_save_path)
    print(f"\nTrained STN weights saved to: {model_save_path}")

    # --- PHASE 2: GENERATE FINAL DATASET ---
    print("\n--- Generating Aligned Dataset & Metadata ---")
    stn_model.eval() # Switch to evaluation mode
    
    # Re-initialize dataloader with shuffle=False to ensure deterministic saving
    save_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    alignment_results = []

    with torch.no_grad():
        for batch_idx, (cc_view, mlo_view, labels, cc_paths, mlo_paths) in enumerate(save_loader):
            cc_view = cc_view.to(device)
            mlo_view = mlo_view.to(device)
            
            aligned_mlo, transformation_matrix = stn_model(cc_view, mlo_view)
            fused_features = torch.cat([cc_view, aligned_mlo], dim=1)
            
            for i in range(len(labels)):
                base_filename = os.path.basename(cc_paths[i]).split('.')[0]
                tensor_save_path = os.path.join(tensor_dir, f"{base_filename}_fused.pt")
                
                torch.save(fused_features[i].cpu(), tensor_save_path)
                theta_list = transformation_matrix[i].cpu().numpy().tolist()
                
                alignment_results.append({
                    "original_cc_path": cc_paths[i],
                    "original_mlo_path": mlo_paths[i],
                    "fused_tensor_path": tensor_save_path,
                    "birads_label": labels[i].item(),
                    "transformation_matrix": theta_list
                })
                
    json_output_path = os.path.join(output_dir, 'alignment_metadata_v2.json')
    with open(json_output_path, 'w') as f:
        json.dump(alignment_results, f, indent=4)
        
    print(f"Pipeline complete! Tensors and Metadata (v2) saved.")