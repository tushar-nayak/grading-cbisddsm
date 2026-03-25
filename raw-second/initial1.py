import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. Dual-View CSV Data Loader 
# ==========================================
class DualViewMammogramDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Expects a CSV file where CC and MLO paths are aligned per patient/study.
        """
        self.csv_file = csv_file
        self.transform = transform
        
        # Load the CSV file
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file at {csv_file}. Error: {e}")

        # Default transformation pipeline
        if self.transform is None:
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
        
        # ⚠️ IMPORTANT: Verify these column names match your dicom_clean_train.csv
        try:
            cc_path = str(row['cc_image_path'])
            mlo_path = str(row['mlo_image_path'])
            label = int(row['birads_label'])
        except KeyError as e:
            raise KeyError(f"Column missing in CSV: {e}. Please check your CSV headers.")

        # Load images
        try:
            cc_image = Image.open(cc_path).convert('L')
            mlo_image = Image.open(mlo_path).convert('L')
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at index {idx}. {e}")

        # Apply transformations
        if self.transform:
            cc_image = self.transform(cc_image)
            mlo_image = self.transform(mlo_image)

        # Return paths alongside the tensors for metadata tracking
        return cc_image, mlo_image, torch.tensor(label, dtype=torch.long), cc_path, mlo_path


# ==========================================
# 2. Raw Spatial Transformer Network (STN)
# ==========================================
class RawSpatialTransformer(nn.Module):
    def __init__(self, input_channels=1, image_size=224):
        super(RawSpatialTransformer, self).__init__()
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels * 2, 16, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((7, 7)) 
        )
        
        # Regressor for the 2x3 affine matrix
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
# 3. Execution & Pipeline Saving
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set paths - Ensure this CSV actually exists and contains your data!
    csv_file_path = '/home/sofa/host_dir/spatial_alignment/dicom_clean_train.csv' 
    output_dir = '/home/sofa/host_dir/spatial_alignment/output'
    tensor_dir = os.path.join(output_dir, 'fused_tensors')
    
    os.makedirs(tensor_dir, exist_ok=True)
    
    # Initialize Dataset and DataLoader
    try:
        dataset = DualViewMammogramDataset(csv_file=csv_file_path)
        # shuffle=False ensures we process the dataset deterministically for saving
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
        print(f"Dataset loaded successfully. Total records: {len(dataset)}")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        exit(1)

    # Initialize the STN
    stn_model = RawSpatialTransformer(input_channels=1).to(device)

    # Forward pass and saving
    alignment_results = []
    print("\nStarting Forward Pass and saving fused tensors...")
    
    # We use torch.no_grad() because we are just doing a forward pass to save data, not training yet.
    with torch.no_grad():
        for batch_idx, (cc_view, mlo_view, labels, cc_paths, mlo_paths) in enumerate(dataloader):
            cc_view = cc_view.to(device)
            mlo_view = mlo_view.to(device)
            
            aligned_mlo, transformation_matrix = stn_model(cc_view, mlo_view)
            fused_features = torch.cat([cc_view, aligned_mlo], dim=1)
            
            for i in range(len(labels)):
                # Create a unique filename based on the original CC image name
                base_filename = os.path.basename(cc_paths[i]).split('.')[0]
                tensor_save_path = os.path.join(tensor_dir, f"{base_filename}_fused.pt")
                
                # Save the fused tensor to disk
                torch.save(fused_features[i].cpu(), tensor_save_path)
                
                # Convert the 2x3 transformation matrix to a standard Python list
                theta_list = transformation_matrix[i].cpu().numpy().tolist()
                
                alignment_results.append({
                    "original_cc_path": cc_paths[i],
                    "original_mlo_path": mlo_paths[i],
                    "fused_tensor_path": tensor_save_path,
                    "birads_label": labels[i].item(),
                    "transformation_matrix": theta_list
                })
                
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
            
    # Save the JSON file containing all spatial alignment interpretations
    json_output_path = os.path.join(output_dir, 'alignment_metadata.json')
    with open(json_output_path, 'w') as f:
        json.dump(alignment_results, f, indent=4)
        
    print(f"\nPipeline execution complete.")
    print(f"Tensors saved to: {tensor_dir}")
    print(f"Metadata saved to: {json_output_path}")