import torch
import json
import random
import matplotlib.pyplot as plt
import os

def visualize_sample(metadata_path):
    # 1. Load the metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # 2. Pick a random patient sample
    sample = random.choice(metadata)
    tensor_path = sample['fused_tensor_path']
    
    if not os.path.exists(tensor_path):
        print(f"Error: Tensor file not found at {tensor_path}")
        return

    # 3. Load the fused tensor [Channel, H, W]
    # In our script, fused_features = torch.cat([cc_view, aligned_mlo], dim=1)
    # So channel 0 is CC, and channel 1 is the Aligned MLO.
    fused_tensor = torch.load(tensor_path)
    
    cc_view = fused_tensor[0].numpy()
    aligned_mlo = fused_tensor[1].numpy()

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cc_view, cmap='gray')
    axes[0].set_title(f"Target: CC View\n(BI-RADS: {sample['birads_label']})")
    axes[0].axis('off')
    
    axes[1].imshow(aligned_mlo, cmap='gray')
    axes[1].set_title("Result: Aligned MLO\n(Warped by STN)")
    axes[1].axis('off')

    # Overlay to check structural overlap
    axes[2].imshow(cc_view, cmap='Reds', alpha=0.5)
    axes[2].imshow(aligned_mlo, cmap='Blues', alpha=0.5)
    axes[2].set_title("Anatomical Overlay\n(Red=CC, Blue=MLO)")
    axes[2].axis('off')

    plt.tight_layout()
    save_path = "/home/sofa/host_dir/spatial_alignment/output/alignment_check.png"
    plt.savefig(save_path)
    print(f"Verification image saved to: {save_path}")

if __name__ == "__main__":
    meta_path = "/home/sofa/host_dir/spatial_alignment/output/alignment_metadata_v2.json"
    visualize_sample(meta_path)