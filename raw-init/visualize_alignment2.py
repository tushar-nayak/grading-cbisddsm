import torch
import torch.nn.functional as F
import json
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Essential for Docker/SSH headless environments
import matplotlib.pyplot as plt

def calculate_ncc(img1, img2):
    """Calculates Normalized Cross-Correlation between two tensors."""
    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)
    
    i1_mean = torch.mean(img1)
    i2_mean = torch.mean(img2)
    
    i1_std = torch.std(img1)
    i2_std = torch.std(img2)
    
    ncc = torch.mean((img1 - i1_mean) * (img2 - i2_mean)) / (i1_std * i2_std + 1e-8)
    return ncc.item()

def evaluate_alignment(metadata_path, num_visuals=3):
    # 1. Load the metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    results = []
    jacobians = []
    ncc_scores = []
    
    print(f"Analyzing {len(metadata)} alignments...")

    # 2. Iterate and calculate metrics
    for item in metadata:
        theta = torch.tensor(item['transformation_matrix'])
        
        # Jacobian Determinant: det(A) where A is the 2x2 scale/rotation part
        jac = torch.det(theta[:, :2]).item()
        jacobians.append(jac)
        
        # Load tensor to calculate NCC
        if os.path.exists(item['fused_tensor_path']):
            fused_tensor = torch.load(item['fused_tensor_path'])
            cc_view = fused_tensor[0]
            aligned_mlo = fused_tensor[1]
            
            ncc = calculate_ncc(cc_view, aligned_mlo)
            ncc_scores.append(ncc)
            
            results.append({
                "path": item['fused_tensor_path'],
                "cc": cc_view,
                "aligned": aligned_mlo,
                "jac": jac,
                "ncc": ncc,
                "label": item['birads_label']
            })

    # 3. Print Summary Metrics
    print("\n--- REGISTRATION METRICS ---")
    print(f"Average NCC (Similarity): {np.mean(ncc_scores):.4f}")
    print(f"Average Jacobian (Integrity): {np.mean(jacobians):.4f}")
    print(f"Minimum Jacobian: {np.min(jacobians):.4f}")
    print(f"Invalid Folds (Jac <= 0): {np.sum(np.array(jacobians) <= 0)}")
    
    # 4. Generate Visual Gallery
    samples = random.sample(results, min(num_visuals, len(results)))
    fig, axes = plt.subplots(num_visuals, 3, figsize=(15, 5 * num_visuals))
    
    for i, res in enumerate(samples):
        # CC Target
        axes[i, 0].imshow(res['cc'].numpy(), cmap='gray')
        axes[i, 0].set_title(f"Sample {i+1}: CC Target\nBI-RADS: {res['label']}")
        
        # Aligned MLO
        axes[i, 1].imshow(res['aligned'].numpy(), cmap='gray')
        axes[i, 1].set_title(f"Aligned MLO\nNCC: {res['ncc']:.3f} | Jac: {res['jac']:.3f}")
        
        # Overlay
        axes[i, 2].imshow(res['cc'].numpy(), cmap='Reds', alpha=0.5)
        axes[i, 2].imshow(res['aligned'].numpy(), cmap='Blues', alpha=0.5)
        axes[i, 2].set_title("Overlay\n(Red=CC, Blue=MLO)")
        
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    gallery_path = "/home/sofa/host_dir/spatial_alignment/output/alignment_gallery.png"
    plt.savefig(gallery_path)
    print(f"\nVisual gallery saved to: {gallery_path}")

if __name__ == "__main__":
    meta_path = "/home/sofa/host_dir/spatial_alignment/output/alignment_metadata_v2.json"
    evaluate_alignment(meta_path)