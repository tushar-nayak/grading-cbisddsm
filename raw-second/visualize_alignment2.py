import torch
import json
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def calculate_ncc(img1, img2):
    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)
    i1_mean, i2_mean = torch.mean(img1), torch.mean(img2)
    i1_std, i2_std = torch.std(img1), torch.std(img2)
    ncc = torch.mean((img1 - i1_mean) * (img2 - i2_mean)) / (i1_std * i2_std + 1e-8)
    return ncc.item()

def evaluate_alignment(metadata_path, num_visuals=3):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    results = []
    ncc_scores = []
    
    print(f"Analyzing {len(metadata)} deformable alignments...")

    for item in metadata:
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
                "ncc": ncc,
                "label": item['birads_label']
            })

    print("\n--- REGISTRATION METRICS ---")
    print(f"Average NCC (Similarity): {np.mean(ncc_scores):.4f}")
    
    samples = random.sample(results, min(num_visuals, len(results)))
    fig, axes = plt.subplots(num_visuals, 3, figsize=(15, 5 * num_visuals))
    
    for i, res in enumerate(samples):
        axes[i, 0].imshow(res['cc'].numpy(), cmap='gray')
        axes[i, 0].set_title(f"Target (CC View)\nBI-RADS: {res['label']}")
        
        axes[i, 1].imshow(res['aligned'].numpy(), cmap='gray')
        axes[i, 1].set_title(f"Deformed MLO\nNCC: {res['ncc']:.3f}")
        
        axes[i, 2].imshow(res['cc'].numpy(), cmap='Reds', alpha=0.5)
        axes[i, 2].imshow(res['aligned'].numpy(), cmap='Blues', alpha=0.5)
        axes[i, 2].set_title("Overlay\n(Red=CC, Blue=MLO)")
        
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    gallery_path = "/home/sofa/host_dir/spatial_alignment/output/alignment_deformable_gallery.png"
    plt.savefig(gallery_path)
    print(f"\nVisual gallery saved to: {gallery_path}")

if __name__ == "__main__":
    meta_path = "/home/sofa/host_dir/spatial_alignment/output/alignment_metadata_v2.json"
    evaluate_alignment(meta_path)