import torch
import torch.nn as nn
import json
import random
import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('Agg') # For headless SSH/Docker environments
import matplotlib.pyplot as plt

# Import your model from the previous script (ensure the file is named correctly)
# If you didn't save it in a module, you can just paste the DualViewViT class here.
from torchvision import models

class DualViewViT_Visualizer(nn.Module):
    """A modified version of your ViT that returns the Attention Maps"""
    def __init__(self, num_classes=6):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT
        self.vit = models.vit_b_16(weights=weights)
        self.vit.heads = nn.Identity() 
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(768 * 2, num_classes - 1)

    def extract_vit_sequence(self, x):
        x = x.repeat(1, 3, 1, 1) 
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return self.vit.encoder(x)

    def forward(self, dual_tensor):
        cc_img = dual_tensor[:, 0:1, :, :] 
        mlo_img = dual_tensor[:, 1:2, :, :]
        
        cc_seq = self.extract_vit_sequence(cc_img)
        mlo_seq = self.extract_vit_sequence(mlo_img)
        
        # CRITICAL CHANGE: We ask PyTorch to return the raw attention weights
        attended_mlo_seq, attn_weights = self.cross_attention(
            query=cc_seq, key=mlo_seq, value=mlo_seq, need_weights=True
        )
        return attn_weights

def generate_visual_diagnostics(metadata_path, model_weights_path, num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/home/sofa/host_dir/spatial_alignment/output/diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load the Model (Optional: If you haven't trained it yet, it will just show untrained attention)
    model = DualViewViT_Visualizer().to(device)
    if os.path.exists(model_weights_path):
        # Allow missing keys because we skipped the Sequential wrapper in the visualizer
        model.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
    model.eval()

    # 2. Load Metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    samples = random.sample(metadata, min(num_samples, len(metadata)))

    print(f"Generating Diagnostics for {len(samples)} patients...")

    for i, item in enumerate(samples):
        if not os.path.exists(item['fused_tensor_path']): continue
        
        # Load Tensor: [2, 224, 224]
        tensor = torch.load(item['fused_tensor_path']).unsqueeze(0).to(device)
        cc_view = tensor[0, 0].cpu().numpy()
        mlo_view = tensor[0, 1].cpu().numpy()

        # --- A. Generate Saliency Mask (Threshold = 0.3) ---
        cc_norm = (cc_view - cc_view.min()) / (cc_view.max() - cc_view.min() + 1e-5)
        saliency_mask = cc_norm > 0.3

        # --- B. Get ViT Attention Map ---
        with torch.no_grad():
            # attn_weights shape: [Batch, Seq_Len_Q, Seq_Len_K] -> [1, 197, 197]
            attn_weights = model(tensor)
            
            # We want to see where the CC's "Class Token" (index 0) is looking in the MLO image (indices 1 to 196)
            # ViT-Base uses 16x16 patches, so 224/16 = 14. The grid is 14x14.
            cls_attention = attn_weights[0, 0, 1:].cpu().numpy().reshape(14, 14)
            
            # Resize the 14x14 heatmap back to 224x224 to overlay on the image
            attention_heatmap = cv2.resize(cls_attention, (224, 224), interpolation=cv2.INTER_CUBIC)
            attention_heatmap = (attention_heatmap - attention_heatmap.min()) / (attention_heatmap.max() - attention_heatmap.min())

        # --- C. Plotting ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Panel 1: Original CC Target
        axes[0].imshow(cc_view, cmap='gray')
        axes[0].set_title(f"Target: CC View\nTrue BI-RADS: {item['birads_label']}")
        
        # Panel 2: Saliency Prioritization
        axes[1].imshow(cc_view, cmap='gray')
        axes[1].imshow(saliency_mask, cmap='YlOrRd', alpha=0.4)
        axes[1].set_title("Affine Saliency Mask\n(Bright Tissue Targeted)")
        
        # Panel 3: Affine Structural Overlay
        axes[2].imshow(cc_view, cmap='Reds', alpha=0.5)
        axes[2].imshow(mlo_view, cmap='Blues', alpha=0.5)
        axes[2].set_title("Saliency-Aligned Overlay\n(Purple = Overlap)")

        # Panel 4: Transformer Attention
        axes[3].imshow(mlo_view, cmap='gray')
        # Overlay the attention heatmap using the "jet" colormap
        im = axes[3].imshow(attention_heatmap, cmap='jet', alpha=0.5)
        axes[3].set_title("ViT Cross-Attention\n(Where the model is looking)")
        
        for ax in axes: ax.axis('off')
        
        save_path = f"{out_dir}/diagnostic_patient_{i+1}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Update this path if you named your saved weights file differently in Step 2
    weights_path = "/home/sofa/host_dir/spatial_alignment/output/vit_attentional_weights.pth"
    meta_path = "/home/sofa/host_dir/spatial_alignment/output/saliency_metadata.json"
    
    generate_visual_diagnostics(meta_path, weights_path)