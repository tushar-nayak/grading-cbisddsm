import torch
import torch.nn as nn
import json
import random
import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torchvision import models

# ==========================================
# 1. CNN Visualizer Architecture
# ==========================================
class CNN_Visualizer(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.cross_attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes - 1)
        )

    def forward(self, dual_tensor):
        cc_img = dual_tensor[:, 0:1, :, :] 
        mlo_img = dual_tensor[:, 1:2, :, :]
        
        cc_feat = self.feature_extractor(cc_img)   
        mlo_feat = self.feature_extractor(mlo_img) 
        B, C, H, W = cc_feat.shape
        
        cc_seq = cc_feat.view(B, C, -1).permute(0, 2, 1) 
        mlo_seq = mlo_feat.view(B, C, -1).permute(0, 2, 1)
        
        # We extract the raw attention weights from the CNN's MultiheadAttention
        _, attn_weights = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq, need_weights=True)
        return attn_weights

# ==========================================
# 2. Generation Logic
# ==========================================
def generate_visual_diagnostics(metadata_path, model_weights_path, num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/home/sofa/host_dir/spatial_alignment/raw-4/output/diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    # Load the CNN Visualizer instead of the ViT Visualizer
    model = CNN_Visualizer().to(device)
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
    else:
        print(f"Warning: Could not find weights at {model_weights_path}")
    model.eval()

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    samples = random.sample(metadata, min(num_samples, len(metadata)))

    print(f"Generating Diagnostics for {len(samples)} patients using CNN...")

    for i, item in enumerate(samples):
        if not os.path.exists(item['fused_tensor_path']): continue
        
        tensor = torch.load(item['fused_tensor_path']).unsqueeze(0).to(device)
        cc_view = tensor[0, 0].cpu().numpy()
        mlo_view = tensor[0, 1].cpu().numpy()

        cc_norm = (cc_view - cc_view.min()) / (cc_view.max() - cc_view.min() + 1e-5)
        saliency_mask = cc_norm > 0.3

        # --- Get CNN Attention Map ---
        with torch.no_grad():
            # attn_weights shape: [Batch, 49 (CC locations), 49 (MLO locations)]
            attn_weights = model(tensor)
            
            # Average the attention across all CC queries to see where it looks in the MLO image overall
            global_mlo_attention = attn_weights[0].mean(dim=0).cpu().numpy().reshape(7, 7)
            
            # Resize 7x7 CNN features to 224x224 image
            attention_heatmap = cv2.resize(global_mlo_attention, (224, 224), interpolation=cv2.INTER_CUBIC)
            attention_heatmap = (attention_heatmap - attention_heatmap.min()) / (attention_heatmap.max() - attention_heatmap.min() + 1e-8)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(cc_view, cmap='gray')
        axes[0].set_title(f"Target: CC View\nTrue BI-RADS: {item['birads_label']}")
        
        axes[1].imshow(cc_view, cmap='gray')
        axes[1].imshow(saliency_mask, cmap='YlOrRd', alpha=0.4)
        axes[1].set_title("Affine Saliency Mask\n(Bright Tissue Targeted)")
        
        axes[2].imshow(cc_view, cmap='Reds', alpha=0.5)
        axes[2].imshow(mlo_view, cmap='Blues', alpha=0.5)
        axes[2].set_title("STN Aligned Overlay\n(Purple = Overlap)")

        axes[3].imshow(mlo_view, cmap='gray')
        im = axes[3].imshow(attention_heatmap, cmap='jet', alpha=0.5)
        axes[3].set_title("ResNet Cross-Attention\n(Where the model is looking)")
        
        for ax in axes: ax.axis('off')
        
        save_path = f"{out_dir}/cnn_diagnostic_patient_{i+1}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    weights_path = "/home/sofa/host_dir/spatial_alignment/raw-4/output/cnn_attentional_weights.pth"
    meta_path = "/home/sofa/host_dir/spatial_alignment/raw-4/output/saliency_metadata.json"
    generate_visual_diagnostics(meta_path, weights_path)