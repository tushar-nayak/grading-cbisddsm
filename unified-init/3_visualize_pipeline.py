import json
import os
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

matplotlib.use("Agg")


class CNN_Visualizer(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.cross_attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes - 1),
        )

    def forward(self, dual_tensor):
        cc_img = dual_tensor[:, 0:1, :, :]
        mlo_img = dual_tensor[:, 1:2, :, :]
        cc_feat = self.feature_extractor(cc_img)
        mlo_feat = self.feature_extractor(mlo_img)
        batch_size, channels, height, width = cc_feat.shape
        cc_seq = cc_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        mlo_seq = mlo_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        _, attn_weights = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq, need_weights=True)
        return attn_weights


def generate_visual_diagnostics(metadata_path, model_weights_path, num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_path = Path(metadata_path)
    out_dir = metadata_path.parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CNN_Visualizer().to(device)
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    samples = random.sample(metadata, min(num_samples, len(metadata)))

    for i, item in enumerate(samples):
        if not os.path.exists(item["fused_tensor_path"]):
            continue

        tensor = torch.load(item["fused_tensor_path"]).unsqueeze(0).to(device)
        cc_view = tensor[0, 0].cpu().numpy()
        mlo_view = tensor[0, 1].cpu().numpy()

        with torch.no_grad():
            attn_weights = model(tensor)
            global_mlo_attention = attn_weights[0].mean(dim=0).cpu().numpy().reshape(7, 7)
            attention_heatmap = cv2.resize(global_mlo_attention, (224, 224))

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(cc_view, cmap="gray")
        axes[0].set_title(f"Target: CC View\nBI-RADS: {item['birads_label']}")
        axes[1].imshow(cc_view, cmap="gray")
        axes[2].imshow(cc_view, cmap="Reds", alpha=0.5)
        axes[2].imshow(mlo_view, cmap="Blues", alpha=0.5)
        axes[2].set_title("STN Aligned Overlay")
        axes[3].imshow(mlo_view, cmap="gray")
        axes[3].imshow(attention_heatmap, cmap="jet", alpha=0.5)
        axes[3].set_title("CNN Cross-Attention")

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"patient_{i + 1}.png")
        plt.close()
    print(f"Diagnostics saved to {out_dir}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    out_dir = Path(os.getenv("RUN_OUT_DIR", str(BASE_DIR / "output")))
    generate_visual_diagnostics(
        out_dir / "saliency_metadata.json",
        out_dir / "cnn_attentional_weights.pth",
    )
