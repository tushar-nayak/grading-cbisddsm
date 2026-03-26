from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models


@dataclass
class ClassificationResult:
    predicted_birads: int
    ordinal_scores: list[float]
    source: str


class CNNCrossAttentionGrader(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        resnet = models.resnet50(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        embed_dim = 2048
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes - 1),
        )

    def forward(self, dual_tensor: torch.Tensor) -> torch.Tensor:
        cc_img = dual_tensor[:, 0:1, :, :]
        mlo_img = dual_tensor[:, 1:2, :, :]
        cc_feat = self.feature_extractor(cc_img)
        mlo_feat = self.feature_extractor(mlo_img)
        batch_size, channels, height, width = cc_feat.shape
        cc_seq = cc_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        mlo_seq = mlo_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        attended_mlo_seq, _ = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq)
        attended_mlo_feat = attended_mlo_seq.permute(0, 2, 1).view(batch_size, channels, height, width)
        cc_pooled = self.global_pool(cc_feat).view(batch_size, -1)
        mlo_pooled = self.global_pool(attended_mlo_feat).view(batch_size, -1)
        fused = torch.cat([cc_pooled, mlo_pooled], dim=1)
        return self.classifier(fused)


class BiradsClassifier:
    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.model = CNNCrossAttentionGrader().to(self.device)
            state = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()

    def _resize(self, image: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
        pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        return np.array(pil.resize((size, size), resample=resample), dtype=np.float32) / 255.0

    def _prepare_channel(self, image: np.ndarray, mask: np.ndarray, bbox: list[int] | None) -> np.ndarray:
        resized = self._resize(image, 224, nearest=False)
        if resized.max() > resized.min():
            resized = (resized - resized.min()) / (resized.max() - resized.min())
        guide = self._resize(mask.astype(np.float32), 224, nearest=True)
        if bbox is not None:
            x, y, w, h = bbox
            sx = 224.0 / image.shape[1]
            sy = 224.0 / image.shape[0]
            x1 = max(0, int(round(x * sx)))
            y1 = max(0, int(round(y * sy)))
            x2 = min(224, int(round((x + w) * sx)))
            y2 = min(224, int(round((y + h) * sy)))
            guide[y1:y2, x1:x2] = 1.0
        enhanced = np.clip(resized * (1.0 + 0.35 * guide), 0.0, 1.0)
        enhanced = (enhanced - 0.5) / 0.5
        return enhanced.astype(np.float32)

    def predict(self, cc_image: np.ndarray, aligned_mlo: np.ndarray, cc_mask: np.ndarray, mlo_mask: np.ndarray, cc_bbox: list[int] | None, mlo_bbox: list[int] | None) -> ClassificationResult:
        if self.model is None:
            lesion_ratio = float((cc_mask.sum() + mlo_mask.sum()) / max(cc_mask.size + mlo_mask.size, 1))
            if lesion_ratio > 0.12:
                pred = 5
            elif lesion_ratio > 0.06:
                pred = 4
            elif lesion_ratio > 0.025:
                pred = 3
            elif lesion_ratio > 0.01:
                pred = 2
            else:
                pred = 1
            return ClassificationResult(predicted_birads=pred, ordinal_scores=[], source="heuristic")

        cc_channel = self._prepare_channel(cc_image, cc_mask, cc_bbox)
        mlo_channel = self._prepare_channel(aligned_mlo, mlo_mask, mlo_bbox)
        stacked = np.stack([cc_channel, mlo_channel], axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            scores = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        pred = max(1, int((torch.tensor(scores) > 0.5).sum().item()))
        return ClassificationResult(predicted_birads=pred, ordinal_scores=[float(x) for x in scores], source="cnn_cross_attention")
