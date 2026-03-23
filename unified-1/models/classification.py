import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class CancerGradeViT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        
        # Modify input layer to accept 2 channels (CC + Warped MLO) instead of 3
        original_proj = self.vit.conv_proj
        self.vit.conv_proj = nn.Conv2d(
            2, original_proj.out_channels, kernel_size=original_proj.kernel_size, 
            stride=original_proj.stride
        )
        
        # Modify the classification head for BI-RADS categories
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        
    def forward(self, fused_x):
        return self.vit(fused_x)