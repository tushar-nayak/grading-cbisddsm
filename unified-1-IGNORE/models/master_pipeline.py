import torch
import torch.nn as nn
from .segmentation import SegmentationUNet
from .detection import DetectionYOLO
from .classification import CancerGradeViT
from .registration import DeformableSTN, warp_tensor

class CompleteMammogramPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = SegmentationUNet(pretrained=True)
        self.yolo = DetectionYOLO()
        self.stn = DeformableSTN(input_channels=6)
        self.classifier = CancerGradeViT(num_classes=5, pretrained=True)

    def forward(self, cc_img, mlo_img):
        # 1. Feature Extraction
        cc_mask, mlo_mask = self.unet(cc_img), self.unet(mlo_img)
        cc_bbox, mlo_bbox = self.yolo(cc_img), self.yolo(mlo_img)
        
        # 2. Registration
        cc_stack = torch.cat([cc_img, cc_mask, cc_bbox], dim=1)
        mlo_stack = torch.cat([mlo_img, mlo_mask, mlo_bbox], dim=1)
        ddf = self.stn(cc_stack, mlo_stack)
        
        warped_mlo_img = warp_tensor(mlo_img, ddf)
        warped_mlo_mask = warp_tensor(mlo_mask, ddf)
        
        # 3. Classification
        fused_views = torch.cat([cc_img, warped_mlo_img], dim=1)
        grade_logits = self.classifier(fused_views)
        
        return grade_logits, warped_mlo_mask, cc_mask, ddf