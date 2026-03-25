import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class SegmentationUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load the model with default settings first (which has 21 classes)
        weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
        self.unet = fcn_resnet50(weights=weights)
        
        # 1. Modify the first convolutional layer to accept 1-channel grayscale
        original_conv = self.unet.backbone.conv1
        self.unet.backbone.conv1 = nn.Conv2d(
            1, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, padding=original_conv.padding, bias=False
        )
        
        # 2. Modify the main classifier head to output 1 class instead of 21
        # The 5th layer (index 4) of the classifier is the final Conv2d layer
        self.unet.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        
        # 3. Modify the auxiliary classifier (if it exists) to output 1 class
        if self.unet.aux_classifier is not None:
            self.unet.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        
    def forward(self, x):
        # Returns a spatial probability map (values between 0 and 1)
        return torch.sigmoid(self.unet(x)['out'])