import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class SegmentationUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
        self.unet = fcn_resnet50(weights=weights, num_classes=1)
        
        # Modify the first convolutional layer to accept 1-channel grayscale instead of 3-channel RGB
        original_conv = self.unet.backbone.conv1
        self.unet.backbone.conv1 = nn.Conv2d(
            1, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, padding=original_conv.padding, bias=False
        )
        
    def forward(self, x):
        # Returns a spatial probability map (values between 0 and 1)
        return torch.sigmoid(self.unet(x)['out'])