import torch
import torch.nn as nn

class DetectionYOLO(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0)) 

    def forward(self, x):
        B, C, H, W = x.size()
        heatmap = torch.zeros((B, 1, H, W), device=self.dummy_param.device)
        heatmap[:, :, int(H*0.4):int(H*0.6), int(W*0.4):int(W*0.6)] = 1.0
        return heatmap