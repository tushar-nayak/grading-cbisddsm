import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableSTN(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, stride=2), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        self.decoder[-1].weight.data.normal_(mean=0.0, std=1e-5)
        self.decoder[-1].bias.data.zero_()

    def forward(self, cc_stack, mlo_stack):
        x = torch.cat([cc_stack, mlo_stack], dim=1)
        return self.decoder(self.encoder(x))

def warp_tensor(tensor, ddf):
    B, C, H, W = tensor.size()
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([xx, yy]).unsqueeze(0).float().to(tensor.device)
    grid = grid.repeat(B, 1, 1, 1)
    
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0
    
    new_grid = (grid + ddf).permute(0, 2, 3, 1)
    return F.grid_sample(tensor, new_grid, align_corners=True, padding_mode='zeros')