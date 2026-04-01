from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LocalizationPrediction:
    probability_map: np.ndarray
    binary_mask: np.ndarray
    confidence: float


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "output" / "localizer_unet.pth"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE: dict[tuple[str, str], SmallUNet | None] = {}


def default_localizer_checkpoint() -> str:
    return str(_DEFAULT_CHECKPOINT)


def _largest_component(binary: np.ndarray, score_map: np.ndarray) -> tuple[np.ndarray, int, float]:
    labels = measure.label(binary)
    if labels.max() == 0:
        return np.zeros_like(binary, dtype=np.uint8), 0, 0.0
    best_mask = np.zeros_like(binary, dtype=np.uint8)
    best_area = 0
    best_score = -1.0
    for region in measure.regionprops(labels, intensity_image=score_map):
        if region.area < 8:
            continue
        score = float(region.mean_intensity * np.sqrt(region.area))
        if score > best_score:
            best_score = score
            best_area = int(region.area)
            best_mask = (labels == region.label).astype(np.uint8)
    return best_mask, best_area, max(best_score, 0.0)


def load_localizer(checkpoint_path: str | None = None, device: str | None = None) -> SmallUNet | None:
    path = str(Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT)
    target_device = str(torch.device(device or _DEVICE))
    cache_key = (path, target_device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    ckpt = Path(path)
    if not ckpt.exists():
        _MODEL_CACHE[cache_key] = None
        return None

    model = SmallUNet()
    state = torch.load(ckpt, map_location=target_device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(target_device)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def predict_localization(
    image: np.ndarray,
    breast_mask: np.ndarray,
    checkpoint_path: str | None = None,
    device: str | None = None,
    input_size: int = 256,
    threshold: float = 0.35,
) -> LocalizationPrediction | None:
    model = load_localizer(checkpoint_path=checkpoint_path, device=device)
    if model is None:
        return None

    image_f = image.astype(np.float32)
    breast_mask_f = breast_mask.astype(np.float32)
    masked = image_f * breast_mask_f
    pil = Image.fromarray(np.clip(masked * 255.0, 0, 255).astype(np.uint8))
    resized = np.array(pil.resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    if resized.max() > resized.min():
        resized = (resized - resized.min()) / (resized.max() - resized.min())

    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=image_f.shape, mode="bilinear", align_corners=False)
    prob_map = probs.squeeze().cpu().numpy().astype(np.float32)
    prob_map *= breast_mask_f

    active = prob_map[breast_mask_f > 0]
    if active.size == 0:
        return LocalizationPrediction(np.zeros_like(image_f, dtype=np.float32), np.zeros_like(image_f, dtype=np.uint8), 0.0)

    binary = prob_map >= threshold
    binary = morphology.binary_opening(binary, morphology.disk(2))
    binary = morphology.binary_closing(binary, morphology.disk(3))
    binary = morphology.remove_small_objects(binary, min_size=12)
    mask, area, score = _largest_component(binary, prob_map)

    if area == 0:
        fallback = (prob_map >= np.quantile(active, 0.995)) & (breast_mask_f > 0)
        fallback = morphology.binary_dilation(fallback, morphology.disk(2))
        mask, area, score = _largest_component(fallback, prob_map)

    confidence = float(min(1.0, max(score, float(prob_map.max()))))
    return LocalizationPrediction(probability_map=prob_map, binary_mask=mask.astype(np.uint8), confidence=confidence)
