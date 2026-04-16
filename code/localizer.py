from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import exposure, measure, morphology
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LocalizationPrediction:
    probability_map: np.ndarray
    binary_mask: np.ndarray
    confidence: float


def _as_float01(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image.astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _normalize_active(image: np.ndarray, active_mask: np.ndarray | None = None) -> np.ndarray:
    image = _as_float01(image)
    active = image[active_mask > 0] if active_mask is not None else image[image > 0]
    if active.size == 0:
        return image
    lo = float(np.percentile(active, 1.0))
    hi = float(np.percentile(active, 99.0))
    if hi <= lo:
        return image
    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _clahe(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    image = _as_float01(image)
    if image.size == 0:
        return image
    try:
        enhanced = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    except Exception:
        enhanced = image
    return np.clip(enhanced.astype(np.float32), 0.0, 1.0)


def _resize_float(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    out = pil.resize(size, Image.Resampling.BILINEAR)
    return np.array(out, dtype=np.float32) / 255.0


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    m = (mask > 0).astype(np.uint8) * 255
    pil = Image.fromarray(m)
    out = pil.resize(size, Image.Resampling.NEAREST)
    return (np.array(out, dtype=np.uint8) > 0).astype(np.uint8)


def _mask_bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return None
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _crop_to_mask(image: np.ndarray, mask: np.ndarray, pad: int = 24) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    bbox = _mask_bbox(mask, pad=pad)
    if bbox is None:
        return image, mask, (0, 0)
    y0, y1, x0, x1 = bbox
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], (y0, x0)


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


def _postprocess_probability(
    prob_map: np.ndarray,
    breast_mask: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float]:
    active = prob_map[breast_mask > 0]
    if active.size == 0:
        return np.zeros_like(prob_map, dtype=np.uint8), 0.0

    binary = (prob_map >= threshold) & (breast_mask > 0)
    binary = morphology.binary_opening(binary, morphology.disk(2))
    binary = morphology.binary_closing(binary, morphology.disk(3))
    binary = morphology.remove_small_objects(binary, min_size=max(12, int(binary.size * 0.0005)))
    binary = ndimage.binary_fill_holes(binary)

    mask, area, score = _largest_component(binary, prob_map)
    if area == 0:
        fallback_thr = float(np.quantile(active, 0.995))
        fallback = (prob_map >= fallback_thr) & (breast_mask > 0)
        fallback = morphology.binary_dilation(fallback, morphology.disk(2))
        mask, area, score = _largest_component(fallback, prob_map)

    confidence = float(min(1.0, max(prob_map.max(), score, float(prob_map[breast_mask > 0].mean()))))
    return mask.astype(np.uint8), confidence


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values],
            dim=1,
        )
        return x * self.sig(self.conv(pooled))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.wg = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.wx = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.wg(g)
        x1 = self.wx(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


class DoubleConvCBAM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.net(x))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: list[int] | None = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConvCBAM(ch, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            ch = feat

        self.bottleneck = DoubleConvCBAM(features[-1], features[-1] * 2, dropout=0.2)

        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.att_gates.append(AttentionGate(f_g=feat, f_l=feat, f_int=max(1, feat // 2)))
            self.decoders.append(DoubleConvCBAM(feat * 2, feat))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, gate, dec, skip in zip(self.upconvs, self.att_gates, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            skip = gate(g=x, x=skip)
            x = dec(torch.cat([skip, x], dim=1))

        return self.final(x)


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


_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "output" / "attention_unet.pth"
_LEGACY_CHECKPOINTS = (
    _DEFAULT_CHECKPOINT,
    Path(__file__).resolve().parents[1] / "output" / "unet_best_512.pth",
    Path(__file__).resolve().parents[1] / "output" / "localizer_unet.pth",
)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE: dict[tuple[str, str], nn.Module | None] = {}


def default_localizer_checkpoint() -> str:
    return str(_DEFAULT_CHECKPOINT)


def _unwrap_state(state) -> dict[str, torch.Tensor] | None:
    if isinstance(state, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            value = state.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in state.values()):
            return state
    return None


def _try_load_model(model: nn.Module, state: dict[str, torch.Tensor]) -> bool:
    try:
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        return False


def _build_model_from_state(state: dict[str, torch.Tensor]) -> nn.Module | None:
    attention = AttentionUNet()
    if _try_load_model(attention, state):
        return attention

    small = SmallUNet()
    if _try_load_model(small, state):
        return small

    return None


def load_localizer(checkpoint_path: str | None = None, device: str | None = None) -> nn.Module | None:
    target_device = str(torch.device(device or _DEVICE))
    checkpoint_candidates = []
    if checkpoint_path:
        checkpoint_candidates.append(Path(checkpoint_path))
    checkpoint_candidates.extend(_LEGACY_CHECKPOINTS)

    for path in checkpoint_candidates:
        cache_key = (str(path), target_device)
        if cache_key in _MODEL_CACHE:
            cached = _MODEL_CACHE[cache_key]
            if cached is not None:
                return cached
            continue

        if not path.exists():
            _MODEL_CACHE[cache_key] = None
            continue

        state = torch.load(path, map_location=target_device)
        state_dict = _unwrap_state(state)
        if state_dict is None:
            _MODEL_CACHE[cache_key] = None
            continue

        model = _build_model_from_state(state_dict)
        if model is None:
            _MODEL_CACHE[cache_key] = None
            continue

        model.to(target_device)
        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model

    return None


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

    image_f = _as_float01(image)
    breast_mask_f = (np.asarray(breast_mask) > 0).astype(np.uint8)
    if image_f.shape != breast_mask_f.shape:
        breast_mask_f = _resize_mask(breast_mask_f, image_f.shape[::-1])

    if image_f.size == 0 or breast_mask_f.sum() == 0:
        return LocalizationPrediction(
            probability_map=np.zeros_like(image_f, dtype=np.float32),
            binary_mask=np.zeros_like(image_f, dtype=np.uint8),
            confidence=0.0,
        )

    crop_img, crop_mask, (y0, x0) = _crop_to_mask(image_f, breast_mask_f, pad=max(24, min(image_f.shape) // 20))
    crop_img = _normalize_active(crop_img, crop_mask)
    crop_img = _clahe(crop_img, clip_limit=0.03)

    resized = _resize_float(crop_img, (input_size, input_size))
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=crop_img.shape, mode="bilinear", align_corners=False)

    crop_prob = probs.squeeze().cpu().numpy().astype(np.float32)
    crop_prob = np.clip(crop_prob, 0.0, 1.0)
    crop_prob *= crop_mask.astype(np.float32)

    crop_mask_pred, confidence = _postprocess_probability(crop_prob, crop_mask, threshold)

    prob_map = np.zeros_like(image_f, dtype=np.float32)
    binary_mask = np.zeros_like(image_f, dtype=np.uint8)
    prob_map[y0 : y0 + crop_prob.shape[0], x0 : x0 + crop_prob.shape[1]] = crop_prob
    binary_mask[y0 : y0 + crop_mask_pred.shape[0], x0 : x0 + crop_mask_pred.shape[1]] = crop_mask_pred
    prob_map *= breast_mask_f.astype(np.float32)
    binary_mask = (binary_mask > 0).astype(np.uint8) * breast_mask_f.astype(np.uint8)

    return LocalizationPrediction(
        probability_map=prob_map.astype(np.float32),
        binary_mask=binary_mask.astype(np.uint8),
        confidence=float(confidence),
    )
