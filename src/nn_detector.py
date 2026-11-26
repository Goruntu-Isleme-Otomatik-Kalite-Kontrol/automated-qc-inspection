from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetPatchFeatureExtractor(nn.Module):
    """Frozen ResNet-18 backbone extracting mid-level spatial patch features (layer2 + layer3)."""

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # Keep early + mid layers (up to layer3); discard layer4 and classification head
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2  # Output: (B, 128, 28, 28) for 224x224 input
        self.layer3 = backbone.layer3  # Output: (B, 256, 14, 14) for 224x224 input

        # Freeze all weights—no gradient training needed
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        self.to(self.device)

        # Standard ImageNet channel normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        )

    def _to_tensor(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Convert (H, W, 3) uint8 RGB image into normalized (1, 3, H, W) tensor."""
        tensor = (
            torch.from_numpy(rgb_image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def extract_patches(self, rgb_image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Return L2-normalized (H_grid * W_grid, 384) patch vectors and grid shape (28, 28)."""
        x = self._to_tensor(rgb_image)
        x = self.stem(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)

        # Upsample layer3 (14x14) to match layer2 (28x28) and concatenate -> 384 channels
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False
        )
        combined = torch.cat([feat2, feat3_up], dim=1)  # (1, 384, 28, 28)

        _, channels, grid_h, grid_w = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(-1, channels)  # (784, 384)
        patches = F.normalize(patches, p=2, dim=1)
        return patches, (grid_h, grid_w)


def build_memory_bank(
    train_rgb_images: list[np.ndarray],
    extractor: ResNetPatchFeatureExtractor,
    max_patches: int = 15000,
    seed: int = 42,
) -> np.ndarray:
    """Build a compact reference bank of healthy patch vectors from train/good images."""
    all_patches = []
    for img in train_rgb_images:
        patches, _ = extractor.extract_patches(img)
        all_patches.append(patches.cpu().numpy())

    bank = np.concatenate(all_patches, axis=0).astype(np.float32)

    if len(bank) > max_patches:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(bank), size=max_patches, replace=False)
        bank = bank[indices]

    return bank


def save_memory_bank(cache_path: str | Path, memory_bank: np.ndarray) -> None:
    """Save the healthy patch memory bank to disk in .npz format."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, memory_bank=memory_bank)


def load_memory_bank(cache_path: str | Path) -> np.ndarray:
    """Load a previously saved patch memory bank from .npz."""
    with np.load(cache_path) as data:
        return data["memory_bank"].astype(np.float32)


def detect_nn_defects(
    rgb_image: np.ndarray,
    memory_bank: np.ndarray,
    extractor: ResNetPatchFeatureExtractor,
    anomaly_threshold: float = 0.45,
    min_defect_area_px: int = 120,
    morph_kernel_size: int = 5,
) -> dict:
    """Compare test image patches to the healthy memory bank and extract OpenCV defect boxes."""
    img_h, img_w = rgb_image.shape[:2]

    # 1. Extract test patch fingerprints and compute nearest-neighbor Euclidean distance
    test_patches, (grid_h, grid_w) = extractor.extract_patches(rgb_image)
    bank_tensor = torch.from_numpy(memory_bank).to(extractor.device)

    distances = torch.cdist(test_patches, bank_tensor, p=2)  # (784, N_bank)
    min_dists, _ = torch.min(distances, dim=1)  # Closest healthy match per patch
    patch_map = min_dists.view(grid_h, grid_w).cpu().numpy()

    # 2. Upsample 28x28 patch distance grid to full image resolution (224x224)
    heatmap = cv2.resize(patch_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    # 3. OpenCV thresholding, morphological cleanup, and contour extraction
    binary_mask = (heatmap > anomaly_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_defect_area_px:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((int(x), int(y), int(w), int(h)))

    label = "FAIL" if len(boxes) > 0 else "PASS"
    return {
        "label": label,
        "boxes": boxes,
        "binary_mask": cleaned_mask,
        "heatmap": heatmap,
        "max_score": float(np.max(heatmap)),
    }