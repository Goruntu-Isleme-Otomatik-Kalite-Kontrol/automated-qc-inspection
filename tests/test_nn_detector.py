import cv2
import numpy as np
import pytest
from src.nn_detector import (
    ResNetPatchFeatureExtractor,
    build_memory_bank,
    detect_nn_defects,
    load_memory_bank,
    save_memory_bank,
)

IMG_SIZE = 224
DEFECT_TOP_LEFT = (80, 80)
DEFECT_SIZE = 60


@pytest.fixture(scope="module")
def extractor() -> ResNetPatchFeatureExtractor:
    """Load the frozen ResNet-18 backbone once for the whole module (CPU for determinism)."""
    return ResNetPatchFeatureExtractor(device="cpu")


def _create_textured_image(seed: int = 0) -> np.ndarray:
    """Generate a deterministic, smoothly textured RGB image that mimics a defect-free surface."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(90, 170, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (7, 7), 0)


def _add_synthetic_defect(img: np.ndarray) -> np.ndarray:
    """Draw a dark square 'defect' onto a copy of the image."""
    defective = img.copy()
    x, y = DEFECT_TOP_LEFT
    cv2.rectangle(defective, (x, y), (x + DEFECT_SIZE, y + DEFECT_SIZE), (10, 10, 10), -1)
    return defective


def _boxes_overlap(box_a: tuple, box_b: tuple) -> bool:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def test_extract_patches_shape_and_l2_norm(extractor):
    patches, grid = extractor.extract_patches(_create_textured_image())

    # layer2 (128 ch) + upsampled layer3 (256 ch) on a 28x28 grid for a 224x224 input
    assert grid == (28, 28)
    assert tuple(patches.shape) == (28 * 28, 384)

    # Every patch vector must lie on the unit hypersphere
    norms = patches.norm(dim=1).cpu().numpy()
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_memory_bank_respects_max_patches_and_is_reproducible(extractor):
    images = [_create_textured_image(seed) for seed in range(3)]  # 3 x 784 = 2352 patches

    bank_a = build_memory_bank(images, extractor, max_patches=1000, seed=42)
    bank_b = build_memory_bank(images, extractor, max_patches=1000, seed=42)

    assert bank_a.shape == (1000, 384)
    assert bank_a.dtype == np.float32
    # Same seed must produce the identical subsampled bank
    assert np.array_equal(bank_a, bank_b)


def test_memory_bank_save_load_roundtrip(tmp_path):
    bank = np.random.default_rng(0).random((50, 384)).astype(np.float32)
    cache_path = tmp_path / "cache" / "bank.npz"

    save_memory_bank(cache_path, bank)
    loaded = load_memory_bank(cache_path)

    assert cache_path.exists()
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, bank)


def test_detect_nn_defects_passes_good_and_localizes_defect(extractor):
    good = _create_textured_image()
    defective = _add_synthetic_defect(good)
    bank = build_memory_bank([good], extractor)

    # Probe the defective image once to read its anomaly level
    probe = detect_nn_defects(defective, bank, extractor, anomaly_threshold=10.0)
    assert probe["label"] == "PASS"  # nothing can exceed an unreachable threshold
    assert probe["max_score"] > 0.2

    # Use a threshold halfway to the defect peak so the test does not depend on absolute scales
    threshold = probe["max_score"] / 2

    good_result = detect_nn_defects(good, bank, extractor, anomaly_threshold=threshold)
    defect_result = detect_nn_defects(defective, bank, extractor, anomaly_threshold=threshold)

    # The reference image itself is in the memory bank, so it must be judged defect-free
    assert good_result["label"] == "PASS"
    assert good_result["boxes"] == []
    assert good_result["max_score"] < defect_result["max_score"]

    # The synthetic defect must be detected and localized at the right place
    assert defect_result["label"] == "FAIL"
    defect_box = (*DEFECT_TOP_LEFT, DEFECT_SIZE, DEFECT_SIZE)
    assert any(_boxes_overlap(box, defect_box) for box in defect_result["boxes"])

    # Output maps must match the input resolution
    assert defect_result["heatmap"].shape == (IMG_SIZE, IMG_SIZE)
    assert defect_result["binary_mask"].shape == (IMG_SIZE, IMG_SIZE)
