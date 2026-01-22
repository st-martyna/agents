"""
Individual degradation transform functions for image quality degradation.

Each transform has 5 severity levels (1-5), where 1 is minimal degradation
and 5 is severe degradation suitable for stress-testing model robustness.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import io
from PIL import Image


# Severity level configurations for each degradation type
SEVERITY_CONFIGS = {
    "lighting": [1.0, 0.7, 0.5, 0.3, 0.2],  # gamma values
    "gaussian_blur": [3, 5, 9, 13, 15],  # kernel sizes
    "gaussian_noise": [0.01, 0.03, 0.07, 0.12, 0.15],  # sigma values
    "motion_blur": [5, 10, 15, 20, 25],  # kernel lengths in pixels
    "occlusion": [0.05, 0.15, 0.25, 0.35, 0.40],  # coverage percentages
    "jpeg_compression": [95, 70, 50, 30, 15],  # quality levels
    "pixelation": [1, 2, 4, 8, 12],  # downsample factors
}


def _validate_severity(severity: int) -> None:
    """Validate that severity is in range 1-5."""
    if not 1 <= severity <= 5:
        raise ValueError(f"Severity must be between 1 and 5, got {severity}")


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 format."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _ensure_float(image: np.ndarray) -> np.ndarray:
    """Ensure image is float format in [0, 1]."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def apply_lighting_change(
    image: np.ndarray,
    severity: int,
    gamma: Optional[float] = None
) -> np.ndarray:
    """
    Apply gamma correction to simulate lighting changes.

    Lower gamma values simulate darker/low-light conditions.

    Args:
        image: Input image (H, W, C) in uint8 or float format
        severity: Degradation severity level 1-5
        gamma: Optional explicit gamma value (overrides severity)

    Returns:
        Degraded image in same format as input
    """
    _validate_severity(severity)

    was_uint8 = image.dtype == np.uint8
    image_float = _ensure_float(image)

    if gamma is None:
        gamma = SEVERITY_CONFIGS["lighting"][severity - 1]

    # Apply gamma correction
    # For dark scenes (gamma < 1), this darkens the image
    corrected = np.power(image_float, 1.0 / gamma)
    corrected = np.clip(corrected, 0, 1)

    if was_uint8:
        return (corrected * 255).astype(np.uint8)
    return corrected


def apply_gaussian_blur(
    image: np.ndarray,
    severity: int,
    kernel_size: Optional[int] = None
) -> np.ndarray:
    """
    Apply Gaussian blur to simulate out-of-focus or motion artifacts.

    Args:
        image: Input image (H, W, C) in uint8 or float format
        severity: Degradation severity level 1-5
        kernel_size: Optional explicit kernel size (must be odd)

    Returns:
        Blurred image in same format as input
    """
    _validate_severity(severity)

    if kernel_size is None:
        kernel_size = SEVERITY_CONFIGS["gaussian_blur"][severity - 1]

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # sigma = 0 means it's computed from kernel size
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred


def apply_gaussian_noise(
    image: np.ndarray,
    severity: int,
    sigma: Optional[float] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to simulate sensor noise.

    Args:
        image: Input image (H, W, C) in uint8 or float format
        severity: Degradation severity level 1-5
        sigma: Optional explicit noise standard deviation
        seed: Optional random seed for reproducibility

    Returns:
        Noisy image in same format as input
    """
    _validate_severity(severity)

    was_uint8 = image.dtype == np.uint8
    image_float = _ensure_float(image)

    if sigma is None:
        sigma = SEVERITY_CONFIGS["gaussian_noise"][severity - 1]

    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(0, sigma, image_float.shape).astype(np.float32)
    noisy = image_float + noise
    noisy = np.clip(noisy, 0, 1)

    if was_uint8:
        return (noisy * 255).astype(np.uint8)
    return noisy


def apply_motion_blur(
    image: np.ndarray,
    severity: int,
    kernel_length: Optional[int] = None,
    angle: Optional[float] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply motion blur to simulate camera or object motion.

    Args:
        image: Input image (H, W, C) in uint8 or float format
        severity: Degradation severity level 1-5
        kernel_length: Optional explicit kernel length in pixels
        angle: Optional explicit angle in degrees (0 = horizontal)
        seed: Optional random seed for reproducibility

    Returns:
        Motion-blurred image in same format as input
    """
    _validate_severity(severity)

    if kernel_length is None:
        kernel_length = SEVERITY_CONFIGS["motion_blur"][severity - 1]

    if seed is not None:
        np.random.seed(seed)

    if angle is None:
        angle = np.random.uniform(0, 360)

    # Create motion blur kernel
    kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)

    # Draw a line in the kernel at the specified angle
    center = kernel_length // 2
    angle_rad = np.deg2rad(angle)

    for i in range(kernel_length):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))
        if 0 <= x < kernel_length and 0 <= y < kernel_length:
            kernel[y, x] = 1

    # Normalize kernel
    kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel

    # Apply convolution
    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


def apply_occlusion(
    image: np.ndarray,
    severity: int,
    coverage: Optional[float] = None,
    n_patches: int = 1,
    fill_value: int = 0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply random rectangular occlusions to simulate partial visibility.

    Args:
        image: Input image (H, W, C) in uint8 or float format
        severity: Degradation severity level 1-5
        coverage: Optional explicit coverage percentage (0-1)
        n_patches: Number of occlusion patches (default: 1)
        fill_value: Value to fill occluded regions (0=black, 255=white)
        seed: Optional random seed for reproducibility

    Returns:
        Occluded image in same format as input
    """
    _validate_severity(severity)

    if coverage is None:
        coverage = SEVERITY_CONFIGS["occlusion"][severity - 1]

    if seed is not None:
        np.random.seed(seed)

    h, w = image.shape[:2]
    total_area = h * w
    target_occluded_area = int(total_area * coverage)

    # Create copy to modify
    occluded = image.copy()

    # Distribute area across patches
    area_per_patch = target_occluded_area // n_patches

    for _ in range(n_patches):
        # Random aspect ratio between 0.5 and 2.0
        aspect = np.random.uniform(0.5, 2.0)

        # Calculate patch dimensions
        patch_h = int(np.sqrt(area_per_patch / aspect))
        patch_w = int(patch_h * aspect)

        # Clamp to image bounds
        patch_h = min(patch_h, h)
        patch_w = min(patch_w, w)

        # Random position
        y = np.random.randint(0, max(1, h - patch_h))
        x = np.random.randint(0, max(1, w - patch_w))

        # Apply occlusion
        if image.dtype == np.uint8:
            occluded[y:y+patch_h, x:x+patch_w] = fill_value
        else:
            occluded[y:y+patch_h, x:x+patch_w] = fill_value / 255.0

    return occluded


def apply_jpeg_compression(
    image: np.ndarray,
    severity: int,
    quality: Optional[int] = None
) -> np.ndarray:
    """
    Apply JPEG compression artifacts.

    Args:
        image: Input image (H, W, C) in uint8 or float format
        severity: Degradation severity level 1-5
        quality: Optional explicit JPEG quality (1-100)

    Returns:
        Compressed image in same format as input
    """
    _validate_severity(severity)

    was_uint8 = image.dtype == np.uint8
    image_uint8 = _ensure_uint8(image)

    if quality is None:
        quality = SEVERITY_CONFIGS["jpeg_compression"][severity - 1]

    # Convert to PIL, compress, and back
    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
        # Convert BGR to RGB for PIL
        pil_image = Image.fromarray(cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image_uint8)

    # Compress to JPEG in memory
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    # Read back
    compressed_pil = Image.open(buffer)
    compressed = np.array(compressed_pil)

    # Convert RGB back to BGR if needed
    if len(compressed.shape) == 3 and compressed.shape[2] == 3:
        compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)

    if not was_uint8:
        return compressed.astype(np.float32) / 255.0
    return compressed


def apply_pixelation(
    image: np.ndarray,
    severity: int,
    factor: Optional[int] = None
) -> np.ndarray:
    """
    Apply pixelation by downsampling and upsampling.

    Args:
        image: Input image (H, W, C) in uint8 or float format
        severity: Degradation severity level 1-5
        factor: Optional explicit downsample factor

    Returns:
        Pixelated image in same format as input
    """
    _validate_severity(severity)

    if factor is None:
        factor = SEVERITY_CONFIGS["pixelation"][severity - 1]

    if factor <= 1:
        return image.copy()

    h, w = image.shape[:2]

    # Downsample
    small = cv2.resize(image, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_NEAREST)

    # Upsample back to original size
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixelated


# Mapping of degradation names to functions
DEGRADATION_FUNCTIONS = {
    "lighting": apply_lighting_change,
    "gaussian_blur": apply_gaussian_blur,
    "blur": apply_gaussian_blur,  # alias
    "gaussian_noise": apply_gaussian_noise,
    "noise": apply_gaussian_noise,  # alias
    "motion_blur": apply_motion_blur,
    "occlusion": apply_occlusion,
    "jpeg_compression": apply_jpeg_compression,
    "jpeg": apply_jpeg_compression,  # alias
    "pixelation": apply_pixelation,
}


def apply_degradation(
    image: np.ndarray,
    degradation_type: str,
    severity: int,
    **kwargs
) -> np.ndarray:
    """
    Apply a degradation by name.

    Args:
        image: Input image
        degradation_type: Name of degradation (e.g., 'blur', 'noise')
        severity: Severity level 1-5
        **kwargs: Additional arguments for the specific degradation

    Returns:
        Degraded image
    """
    if degradation_type not in DEGRADATION_FUNCTIONS:
        raise ValueError(
            f"Unknown degradation type: {degradation_type}. "
            f"Available: {list(DEGRADATION_FUNCTIONS.keys())}"
        )

    return DEGRADATION_FUNCTIONS[degradation_type](image, severity, **kwargs)


def get_degradation_info(degradation_type: str) -> dict:
    """
    Get information about a degradation type.

    Args:
        degradation_type: Name of degradation

    Returns:
        Dict with severity levels and description
    """
    # Normalize aliases
    canonical_name = degradation_type
    if degradation_type == "blur":
        canonical_name = "gaussian_blur"
    elif degradation_type == "noise":
        canonical_name = "gaussian_noise"
    elif degradation_type == "jpeg":
        canonical_name = "jpeg_compression"

    if canonical_name not in SEVERITY_CONFIGS:
        raise ValueError(f"Unknown degradation type: {degradation_type}")

    descriptions = {
        "lighting": "Gamma correction for lighting changes (lower = darker)",
        "gaussian_blur": "Gaussian blur kernel size (larger = more blur)",
        "gaussian_noise": "Gaussian noise sigma (larger = more noise)",
        "motion_blur": "Motion blur kernel length in pixels",
        "occlusion": "Random rectangular patch coverage percentage",
        "jpeg_compression": "JPEG quality level (lower = more artifacts)",
        "pixelation": "Downsample factor (larger = more pixelated)",
    }

    return {
        "name": canonical_name,
        "description": descriptions.get(canonical_name, ""),
        "severity_levels": SEVERITY_CONFIGS[canonical_name],
        "n_levels": 5,
    }
