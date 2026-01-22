"""
Tests for the degradation module.

Tests individual transforms and the composable pipeline.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.degradation import (
    apply_degradation,
    apply_lighting_change,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_motion_blur,
    apply_occlusion,
    apply_jpeg_compression,
    apply_pixelation,
    get_degradation_info,
    DegradationPipeline,
    DegradationStep,
    TestCase,
    generate_test_matrix,
    generate_single_degradation_cases,
    generate_severity_sweep,
    SEVERITY_CONFIGS,
)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple gradient image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[i, :, :] = i
    return img


@pytest.fixture
def sample_image_float():
    """Create a sample test image in float format."""
    img = np.zeros((256, 256, 3), dtype=np.float32)
    for i in range(256):
        img[i, :, :] = i / 255.0
    return img


class TestIndividualTransforms:
    """Test individual degradation transforms."""

    def test_lighting_change(self, sample_image):
        """Test lighting/gamma correction."""
        for severity in range(1, 6):
            result = apply_lighting_change(sample_image, severity)

            assert result.shape == sample_image.shape
            assert result.dtype == sample_image.dtype

            # Higher severity should make image darker (lower gamma)
            if severity > 1:
                assert result.mean() < sample_image.mean()

    def test_lighting_change_float(self, sample_image_float):
        """Test lighting change on float images."""
        result = apply_lighting_change(sample_image_float, severity=3)

        assert result.shape == sample_image_float.shape
        assert result.dtype == np.float32
        assert result.min() >= 0
        assert result.max() <= 1

    def test_gaussian_blur(self, sample_image):
        """Test Gaussian blur."""
        for severity in range(1, 6):
            result = apply_gaussian_blur(sample_image, severity)

            assert result.shape == sample_image.shape
            assert result.dtype == sample_image.dtype

    def test_gaussian_blur_kernel_size(self, sample_image):
        """Test that blur increases with kernel size."""
        blur_1 = apply_gaussian_blur(sample_image, severity=1)
        blur_5 = apply_gaussian_blur(sample_image, severity=5)

        # Higher blur should have less variance (more uniform)
        assert np.var(blur_5) < np.var(blur_1)

    def test_gaussian_noise(self, sample_image):
        """Test Gaussian noise addition."""
        for severity in range(1, 6):
            result = apply_gaussian_noise(sample_image, severity, seed=42)

            assert result.shape == sample_image.shape
            assert result.dtype == sample_image.dtype

    def test_gaussian_noise_reproducible(self, sample_image):
        """Test that noise is reproducible with seed."""
        result1 = apply_gaussian_noise(sample_image, severity=3, seed=42)
        result2 = apply_gaussian_noise(sample_image, severity=3, seed=42)

        np.testing.assert_array_equal(result1, result2)

    def test_gaussian_noise_different_seeds(self, sample_image):
        """Test that different seeds give different results."""
        result1 = apply_gaussian_noise(sample_image, severity=3, seed=42)
        result2 = apply_gaussian_noise(sample_image, severity=3, seed=123)

        assert not np.array_equal(result1, result2)

    def test_motion_blur(self, sample_image):
        """Test motion blur."""
        for severity in range(1, 6):
            result = apply_motion_blur(sample_image, severity, seed=42)

            assert result.shape == sample_image.shape
            assert result.dtype == sample_image.dtype

    def test_motion_blur_with_angle(self, sample_image):
        """Test motion blur with specific angle."""
        result_h = apply_motion_blur(sample_image, severity=3, angle=0)
        result_v = apply_motion_blur(sample_image, severity=3, angle=90)

        # Different angles should give different results
        assert not np.array_equal(result_h, result_v)

    def test_occlusion(self, sample_image):
        """Test random occlusion."""
        for severity in range(1, 6):
            result = apply_occlusion(sample_image, severity, seed=42)

            assert result.shape == sample_image.shape
            assert result.dtype == sample_image.dtype

            # Some pixels should be black (occluded)
            if severity > 1:
                assert np.sum(result == 0) > np.sum(sample_image == 0)

    def test_occlusion_coverage(self, sample_image):
        """Test that occlusion coverage matches severity."""
        # Use multiple patches to better approximate coverage
        result = apply_occlusion(sample_image, severity=3, n_patches=3, seed=42)

        # Count occluded pixels (black)
        total_pixels = sample_image.shape[0] * sample_image.shape[1]
        occluded = np.sum(np.all(result == 0, axis=2))

        # Should be roughly 25% (severity 3)
        expected_coverage = SEVERITY_CONFIGS["occlusion"][2]  # 0.25
        # Allow 50% tolerance due to random placement
        assert 0.1 < occluded / total_pixels < 0.5

    def test_jpeg_compression(self, sample_image):
        """Test JPEG compression artifacts."""
        for severity in range(1, 6):
            result = apply_jpeg_compression(sample_image, severity)

            assert result.shape == sample_image.shape
            assert result.dtype == sample_image.dtype

    def test_jpeg_compression_quality(self, sample_image):
        """Test that lower quality creates more artifacts."""
        high_quality = apply_jpeg_compression(sample_image, severity=1)
        low_quality = apply_jpeg_compression(sample_image, severity=5)

        # Lower quality should differ more from original
        diff_high = np.mean(np.abs(high_quality.astype(float) - sample_image.astype(float)))
        diff_low = np.mean(np.abs(low_quality.astype(float) - sample_image.astype(float)))

        assert diff_low > diff_high

    def test_pixelation(self, sample_image):
        """Test pixelation."""
        for severity in range(1, 6):
            result = apply_pixelation(sample_image, severity)

            assert result.shape == sample_image.shape
            assert result.dtype == sample_image.dtype

    def test_pixelation_factor(self, sample_image):
        """Test that higher factors create more pixelation."""
        pix_1 = apply_pixelation(sample_image, severity=1)
        pix_5 = apply_pixelation(sample_image, severity=5)

        # Severity 1 should be nearly identical to original
        np.testing.assert_array_equal(pix_1, sample_image)

        # Severity 5 should be very different
        assert not np.array_equal(pix_5, sample_image)


class TestApplyDegradation:
    """Test the generic apply_degradation function."""

    def test_apply_by_name(self, sample_image):
        """Test applying degradation by name."""
        result = apply_degradation(sample_image, "blur", severity=3)
        assert result.shape == sample_image.shape

    def test_apply_with_alias(self, sample_image):
        """Test that aliases work."""
        result1 = apply_degradation(sample_image, "blur", severity=3)
        result2 = apply_degradation(sample_image, "gaussian_blur", severity=3)

        np.testing.assert_array_equal(result1, result2)

    def test_invalid_degradation(self, sample_image):
        """Test that invalid degradation raises error."""
        with pytest.raises(ValueError, match="Unknown degradation"):
            apply_degradation(sample_image, "invalid_degradation", severity=3)

    def test_invalid_severity(self, sample_image):
        """Test that invalid severity raises error."""
        with pytest.raises(ValueError, match="Severity must be"):
            apply_degradation(sample_image, "blur", severity=0)

        with pytest.raises(ValueError, match="Severity must be"):
            apply_degradation(sample_image, "blur", severity=6)


class TestDegradationInfo:
    """Test degradation info retrieval."""

    def test_get_info(self):
        """Test getting degradation info."""
        info = get_degradation_info("gaussian_blur")

        assert info["name"] == "gaussian_blur"
        assert len(info["severity_levels"]) == 5
        assert info["n_levels"] == 5
        assert "description" in info

    def test_get_info_with_alias(self):
        """Test getting info with alias."""
        info = get_degradation_info("blur")
        assert info["name"] == "gaussian_blur"


class TestDegradationPipeline:
    """Test the composable DegradationPipeline."""

    def test_empty_pipeline(self, sample_image):
        """Test empty pipeline returns unchanged image."""
        pipeline = DegradationPipeline([])
        result = pipeline(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    def test_single_step_pipeline(self, sample_image):
        """Test single-step pipeline."""
        pipeline = DegradationPipeline([("blur", 3)])
        result = pipeline(sample_image)

        expected = apply_gaussian_blur(sample_image, severity=3)
        np.testing.assert_array_equal(result, expected)

    def test_multi_step_pipeline(self, sample_image):
        """Test multi-step pipeline."""
        pipeline = DegradationPipeline([
            ("lighting", 2),
            ("blur", 3),
        ])
        result = pipeline(sample_image)

        assert result.shape == sample_image.shape

    def test_pipeline_add_method(self, sample_image):
        """Test adding steps with add() method."""
        pipeline = DegradationPipeline()
        pipeline.add("noise", severity=2)
        pipeline.add("jpeg", severity=3)

        assert len(pipeline) == 2
        result = pipeline(sample_image)
        assert result.shape == sample_image.shape

    def test_pipeline_with_seed(self, sample_image):
        """Test reproducible pipeline with seed."""
        pipeline = DegradationPipeline([
            ("noise", 3),
            ("motion_blur", 2),
        ], seed=42)

        result1 = pipeline(sample_image)
        result2 = pipeline(sample_image)

        np.testing.assert_array_equal(result1, result2)

    def test_pipeline_copy(self):
        """Test pipeline copy."""
        pipeline = DegradationPipeline([("blur", 3)])
        copy = pipeline.copy()

        assert len(copy) == len(pipeline)
        copy.add("noise", severity=2)
        assert len(copy) != len(pipeline)

    def test_pipeline_serialization(self):
        """Test pipeline to/from dict."""
        pipeline = DegradationPipeline([
            ("lighting", 2),
            ("blur", 3),
        ], seed=42)

        d = pipeline.to_dict()
        restored = DegradationPipeline.from_dict(d)

        assert len(restored) == len(pipeline)
        assert restored.seed == pipeline.seed

    def test_pipeline_name(self):
        """Test pipeline name generation."""
        pipeline = DegradationPipeline([
            ("lighting", 2),
            ("blur", 3),
        ])
        name = pipeline.get_name()

        assert "lighting2" in name
        assert "blur3" in name

    def test_empty_pipeline_name(self):
        """Test empty pipeline name."""
        pipeline = DegradationPipeline([])
        assert pipeline.get_name() == "clean"


class TestTestCaseGeneration:
    """Test test case generation functions."""

    def test_generate_test_matrix(self):
        """Test full test matrix generation."""
        cases = generate_test_matrix(
            include_clean=True,
            include_single=True,
            include_combinations=True
        )

        # Should have clean + (7 degradations * 5 severities) + combinations
        assert len(cases) > 35

        # Check for clean case
        clean_cases = [c for c in cases if c.degradation_type == "clean"]
        assert len(clean_cases) == 1

    def test_generate_single_degradation_cases(self):
        """Test single degradation case generation."""
        cases = generate_single_degradation_cases(
            degradation_types=["blur", "noise"],
            severity_levels=[1, 3, 5]
        )

        assert len(cases) == 6  # 2 types * 3 severities

        for case in cases:
            assert case.degradation_type == "single"
            assert case.severity in [1, 3, 5]

    def test_generate_severity_sweep(self):
        """Test severity sweep generation."""
        cases = generate_severity_sweep("gaussian_blur")

        assert len(cases) == 5  # All 5 severity levels

        severities = [c.severity for c in cases]
        assert severities == [1, 2, 3, 4, 5]

    def test_test_case_apply(self, sample_image):
        """Test TestCase.apply()."""
        case = TestCase(
            pipeline=DegradationPipeline([("blur", 3)]),
            name="blur_s3",
            degradation_type="single",
            severity=3
        )

        result = case.apply(sample_image)
        assert result.shape == sample_image.shape


class TestDegradationStep:
    """Test DegradationStep dataclass."""

    def test_valid_step(self):
        """Test creating valid step."""
        step = DegradationStep("blur", 3)
        assert step.degradation_type == "blur"
        assert step.severity == 3

    def test_invalid_degradation_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError):
            DegradationStep("invalid", 3)

    def test_invalid_severity(self):
        """Test that invalid severity raises error."""
        with pytest.raises(ValueError):
            DegradationStep("blur", 0)

        with pytest.raises(ValueError):
            DegradationStep("blur", 6)


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_tiny_image(self):
        """Test with very small image."""
        tiny = np.zeros((8, 8, 3), dtype=np.uint8)
        tiny[:, :] = 128

        for deg_type in ["lighting", "blur", "noise", "jpeg", "pixelation"]:
            result = apply_degradation(tiny, deg_type, severity=3)
            assert result.shape == tiny.shape

    def test_large_image(self):
        """Test with larger image."""
        large = np.zeros((1024, 1024, 3), dtype=np.uint8)
        large[:, :] = 128

        result = apply_degradation(large, "blur", severity=3)
        assert result.shape == large.shape

    def test_grayscale_image(self):
        """Test with grayscale (2D) image."""
        gray = np.zeros((256, 256), dtype=np.uint8)
        gray[:, :] = 128

        # Some transforms should work with grayscale
        result = apply_gaussian_blur(gray, severity=3)
        assert result.shape == gray.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
