"""
Degradation module for applying image quality degradations.

Provides individual transform functions and a composable pipeline for
systematically degrading images to test model robustness and uncertainty
calibration.
"""

from .transforms import (
    apply_degradation,
    apply_lighting_change,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_motion_blur,
    apply_occlusion,
    apply_jpeg_compression,
    apply_pixelation,
    get_degradation_info,
    DEGRADATION_FUNCTIONS,
    SEVERITY_CONFIGS,
)

from .pipeline import (
    DegradationPipeline,
    DegradationStep,
    TestCase,
    generate_test_matrix,
    generate_single_degradation_cases,
    generate_combination_cases,
    generate_severity_sweep,
    iterate_degraded_images,
    get_test_matrix_summary,
)

__all__ = [
    # Transforms
    "apply_degradation",
    "apply_lighting_change",
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_motion_blur",
    "apply_occlusion",
    "apply_jpeg_compression",
    "apply_pixelation",
    "get_degradation_info",
    "DEGRADATION_FUNCTIONS",
    "SEVERITY_CONFIGS",
    # Pipeline
    "DegradationPipeline",
    "DegradationStep",
    "TestCase",
    "generate_test_matrix",
    "generate_single_degradation_cases",
    "generate_combination_cases",
    "generate_severity_sweep",
    "iterate_degraded_images",
    "get_test_matrix_summary",
]
