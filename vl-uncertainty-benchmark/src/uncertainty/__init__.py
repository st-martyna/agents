"""
Uncertainty extraction and calibration module.

Provides methods for extracting uncertainty estimates from various model
architectures and calibrating confidence scores for improved reliability.
"""

from .extractors import (
    token_entropy,
    embedding_distance,
    diffusion_variance,
    flow_variance,
    iou_confidence,
    detection_confidence,
    logit_based_uncertainty,
    aggregate_uncertainties,
)

from .calibration import (
    compute_optimal_temperature,
    fit_platt_scaling,
    TemperatureScaling,
    PlattScaling,
    FocalLossCalibration,
    calibrate_predictions,
    cross_validate_calibration,
)

__all__ = [
    # Extractors
    "token_entropy",
    "embedding_distance",
    "diffusion_variance",
    "flow_variance",
    "iou_confidence",
    "detection_confidence",
    "logit_based_uncertainty",
    "aggregate_uncertainties",
    # Calibration
    "compute_optimal_temperature",
    "fit_platt_scaling",
    "TemperatureScaling",
    "PlattScaling",
    "FocalLossCalibration",
    "calibrate_predictions",
    "cross_validate_calibration",
]
