"""
Metrics module for calibration and uncertainty evaluation.
"""

from .calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    reliability_diagram,
    auroc_uncertainty,
    compute_calibration_metrics,
    calibration_curve,
    plot_calibration_comparison,
    compute_calibration_by_degradation,
)

__all__ = [
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "reliability_diagram",
    "auroc_uncertainty",
    "compute_calibration_metrics",
    "calibration_curve",
    "plot_calibration_comparison",
    "compute_calibration_by_degradation",
]
