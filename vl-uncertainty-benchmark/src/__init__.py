"""
VL-Uncertainty-Benchmark: Vision Model Uncertainty Calibration Benchmark.

A systematic framework for benchmarking vision model uncertainty calibration
under degraded conditions, designed for robotics middleware that needs to
determine when to switch from reactive to deliberative control.

Main modules:
- degradation: Image degradation transforms and pipelines
- models: Model wrappers for various vision architectures
- uncertainty: Uncertainty extraction and calibration methods
- metrics: Calibration metrics (ECE, MCE, Brier, AUROC)
- analysis: Failure mode and Pareto analysis
"""

__version__ = "0.1.0"

from . import degradation
from . import models
from . import uncertainty
from . import metrics
from . import analysis

__all__ = [
    "degradation",
    "models",
    "uncertainty",
    "metrics",
    "analysis",
]
