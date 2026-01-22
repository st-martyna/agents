"""
Analysis module for failure mode and Pareto analysis.
"""

from .failure_modes import (
    find_high_confidence_failures,
    cluster_failures_by_degradation,
    cluster_failures_by_class,
    cluster_failures_by_embedding,
    generate_failure_report,
    visualize_failure_grid,
    compare_failure_rates,
)

from .pareto import (
    compute_pareto_frontier,
    compute_efficiency_ratio,
    plot_pareto_frontier,
    plot_edge_vs_cloud_comparison,
    compute_scaling_efficiency,
    generate_pareto_report,
)

__all__ = [
    # Failure modes
    "find_high_confidence_failures",
    "cluster_failures_by_degradation",
    "cluster_failures_by_class",
    "cluster_failures_by_embedding",
    "generate_failure_report",
    "visualize_failure_grid",
    "compare_failure_rates",
    # Pareto analysis
    "compute_pareto_frontier",
    "compute_efficiency_ratio",
    "plot_pareto_frontier",
    "plot_edge_vs_cloud_comparison",
    "compute_scaling_efficiency",
    "generate_pareto_report",
]
