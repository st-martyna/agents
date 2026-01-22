"""
Pareto analysis for compute vs. uncertainty quality tradeoffs.

Helps answer: "Does scaling up models give meaningfully better
uncertainty calibration, or are edge models sufficient for
reactive→deliberative mode switching?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt


def compute_pareto_frontier(
    results_df: pd.DataFrame,
    x: str = "params",
    y: str = "uncertainty_auroc",
    minimize_x: bool = True,
    maximize_y: bool = True
) -> pd.DataFrame:
    """
    Compute Pareto frontier for compute vs. quality tradeoff.

    Points on the Pareto frontier are optimal in the sense that no other
    point is better on both dimensions.

    Args:
        results_df: DataFrame with model results (one row per model)
        x: Column name for compute cost (e.g., 'params', 'inference_time_ms')
        y: Column name for quality metric (e.g., 'uncertainty_auroc', 'ece')
        minimize_x: Whether lower x is better (True for params, time)
        maximize_y: Whether higher y is better (True for AUROC, False for ECE)

    Returns:
        DataFrame containing only Pareto-optimal points
    """
    df = results_df.copy()

    # Normalize directions (we want to maximize both for comparison)
    x_vals = df[x].values
    y_vals = df[y].values

    if minimize_x:
        x_vals = -x_vals  # Negate so "larger" is better
    if not maximize_y:
        y_vals = -y_vals  # Negate so "larger" is better

    # Find Pareto frontier
    is_pareto = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                # Check if j dominates i
                if x_vals[j] >= x_vals[i] and y_vals[j] >= y_vals[i]:
                    if x_vals[j] > x_vals[i] or y_vals[j] > y_vals[i]:
                        is_pareto[i] = False
                        break

    return df[is_pareto].copy()


def compute_efficiency_ratio(
    results_df: pd.DataFrame,
    quality_col: str = "uncertainty_auroc",
    cost_col: str = "params",
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute quality-per-cost efficiency ratio.

    Higher efficiency = better quality for the compute cost.

    Args:
        results_df: DataFrame with model results
        quality_col: Column for quality metric
        cost_col: Column for compute cost
        normalize: Normalize to [0, 1] range

    Returns:
        DataFrame with efficiency ratio column added
    """
    df = results_df.copy()

    quality = df[quality_col].values
    cost = df[cost_col].values

    # Avoid division by zero
    cost = np.maximum(cost, 1e-10)

    efficiency = quality / cost

    if normalize:
        efficiency = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min() + 1e-10)

    df["efficiency_ratio"] = efficiency
    return df


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    x: str = "params",
    y: str = "uncertainty_auroc",
    model_col: str = "model",
    tier_col: Optional[str] = "deployment_tier",
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Pareto Frontier: Compute vs. Uncertainty Quality",
    log_x: bool = True,
    highlight_pareto: bool = True
) -> plt.Figure:
    """
    Plot Pareto frontier for compute vs. quality tradeoff.

    Args:
        results_df: DataFrame with model results
        x: Column for x-axis (compute cost)
        y: Column for y-axis (quality metric)
        model_col: Column for model names (for labels)
        tier_col: Column for deployment tier (for coloring)
        figsize: Figure size
        title: Plot title
        log_x: Use log scale for x-axis
        highlight_pareto: Highlight Pareto-optimal points

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color mapping for tiers
    tier_colors = {
        "edge": "green",
        "edge_plus": "orange",
        "cloud": "blue",
    }

    # Plot all points
    for tier in results_df[tier_col].unique() if tier_col else [None]:
        if tier_col:
            mask = results_df[tier_col] == tier
            subset = results_df[mask]
            color = tier_colors.get(tier, "gray")
            label = tier
        else:
            subset = results_df
            color = "steelblue"
            label = "models"

        ax.scatter(
            subset[x],
            subset[y],
            c=color,
            s=100,
            alpha=0.7,
            label=label,
            edgecolors="black",
            linewidths=0.5
        )

        # Add model labels
        for _, row in subset.iterrows():
            ax.annotate(
                row[model_col],
                (row[x], row[y]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8
            )

    # Highlight Pareto frontier
    if highlight_pareto:
        pareto = compute_pareto_frontier(
            results_df, x=x, y=y,
            minimize_x=True,
            maximize_y=(y != "ece")  # ECE should be minimized
        )

        # Sort for line plot
        pareto_sorted = pareto.sort_values(x)

        ax.plot(
            pareto_sorted[x],
            pareto_sorted[y],
            'r--',
            linewidth=2,
            label='Pareto Frontier',
            alpha=0.7
        )

        ax.scatter(
            pareto[x],
            pareto[y],
            c='red',
            s=200,
            marker='*',
            zorder=5,
            edgecolors="darkred",
            linewidths=1
        )

    # Formatting
    if log_x:
        ax.set_xscale('log')

    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_edge_vs_cloud_comparison(
    results_df: pd.DataFrame,
    model_families: Optional[Dict[str, List[str]]] = None,
    metric_col: str = "uncertainty_auroc",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Compare edge vs. cloud models within the same model family.

    Answers: How much do we lose by using edge vs. cloud versions?

    Args:
        results_df: DataFrame with model results
        model_families: Dict mapping family name to [edge_model, cloud_model]
                       If None, auto-detect from data
        metric_col: Quality metric to compare
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if model_families is None:
        # Auto-detect families based on common prefixes
        model_families = {
            "Florence-2": ["florence2_base", "florence2_large"],
            "PaliGemma 2": ["paligemma2_3b", "paligemma2_28b"],
            "Qwen2.5-VL": ["qwen25vl_3b", "qwen25vl_72b"],
            "LLaVA-OneVision": ["llava_onevision_05b", "llava_onevision_72b"],
            "DINOv2": ["dinov2_b", "dinov2_g"],
            "Octo": ["octo_small", "octo_base"],
            "SAM": ["sam", "sam2_large"],
            "YOLO-World": ["yolo_world_s", "yolo_world_x"],
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Data for plotting
    families = []
    edge_values = []
    cloud_values = []
    edge_params = []
    cloud_params = []

    for family, (edge_model, cloud_model) in model_families.items():
        edge_row = results_df[results_df["model"] == edge_model]
        cloud_row = results_df[results_df["model"] == cloud_model]

        if len(edge_row) > 0 and len(cloud_row) > 0:
            families.append(family)
            edge_values.append(edge_row[metric_col].values[0])
            cloud_values.append(cloud_row[metric_col].values[0])
            edge_params.append(edge_row["params"].values[0] if "params" in edge_row else 0)
            cloud_params.append(cloud_row["params"].values[0] if "params" in cloud_row else 0)

    if not families:
        ax1.text(0.5, 0.5, "No matching model families found",
                ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No matching model families found",
                ha='center', va='center', transform=ax2.transAxes)
        return fig

    x = np.arange(len(families))
    width = 0.35

    # Plot 1: Absolute comparison
    bars1 = ax1.bar(x - width/2, edge_values, width, label='Edge', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, cloud_values, width, label='Cloud', color='blue', alpha=0.7)

    ax1.set_ylabel(metric_col.replace("_", " ").title())
    ax1.set_title('Edge vs Cloud: Absolute Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Relative difference (percentage drop from cloud to edge)
    relative_diff = [(c - e) / c * 100 if c > 0 else 0
                     for e, c in zip(edge_values, cloud_values)]
    param_ratio = [c / e if e > 0 else 0 for e, c in zip(edge_params, cloud_params)]

    colors = ['red' if d > 10 else 'orange' if d > 5 else 'green' for d in relative_diff]
    bars3 = ax2.bar(x, relative_diff, width * 2, color=colors, alpha=0.7)

    # Add param ratio annotations
    for i, (rect, ratio) in enumerate(zip(bars3, param_ratio)):
        height = rect.get_height()
        ax2.annotate(
            f'{ratio:.0f}x params',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=8
        )

    ax2.set_ylabel('Performance Drop (%)')
    ax2.set_title('Cloud→Edge: Performance Loss vs. Compute Savings')
    ax2.set_xticks(x)
    ax2.set_xticklabels(families, rotation=45, ha='right')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def compute_scaling_efficiency(
    results_df: pd.DataFrame,
    param_col: str = "params",
    quality_col: str = "uncertainty_auroc"
) -> Dict[str, Any]:
    """
    Analyze how quality scales with model size.

    Answers: Is the scaling curve sublinear, linear, or superlinear?

    Args:
        results_df: DataFrame with model results
        param_col: Parameter count column
        quality_col: Quality metric column

    Returns:
        Dictionary with scaling analysis results
    """
    params = np.log10(results_df[param_col].values)  # Log scale
    quality = results_df[quality_col].values

    # Linear regression on log scale
    coeffs = np.polyfit(params, quality, 1)
    slope = coeffs[0]

    # Compute R² for fit quality
    predicted = np.polyval(coeffs, params)
    ss_res = np.sum((quality - predicted) ** 2)
    ss_tot = np.sum((quality - np.mean(quality)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Interpretation
    if slope < 0.01:
        interpretation = "Minimal scaling benefit - edge models may be sufficient"
    elif slope < 0.05:
        interpretation = "Moderate scaling benefit - consider task requirements"
    else:
        interpretation = "Strong scaling benefit - larger models significantly better"

    return {
        "slope": float(slope),
        "r_squared": float(r_squared),
        "interpretation": interpretation,
        "quality_range": {
            "min": float(quality.min()),
            "max": float(quality.max()),
            "spread": float(quality.max() - quality.min())
        },
        "param_range": {
            "min": float(results_df[param_col].min()),
            "max": float(results_df[param_col].max()),
            "ratio": float(results_df[param_col].max() / results_df[param_col].min())
        }
    }


def generate_pareto_report(
    results_df: pd.DataFrame,
    metrics: List[str] = ["uncertainty_auroc", "ece"],
    cost_metrics: List[str] = ["params", "inference_time_ms"]
) -> Dict[str, Any]:
    """
    Generate comprehensive Pareto analysis report.

    Args:
        results_df: DataFrame with model results
        metrics: Quality metrics to analyze
        cost_metrics: Cost metrics to analyze

    Returns:
        Comprehensive report dictionary
    """
    report = {
        "summary": {},
        "pareto_frontiers": {},
        "efficiency_analysis": {},
        "scaling_analysis": {},
        "recommendations": []
    }

    # Summary statistics
    report["summary"] = {
        "n_models": len(results_df),
        "n_edge": len(results_df[results_df["deployment_tier"] == "edge"]) if "deployment_tier" in results_df else None,
        "n_cloud": len(results_df[results_df["deployment_tier"] == "cloud"]) if "deployment_tier" in results_df else None,
    }

    # Pareto frontiers for each metric/cost combination
    for metric in metrics:
        if metric in results_df.columns:
            for cost in cost_metrics:
                if cost in results_df.columns:
                    key = f"{metric}_vs_{cost}"
                    pareto = compute_pareto_frontier(
                        results_df, x=cost, y=metric,
                        maximize_y=(metric != "ece")
                    )
                    report["pareto_frontiers"][key] = {
                        "n_pareto_optimal": len(pareto),
                        "pareto_models": pareto["model"].tolist() if "model" in pareto else [],
                    }

    # Scaling analysis
    for metric in metrics:
        if metric in results_df.columns and "params" in results_df.columns:
            report["scaling_analysis"][metric] = compute_scaling_efficiency(
                results_df, "params", metric
            )

    # Recommendations
    for metric, analysis in report["scaling_analysis"].items():
        if analysis["slope"] < 0.01:
            report["recommendations"].append(
                f"For {metric}: Scaling provides minimal benefit. "
                f"Edge models achieve {analysis['quality_range']['min']:.3f}-"
                f"{analysis['quality_range']['max']:.3f}, a spread of only "
                f"{analysis['quality_range']['spread']:.3f}."
            )

    return report
