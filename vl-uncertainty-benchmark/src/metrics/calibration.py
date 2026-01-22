"""
Calibration metrics for evaluating model uncertainty quality.

Provides metrics for measuring how well model confidence aligns with
actual accuracy, and tools for visualizing calibration quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform"
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the weighted average of the difference between
    confidence and accuracy across bins.

    ECE = sum(|acc(b) - conf(b)| * n(b) / N) for each bin b

    Args:
        confidences: Model confidence scores [0, 1] of shape (n_samples,)
        accuracies: Binary correctness indicators (0 or 1) of shape (n_samples,)
        n_bins: Number of bins for confidence discretization
        strategy: Binning strategy ('uniform' or 'quantile')

    Returns:
        ECE value [0, 1]. Lower is better (0 = perfectly calibrated).
    """
    confidences = np.asarray(confidences).flatten()
    accuracies = np.asarray(accuracies).flatten()

    if len(confidences) != len(accuracies):
        raise ValueError("confidences and accuracies must have same length")

    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0

    # Create bins
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        # Quantile bins ensure equal samples per bin
        bin_edges = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    ece = 0.0
    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            # Include right edge for last bin
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += abs(bin_accuracy - bin_confidence) * n_in_bin

    return ece / n_samples


def maximum_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform"
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum difference between confidence and accuracy
    across all bins. Captures worst-case calibration.

    Args:
        confidences: Model confidence scores [0, 1]
        accuracies: Binary correctness indicators
        n_bins: Number of bins
        strategy: Binning strategy

    Returns:
        MCE value [0, 1]. Lower is better.
    """
    confidences = np.asarray(confidences).flatten()
    accuracies = np.asarray(accuracies).flatten()

    if len(confidences) == 0:
        return 0.0

    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

    max_error = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            error = abs(bin_accuracy - bin_confidence)
            max_error = max(max_error, error)

    return max_error


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Brier score measures both calibration and refinement.
    BS = mean((p - y)^2) where y in {0, 1}

    Args:
        probabilities: Predicted probabilities [0, 1]
        labels: Binary ground truth labels

    Returns:
        Brier score [0, 1]. Lower is better (0 = perfect).
    """
    probabilities = np.asarray(probabilities).flatten()
    labels = np.asarray(labels).flatten()

    return float(np.mean((probabilities - labels) ** 2))


def reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    figsize: Tuple[int, int] = (8, 6),
    show_gap: bool = True,
    show_counts: bool = True
) -> plt.Figure:
    """
    Generate a reliability diagram for visualizing calibration.

    A reliability diagram plots accuracy vs. confidence for each bin.
    A perfectly calibrated model follows the diagonal.

    Args:
        confidences: Model confidence scores [0, 1]
        accuracies: Binary correctness indicators
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size
        show_gap: Show shaded gap between perfect and actual calibration
        show_counts: Show sample counts per bin

    Returns:
        Matplotlib figure
    """
    confidences = np.asarray(confidences).flatten()
    accuracies = np.asarray(accuracies).flatten()

    # Compute bin statistics
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])

        n_in_bin = mask.sum()
        bin_counts.append(n_in_bin)

        if n_in_bin > 0:
            bin_accuracies.append(accuracies[mask].mean())
            bin_confidences.append(confidences[mask].mean())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(bin_centers[i])

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    # Plot bar chart for accuracy
    width = 1.0 / n_bins * 0.8
    bars = ax.bar(
        bin_centers,
        bin_accuracies,
        width=width,
        alpha=0.7,
        color='steelblue',
        edgecolor='navy',
        label='Accuracy'
    )

    # Show gap
    if show_gap:
        for i, (center, acc, conf) in enumerate(zip(bin_centers, bin_accuracies, bin_confidences)):
            if not np.isnan(acc):
                gap_color = 'salmon' if acc < conf else 'lightgreen'
                ax.fill_between(
                    [center - width/2, center + width/2],
                    [acc, acc],
                    [min(1, conf), min(1, conf)],
                    alpha=0.3,
                    color=gap_color
                )

    # Show counts
    if show_counts:
        for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
            if count > 0 and not np.isnan(acc):
                ax.text(
                    center,
                    acc + 0.02,
                    str(count),
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

    # Compute and display ECE
    ece = expected_calibration_error(confidences, accuracies, n_bins)
    mce = maximum_calibration_error(confidences, accuracies, n_bins)

    textstr = f'ECE = {ece:.3f}\nMCE = {mce:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Labels and formatting
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def auroc_uncertainty(
    uncertainties: np.ndarray,
    is_correct: np.ndarray
) -> float:
    """
    Compute AUROC for uncertainty as an error predictor.

    High AUROC means uncertainty reliably predicts model errors.
    Uncertainty should be HIGH when model is WRONG.

    Args:
        uncertainties: Uncertainty scores (higher = more uncertain)
        is_correct: Binary correctness (1 = correct, 0 = error)

    Returns:
        AUROC value [0, 1]. 0.5 = random, 1.0 = perfect error prediction.
    """
    from sklearn.metrics import roc_auc_score

    uncertainties = np.asarray(uncertainties).flatten()
    is_correct = np.asarray(is_correct).flatten()

    # We want to predict errors (is_correct == 0)
    # So high uncertainty should correspond to is_correct == 0
    is_error = 1 - is_correct

    try:
        return float(roc_auc_score(is_error, uncertainties))
    except ValueError:
        # Handle edge cases (e.g., all same class)
        return 0.5


def compute_calibration_metrics(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Compute all calibration metrics in one call.

    Args:
        confidences: Model confidence scores
        accuracies: Binary correctness indicators
        uncertainties: Optional uncertainty scores for AUROC
        n_bins: Number of bins for ECE/MCE

    Returns:
        Dictionary with all calibration metrics
    """
    confidences = np.asarray(confidences).flatten()
    accuracies = np.asarray(accuracies).flatten()

    metrics = {
        "ece": expected_calibration_error(confidences, accuracies, n_bins),
        "mce": maximum_calibration_error(confidences, accuracies, n_bins),
        "brier_score": brier_score(confidences, accuracies),
        "mean_confidence": float(np.mean(confidences)),
        "mean_accuracy": float(np.mean(accuracies)),
        "overconfidence": float(np.mean(confidences) - np.mean(accuracies)),
        "n_samples": len(confidences),
    }

    if uncertainties is not None:
        uncertainties = np.asarray(uncertainties).flatten()
        metrics["auroc_uncertainty"] = auroc_uncertainty(uncertainties, accuracies)

    return metrics


def calibration_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.

    Args:
        confidences: Model confidence scores
        accuracies: Binary correctness indicators
        n_bins: Number of bins

    Returns:
        Tuple of (mean_confidence_per_bin, accuracy_per_bin, count_per_bin)
    """
    confidences = np.asarray(confidences).flatten()
    accuracies = np.asarray(accuracies).flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)

    mean_confs = []
    mean_accs = []
    counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])

        count = mask.sum()
        counts.append(count)

        if count > 0:
            mean_confs.append(confidences[mask].mean())
            mean_accs.append(accuracies[mask].mean())
        else:
            mean_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            mean_accs.append(np.nan)

    return np.array(mean_confs), np.array(mean_accs), np.array(counts)


def plot_calibration_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 15,
    title: str = "Calibration Comparison",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Compare calibration across multiple models.

    Args:
        results: Dict mapping model names to (confidences, accuracies) tuples
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, (confs, accs)), color in zip(results.items(), colors):
        mean_confs, mean_accs, _ = calibration_curve(confs, accs, n_bins)
        ece = expected_calibration_error(confs, accs, n_bins)

        # Filter out NaN values
        valid = ~np.isnan(mean_accs)
        ax.plot(
            mean_confs[valid],
            mean_accs[valid],
            'o-',
            color=color,
            label=f'{name} (ECE={ece:.3f})',
            linewidth=2,
            markersize=6
        )

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_calibration_by_degradation(
    results_df,  # pandas DataFrame
    confidence_col: str = "confidence",
    correct_col: str = "is_correct",
    degradation_col: str = "degradation_type",
    n_bins: int = 15
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration metrics grouped by degradation type.

    Args:
        results_df: DataFrame with benchmark results
        confidence_col: Column name for confidence scores
        correct_col: Column name for correctness
        degradation_col: Column name for degradation type
        n_bins: Number of bins

    Returns:
        Dict mapping degradation type to calibration metrics
    """
    metrics_by_degradation = {}

    for deg_type in results_df[degradation_col].unique():
        subset = results_df[results_df[degradation_col] == deg_type]
        confs = subset[confidence_col].values
        accs = subset[correct_col].values

        metrics_by_degradation[deg_type] = compute_calibration_metrics(
            confs, accs, n_bins=n_bins
        )

    return metrics_by_degradation
