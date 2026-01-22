"""
Failure mode analysis for identifying patterns in model errors.

Provides tools for finding and clustering high-confidence failures,
which are particularly dangerous for robotics applications where
confident wrong predictions can lead to unsafe actions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_high_confidence_failures(
    results_df: pd.DataFrame,
    confidence_threshold: float = 0.9,
    confidence_col: str = "confidence",
    correct_col: str = "is_correct"
) -> pd.DataFrame:
    """
    Find predictions that were confident but wrong.

    High-confidence failures are particularly problematic for robotics:
    they indicate the model is confidently wrong, which could lead to
    unsafe actions if the robot trusts the prediction.

    Args:
        results_df: DataFrame with benchmark results
        confidence_threshold: Minimum confidence for "confident" predictions
        confidence_col: Column name for confidence scores
        correct_col: Column name for correctness (0 or 1)

    Returns:
        DataFrame containing only high-confidence failures
    """
    mask = (results_df[confidence_col] >= confidence_threshold) & \
           (results_df[correct_col] == 0)

    failures = results_df[mask].copy()
    failures["confidence_excess"] = failures[confidence_col] - confidence_threshold

    return failures.sort_values(confidence_col, ascending=False)


def cluster_failures_by_degradation(
    failures_df: pd.DataFrame,
    degradation_col: str = "degradation_type",
    severity_col: str = "severity"
) -> Dict[str, Dict[str, Any]]:
    """
    Cluster failures by degradation type to identify vulnerability patterns.

    Args:
        failures_df: DataFrame of failure cases
        degradation_col: Column name for degradation type
        severity_col: Column name for severity level

    Returns:
        Dictionary mapping degradation type to failure statistics:
        {
            "blur": {
                "count": 45,
                "percentage": 0.23,
                "severity_distribution": {1: 5, 2: 10, 3: 15, 4: 10, 5: 5},
                "mean_confidence": 0.92
            },
            ...
        }
    """
    clusters = {}
    total_failures = len(failures_df)

    for deg_type in failures_df[degradation_col].unique():
        deg_failures = failures_df[failures_df[degradation_col] == deg_type]

        severity_dist = deg_failures[severity_col].value_counts().to_dict()

        clusters[deg_type] = {
            "count": len(deg_failures),
            "percentage": len(deg_failures) / total_failures if total_failures > 0 else 0,
            "severity_distribution": severity_dist,
            "mean_confidence": float(deg_failures["confidence"].mean()) if "confidence" in deg_failures else None,
            "mean_severity": float(deg_failures[severity_col].mean()) if severity_col in deg_failures else None,
        }

    # Sort by count descending
    clusters = dict(sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True))

    return clusters


def cluster_failures_by_class(
    failures_df: pd.DataFrame,
    class_col: str = "ground_truth",
    top_k: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Cluster failures by ground truth class.

    Identifies which classes the model struggles with most.

    Args:
        failures_df: DataFrame of failure cases
        class_col: Column name for ground truth class
        top_k: Number of top failure classes to return

    Returns:
        Dictionary mapping class to failure statistics
    """
    clusters = {}
    total_failures = len(failures_df)

    class_counts = failures_df[class_col].value_counts()

    for cls in class_counts.head(top_k).index:
        cls_failures = failures_df[failures_df[class_col] == cls]

        clusters[str(cls)] = {
            "count": len(cls_failures),
            "percentage": len(cls_failures) / total_failures if total_failures > 0 else 0,
            "mean_confidence": float(cls_failures["confidence"].mean()) if "confidence" in cls_failures else None,
        }

        # Add common misclassifications if prediction column exists
        if "prediction" in failures_df.columns:
            pred_counts = cls_failures["prediction"].value_counts().head(3)
            clusters[str(cls)]["common_predictions"] = pred_counts.to_dict()

    return clusters


def cluster_failures_by_embedding(
    failures_df: pd.DataFrame,
    embedding_col: str = "embedding",
    n_clusters: int = 5
) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    Cluster failures by embedding similarity using K-means.

    Useful for discovering semantic patterns in failures.

    Args:
        failures_df: DataFrame with failure cases
        embedding_col: Column containing embeddings
        n_clusters: Number of clusters

    Returns:
        Tuple of (cluster_labels, cluster_info)
    """
    if embedding_col not in failures_df.columns:
        raise ValueError(f"Embedding column '{embedding_col}' not found")

    embeddings = np.stack(failures_df[embedding_col].values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    cluster_info = {}
    for i in range(n_clusters):
        mask = labels == i
        cluster_failures = failures_df.iloc[mask]

        cluster_info[i] = {
            "count": int(mask.sum()),
            "centroid": kmeans.cluster_centers_[i],
            "mean_confidence": float(cluster_failures["confidence"].mean()) if "confidence" in cluster_failures else None,
        }

        # Add common degradations in this cluster
        if "degradation_type" in failures_df.columns:
            deg_counts = cluster_failures["degradation_type"].value_counts().head(3)
            cluster_info[i]["top_degradations"] = deg_counts.to_dict()

    return labels, cluster_info


def generate_failure_report(
    failures_df: pd.DataFrame,
    total_samples: Optional[int] = None,
    confidence_col: str = "confidence",
    degradation_col: str = "degradation_type",
    severity_col: str = "severity",
    class_col: str = "ground_truth"
) -> Dict[str, Any]:
    """
    Generate comprehensive failure analysis report.

    Args:
        failures_df: DataFrame of failure cases
        total_samples: Total number of samples (for percentage calculation)
        confidence_col: Confidence column name
        degradation_col: Degradation type column name
        severity_col: Severity column name
        class_col: Ground truth class column name

    Returns:
        Comprehensive report dictionary
    """
    n_failures = len(failures_df)

    report = {
        "summary": {
            "total_failures": n_failures,
            "failure_rate": n_failures / total_samples if total_samples else None,
        },
        "confidence_stats": {},
        "degradation_analysis": {},
        "class_analysis": {},
        "severity_analysis": {},
        "recommendations": [],
    }

    # Confidence statistics
    if confidence_col in failures_df.columns:
        confidences = failures_df[confidence_col]
        report["confidence_stats"] = {
            "mean": float(confidences.mean()),
            "std": float(confidences.std()),
            "min": float(confidences.min()),
            "max": float(confidences.max()),
            "median": float(confidences.median()),
            "above_95": int((confidences >= 0.95).sum()),
            "above_99": int((confidences >= 0.99).sum()),
        }

    # Degradation analysis
    if degradation_col in failures_df.columns:
        report["degradation_analysis"] = cluster_failures_by_degradation(
            failures_df, degradation_col, severity_col
        )

        # Find most problematic degradation
        if report["degradation_analysis"]:
            worst_deg = max(
                report["degradation_analysis"].items(),
                key=lambda x: x[1]["count"]
            )
            report["recommendations"].append(
                f"Model is most vulnerable to '{worst_deg[0]}' degradation "
                f"({worst_deg[1]['count']} failures, {worst_deg[1]['percentage']:.1%})"
            )

    # Class analysis
    if class_col in failures_df.columns:
        report["class_analysis"] = cluster_failures_by_class(
            failures_df, class_col, top_k=5
        )

    # Severity analysis
    if severity_col in failures_df.columns:
        severity_counts = failures_df[severity_col].value_counts().sort_index()
        report["severity_analysis"] = {
            "distribution": severity_counts.to_dict(),
            "mean_severity": float(failures_df[severity_col].mean()),
        }

        # Check if failures increase with severity
        if len(severity_counts) > 2:
            severities = severity_counts.index.tolist()
            counts = severity_counts.values.tolist()
            if counts[-1] > counts[0]:
                report["recommendations"].append(
                    "Failure rate increases with degradation severity - "
                    "consider more robust preprocessing or training augmentation"
                )

    # Add general recommendations
    if report["confidence_stats"].get("above_95", 0) > n_failures * 0.5:
        report["recommendations"].append(
            "Many failures have >95% confidence - model is poorly calibrated. "
            "Temperature scaling or Platt scaling recommended."
        )

    return report


def visualize_failure_grid(
    failures_df: pd.DataFrame,
    image_col: str = "image_path",
    n_examples: int = 16,
    figsize: Tuple[int, int] = (16, 16),
    title: str = "High-Confidence Failures"
) -> plt.Figure:
    """
    Visualize a grid of failure examples.

    Args:
        failures_df: DataFrame with failure cases
        image_col: Column containing image paths or arrays
        n_examples: Number of examples to show
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    import cv2

    n_cols = int(np.ceil(np.sqrt(n_examples)))
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_examples > 1 else [axes]

    # Sort by confidence (highest first)
    sorted_failures = failures_df.sort_values("confidence", ascending=False)

    for idx, ax in enumerate(axes):
        if idx < min(n_examples, len(sorted_failures)):
            row = sorted_failures.iloc[idx]

            # Load image
            if image_col in row:
                img_data = row[image_col]
                if isinstance(img_data, str):
                    # It's a path
                    img = cv2.imread(img_data)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # It's an array
                    img = img_data
                    if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if img is not None:
                    ax.imshow(img)

            # Add annotations
            conf = row.get("confidence", 0)
            pred = row.get("prediction", "?")
            gt = row.get("ground_truth", "?")
            deg = row.get("degradation_type", "")
            sev = row.get("severity", "")

            title_str = f"Conf: {conf:.2f}\nPred: {pred}\nGT: {gt}"
            if deg:
                title_str += f"\n{deg} s{sev}"

            ax.set_title(title_str, fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def compare_failure_rates(
    results_df: pd.DataFrame,
    model_col: str = "model",
    confidence_thresholds: List[float] = [0.5, 0.7, 0.9, 0.95]
) -> pd.DataFrame:
    """
    Compare high-confidence failure rates across models.

    Args:
        results_df: DataFrame with benchmark results
        model_col: Column name for model identifier
        confidence_thresholds: List of confidence thresholds to analyze

    Returns:
        DataFrame with failure rates per model and threshold
    """
    results = []

    for model in results_df[model_col].unique():
        model_df = results_df[results_df[model_col] == model]
        model_correct = model_df["is_correct"]
        model_conf = model_df["confidence"]

        row = {"model": model, "total_samples": len(model_df)}

        for thresh in confidence_thresholds:
            confident_mask = model_conf >= thresh
            n_confident = confident_mask.sum()
            n_confident_wrong = ((model_conf >= thresh) & (model_correct == 0)).sum()

            row[f"n_conf_{thresh}"] = n_confident
            row[f"failure_rate_{thresh}"] = n_confident_wrong / n_confident if n_confident > 0 else 0

        results.append(row)

    return pd.DataFrame(results)
