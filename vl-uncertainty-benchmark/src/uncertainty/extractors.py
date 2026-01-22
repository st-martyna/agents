"""
Uncertainty extraction methods for different model types.

Each model architecture requires specific methods to extract meaningful
uncertainty estimates:
- VLMs/VLAs: Token entropy from autoregressive decoding
- Self-supervised: Embedding distance to learned centroids
- Detection: IoU prediction confidence
- Diffusion/Flow: Variance across sampled trajectories
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy


def token_entropy(
    logits: np.ndarray,
    temperature: float = 1.0,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute entropy-based uncertainty from token logits.

    For autoregressive VLMs and VLAs, higher entropy indicates lower
    confidence in the next token prediction.

    Args:
        logits: Token logits of shape (seq_len, vocab_size) or (vocab_size,)
        temperature: Temperature for softmax (lower = sharper distribution)
        normalize: If True, normalize entropy to [0, 1] range

    Returns:
        Dictionary with:
        - token_entropy: Mean entropy across tokens (or single token entropy)
        - max_entropy: Maximum entropy in sequence
        - min_entropy: Minimum entropy in sequence
        - sequence_entropy: Entropy of mean probabilities
    """
    # Handle single token case
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]

    seq_len, vocab_size = logits.shape

    # Apply temperature
    scaled_logits = logits / temperature

    # Compute probabilities
    probs = softmax(scaled_logits, axis=-1)

    # Compute per-token entropy
    token_entropies = []
    for t in range(seq_len):
        h = scipy_entropy(probs[t], base=2)
        token_entropies.append(h)

    token_entropies = np.array(token_entropies)

    # Maximum possible entropy (uniform distribution)
    max_possible_entropy = np.log2(vocab_size) if normalize else 1.0

    # Normalize entropies
    if normalize and max_possible_entropy > 0:
        token_entropies = token_entropies / max_possible_entropy

    # Sequence-level entropy (entropy of mean probabilities)
    mean_probs = probs.mean(axis=0)
    seq_entropy = scipy_entropy(mean_probs, base=2)
    if normalize and max_possible_entropy > 0:
        seq_entropy = seq_entropy / max_possible_entropy

    return {
        "token_entropy": float(np.mean(token_entropies)),
        "max_entropy": float(np.max(token_entropies)),
        "min_entropy": float(np.min(token_entropies)),
        "sequence_entropy": float(seq_entropy),
        "entropy_std": float(np.std(token_entropies)),
    }


def embedding_distance(
    embedding: np.ndarray,
    centroids: np.ndarray,
    metric: str = "cosine"
) -> Dict[str, float]:
    """
    Compute uncertainty from embedding distance to class centroids.

    For self-supervised models (DINOv2, V-JEPA), embeddings far from
    learned centroids indicate out-of-distribution or uncertain samples.

    Args:
        embedding: Query embedding of shape (embed_dim,)
        centroids: Class centroids of shape (n_classes, embed_dim)
        metric: Distance metric ('cosine', 'euclidean', 'mahalanobis')

    Returns:
        Dictionary with:
        - min_distance: Distance to nearest centroid
        - mean_distance: Mean distance to all centroids
        - margin: Difference between closest and second-closest
        - assigned_class: Index of nearest centroid
    """
    embedding = np.asarray(embedding).flatten()
    centroids = np.asarray(centroids)

    if centroids.ndim == 1:
        centroids = centroids[np.newaxis, :]

    n_classes = centroids.shape[0]

    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        centroid_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        similarities = centroid_norms @ embedding_norm
        distances = 1 - similarities

    elif metric == "euclidean":
        distances = np.linalg.norm(centroids - embedding, axis=1)

    elif metric == "mahalanobis":
        # Simplified Mahalanobis using diagonal covariance
        # For full Mahalanobis, would need covariance matrices per class
        distances = np.linalg.norm(centroids - embedding, axis=1)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Sort distances
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]

    min_dist = sorted_distances[0]
    assigned_class = sorted_indices[0]

    # Margin (confidence in assignment)
    margin = sorted_distances[1] - sorted_distances[0] if n_classes > 1 else 0.0

    return {
        "min_distance": float(min_dist),
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "margin": float(margin),
        "assigned_class": int(assigned_class),
        "distance_std": float(np.std(distances)),
    }


def diffusion_variance(
    samples: np.ndarray,
    aggregation: str = "mean"
) -> Dict[str, float]:
    """
    Compute uncertainty from variance across diffusion samples.

    For diffusion-based action models (Octo), sample multiple trajectories
    and compute variance to estimate uncertainty.

    Args:
        samples: Sampled trajectories of shape (n_samples, ...) where ...
                is the action/trajectory shape
        aggregation: How to aggregate across dimensions ('mean', 'max', 'sum')

    Returns:
        Dictionary with:
        - action_variance: Aggregated variance across samples
        - per_dim_variance: Variance per action dimension (if applicable)
        - sample_std: Standard deviation of samples
    """
    samples = np.asarray(samples)
    n_samples = samples.shape[0]

    if n_samples < 2:
        return {
            "action_variance": 0.0,
            "sample_std": 0.0,
            "n_samples": n_samples,
        }

    # Compute variance along sample dimension
    variance = np.var(samples, axis=0)
    std = np.std(samples, axis=0)

    # Aggregate variance
    if aggregation == "mean":
        agg_variance = float(np.mean(variance))
        agg_std = float(np.mean(std))
    elif aggregation == "max":
        agg_variance = float(np.max(variance))
        agg_std = float(np.max(std))
    elif aggregation == "sum":
        agg_variance = float(np.sum(variance))
        agg_std = float(np.sum(std))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    result = {
        "action_variance": agg_variance,
        "sample_std": agg_std,
        "n_samples": n_samples,
        "variance_per_dim": variance.tolist() if variance.ndim <= 1 else None,
    }

    # For trajectory samples, compute temporal variance
    if samples.ndim >= 3:  # (n_samples, timesteps, action_dim)
        temporal_var = np.var(samples, axis=0).mean(axis=-1)  # variance per timestep
        result["temporal_variance"] = temporal_var.tolist()
        result["mean_temporal_variance"] = float(np.mean(temporal_var))

    return result


def flow_variance(
    samples: np.ndarray,
    aggregation: str = "mean"
) -> Dict[str, float]:
    """
    Compute uncertainty from variance across flow matching samples.

    Similar to diffusion_variance but for flow-based models (SmolVLA).
    Flow matching may have different characteristics than diffusion.

    Args:
        samples: Sampled actions of shape (n_samples, action_dim) or
                trajectories of shape (n_samples, timesteps, action_dim)
        aggregation: How to aggregate ('mean', 'max', 'sum')

    Returns:
        Dictionary with flow-specific uncertainty metrics
    """
    # Flow variance computation is similar to diffusion
    result = diffusion_variance(samples, aggregation)

    # Rename for clarity
    result["flow_variance"] = result.pop("action_variance")

    # Additional flow-specific metrics could be added here
    # e.g., velocity consistency, trajectory smoothness

    return result


def iou_confidence(
    iou_predictions: Union[float, np.ndarray],
    stability_scores: Optional[Union[float, np.ndarray]] = None,
    occlusion_scores: Optional[Union[float, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute uncertainty from SAM's IoU prediction head.

    SAM predicts IoU (Intersection over Union) as a confidence measure.
    SAM2 also provides stability and occlusion scores.

    Args:
        iou_predictions: Predicted IoU score(s) [0, 1]
        stability_scores: Optional stability scores from SAM2
        occlusion_scores: Optional occlusion scores from SAM2

    Returns:
        Dictionary with:
        - iou_confidence: Mean IoU prediction as confidence
        - iou_uncertainty: 1 - IoU as uncertainty
        - combined_confidence: Weighted combination with other scores
    """
    iou_predictions = np.atleast_1d(iou_predictions)

    mean_iou = float(np.mean(iou_predictions))
    max_iou = float(np.max(iou_predictions))
    min_iou = float(np.min(iou_predictions))

    result = {
        "iou_confidence": mean_iou,
        "iou_uncertainty": 1.0 - mean_iou,
        "max_iou": max_iou,
        "min_iou": min_iou,
        "iou_std": float(np.std(iou_predictions)) if len(iou_predictions) > 1 else 0.0,
    }

    # Combine with stability and occlusion if available
    combined = mean_iou
    weights_sum = 1.0

    if stability_scores is not None:
        stability_scores = np.atleast_1d(stability_scores)
        mean_stability = float(np.mean(stability_scores))
        result["stability_score"] = mean_stability
        combined += mean_stability
        weights_sum += 1.0

    if occlusion_scores is not None:
        occlusion_scores = np.atleast_1d(occlusion_scores)
        mean_occlusion = float(np.mean(occlusion_scores))
        # Lower occlusion is better, so use (1 - occlusion)
        result["occlusion_score"] = mean_occlusion
        combined += (1.0 - mean_occlusion)
        weights_sum += 1.0

    result["combined_confidence"] = combined / weights_sum

    return result


def detection_confidence(
    objectness_scores: np.ndarray,
    class_scores: np.ndarray,
    aggregation: str = "max"
) -> Dict[str, float]:
    """
    Compute uncertainty from YOLO-style detection confidence.

    Detection confidence is typically objectness * class_probability.

    Args:
        objectness_scores: Objectness/box confidence scores
        class_scores: Class probability scores (after softmax)
        aggregation: How to aggregate across detections ('max', 'mean', 'top_k')

    Returns:
        Dictionary with:
        - detection_confidence: Aggregated confidence
        - objectness_uncertainty: From objectness scores
        - class_uncertainty: From class probability entropy
    """
    objectness_scores = np.atleast_1d(objectness_scores)
    class_scores = np.atleast_2d(class_scores)

    # Combined confidence: objectness * max_class_prob
    max_class_probs = np.max(class_scores, axis=-1)
    combined_conf = objectness_scores * max_class_probs

    # Aggregate across detections
    if aggregation == "max":
        agg_confidence = float(np.max(combined_conf)) if len(combined_conf) > 0 else 0.0
    elif aggregation == "mean":
        agg_confidence = float(np.mean(combined_conf)) if len(combined_conf) > 0 else 0.0
    elif aggregation == "top_k":
        k = min(5, len(combined_conf))
        top_k = np.sort(combined_conf)[-k:]
        agg_confidence = float(np.mean(top_k)) if len(top_k) > 0 else 0.0
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Class entropy (uncertainty in class assignment)
    class_entropies = []
    for probs in class_scores:
        h = scipy_entropy(probs, base=2)
        max_h = np.log2(len(probs)) if len(probs) > 1 else 1.0
        class_entropies.append(h / max_h if max_h > 0 else 0.0)

    mean_class_entropy = float(np.mean(class_entropies)) if class_entropies else 0.0

    return {
        "detection_confidence": agg_confidence,
        "detection_uncertainty": 1.0 - agg_confidence,
        "mean_objectness": float(np.mean(objectness_scores)),
        "mean_class_prob": float(np.mean(max_class_probs)),
        "class_entropy": mean_class_entropy,
        "n_detections": len(objectness_scores),
    }


def logit_based_uncertainty(
    logits: np.ndarray,
    method: str = "entropy"
) -> Dict[str, float]:
    """
    General logit-based uncertainty computation.

    Supports multiple methods for extracting uncertainty from raw logits.

    Args:
        logits: Raw logits of shape (n_classes,) or (seq_len, n_classes)
        method: Uncertainty method ('entropy', 'margin', 'least_confident',
               'ratio', 'variation_ratio')

    Returns:
        Dictionary with uncertainty metrics
    """
    logits = np.atleast_2d(logits)

    # Compute probabilities
    probs = softmax(logits, axis=-1)

    results = {}

    if method == "entropy" or method == "all":
        # Predictive entropy
        entropies = [scipy_entropy(p, base=2) for p in probs]
        max_entropy = np.log2(probs.shape[-1])
        norm_entropies = [e / max_entropy for e in entropies]
        results["entropy"] = float(np.mean(norm_entropies))

    if method == "margin" or method == "all":
        # Margin between top-2 predictions
        sorted_probs = np.sort(probs, axis=-1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        results["margin"] = float(np.mean(margins))
        results["margin_uncertainty"] = 1.0 - float(np.mean(margins))

    if method == "least_confident" or method == "all":
        # 1 - max probability
        max_probs = np.max(probs, axis=-1)
        results["least_confident"] = 1.0 - float(np.mean(max_probs))
        results["max_confidence"] = float(np.mean(max_probs))

    if method == "ratio" or method == "all":
        # Ratio of top-2 probabilities
        sorted_probs = np.sort(probs, axis=-1)
        ratios = sorted_probs[:, -2] / (sorted_probs[:, -1] + 1e-10)
        results["probability_ratio"] = float(np.mean(ratios))

    return results


def aggregate_uncertainties(
    uncertainty_dicts: List[Dict[str, float]],
    primary_key: str = "token_entropy"
) -> Dict[str, float]:
    """
    Aggregate uncertainty metrics across multiple predictions.

    Useful for batch processing or temporal aggregation.

    Args:
        uncertainty_dicts: List of uncertainty dictionaries
        primary_key: Key to use for primary uncertainty value

    Returns:
        Aggregated uncertainty dictionary
    """
    if not uncertainty_dicts:
        return {}

    # Collect all keys
    all_keys = set()
    for d in uncertainty_dicts:
        all_keys.update(d.keys())

    result = {}
    for key in all_keys:
        values = [d.get(key) for d in uncertainty_dicts if key in d]
        values = [v for v in values if v is not None and not isinstance(v, (list, dict))]

        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
            result[f"{key}_max"] = float(np.max(values))
            result[f"{key}_min"] = float(np.min(values))

    # Set primary uncertainty
    if f"{primary_key}_mean" in result:
        result["primary_uncertainty"] = result[f"{primary_key}_mean"]

    return result
