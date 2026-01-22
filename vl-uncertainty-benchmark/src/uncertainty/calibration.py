"""
Calibration methods for improving model confidence estimates.

Provides temperature scaling, Platt scaling, and utilities for
computing optimal calibration parameters.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.optimize import minimize
from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression


def compute_optimal_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    init_temp: float = 1.5,
    max_iter: int = 100
) -> float:
    """
    Find optimal temperature for temperature scaling.

    Temperature scaling divides logits by a scalar T before softmax,
    which adjusts confidence without changing predictions.

    Args:
        logits: Model logits of shape (n_samples,) for binary or
               (n_samples, n_classes) for multiclass
        labels: Ground truth labels of shape (n_samples,)
        init_temp: Initial temperature for optimization
        max_iter: Maximum optimization iterations

    Returns:
        Optimal temperature value
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # Handle binary case (single logit per sample)
    if logits.ndim == 1:
        # Convert to binary classification format
        logits = logits.reshape(-1, 1)

    n_samples = logits.shape[0]

    def nll_loss(temperature: float) -> float:
        """Negative log-likelihood loss for temperature scaling."""
        t = max(temperature, 1e-6)  # Prevent division by zero

        if logits.shape[1] == 1:
            # Binary classification
            probs = expit(logits.flatten() / t)
            # Clamp for numerical stability
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            loss = -np.mean(
                labels * np.log(probs) +
                (1 - labels) * np.log(1 - probs)
            )
        else:
            # Multiclass classification
            scaled_logits = logits / t
            probs = softmax(scaled_logits, axis=1)
            # Clamp for numerical stability
            probs = np.clip(probs, 1e-10, 1.0)
            # Get probability of correct class
            correct_probs = probs[np.arange(n_samples), labels.astype(int)]
            loss = -np.mean(np.log(correct_probs))

        return loss

    # Optimize temperature
    result = minimize(
        nll_loss,
        x0=init_temp,
        method='L-BFGS-B',
        bounds=[(0.01, 10.0)],
        options={'maxiter': max_iter}
    )

    return float(result.x[0])


def fit_platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Fit Platt scaling (logistic regression) for calibration.

    Platt scaling fits sigmoid(a * logit + b) to the data,
    allowing for per-class calibration adjustment.

    Args:
        logits: Model logits of shape (n_samples,) or (n_samples, n_classes)
        labels: Ground truth labels

    Returns:
        Tuple of (a, b) parameters for sigmoid(a * logit + b)
    """
    logits = np.asarray(logits).flatten()
    labels = np.asarray(labels).flatten()

    # Fit logistic regression
    # sklearn expects 2D input
    X = logits.reshape(-1, 1)
    y = labels

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=True
    )
    model.fit(X, y)

    # Extract parameters
    a = float(model.coef_[0, 0])
    b = float(model.intercept_[0])

    return (a, b)


class TemperatureScaling:
    """
    Temperature scaling calibration method.

    Learns a single temperature parameter T that scales logits
    before softmax: p = softmax(logits / T)
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaling.

        Args:
            temperature: Initial temperature value
        """
        self.temperature = temperature
        self._is_fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        **kwargs
    ) -> "TemperatureScaling":
        """
        Fit temperature on validation data.

        Args:
            logits: Validation logits
            labels: Validation labels
            **kwargs: Additional arguments for optimization

        Returns:
            Self
        """
        self.temperature = compute_optimal_temperature(logits, labels, **kwargs)
        self._is_fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Input logits

        Returns:
            Calibrated probabilities
        """
        scaled = logits / self.temperature

        if logits.ndim == 1 or logits.shape[-1] == 1:
            # Binary classification
            return expit(scaled.flatten())
        else:
            # Multiclass
            return softmax(scaled, axis=-1)

    def calibrate_confidence(self, confidence: float) -> float:
        """
        Calibrate a single confidence value.

        Args:
            confidence: Raw confidence [0, 1]

        Returns:
            Calibrated confidence [0, 1]
        """
        # Convert confidence to logit, scale, convert back
        eps = 1e-10
        confidence = np.clip(confidence, eps, 1 - eps)
        logit = np.log(confidence / (1 - confidence))
        scaled_logit = logit / self.temperature
        return float(expit(scaled_logit))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class PlattScaling:
    """
    Platt scaling calibration method.

    Fits a logistic regression model: p = sigmoid(a * logit + b)
    Allows for affine transformation of logits.
    """

    def __init__(self):
        """Initialize Platt scaling."""
        self.a = 1.0
        self.b = 0.0
        self._is_fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> "PlattScaling":
        """
        Fit Platt scaling on validation data.

        Args:
            logits: Validation logits
            labels: Validation labels

        Returns:
            Self
        """
        self.a, self.b = fit_platt_scaling(logits, labels)
        self._is_fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to logits.

        Args:
            logits: Input logits

        Returns:
            Calibrated probabilities
        """
        scaled = self.a * logits + self.b
        return expit(scaled)

    def calibrate_confidence(self, confidence: float) -> float:
        """
        Calibrate a single confidence value.

        Args:
            confidence: Raw confidence [0, 1]

        Returns:
            Calibrated confidence [0, 1]
        """
        eps = 1e-10
        confidence = np.clip(confidence, eps, 1 - eps)
        logit = np.log(confidence / (1 - confidence))
        scaled_logit = self.a * logit + self.b
        return float(expit(scaled_logit))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class FocalLossCalibration:
    """
    Focal loss inspired calibration.

    Uses focal loss weighting to improve calibration on hard examples.
    Not a direct calibration method but can be used during training.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Initialize focal loss calibration.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class balancing weight
        """
        self.gamma = gamma
        self.alpha = alpha

    def compute_weights(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute focal loss weights for samples.

        Args:
            probabilities: Predicted probabilities
            labels: Ground truth labels

        Returns:
            Sample weights
        """
        probabilities = np.asarray(probabilities).flatten()
        labels = np.asarray(labels).flatten()

        # Get probability of correct class
        p_t = np.where(labels == 1, probabilities, 1 - probabilities)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_weight = np.where(labels == 1, self.alpha, 1 - self.alpha)

        return alpha_weight * focal_weight

    def focal_loss(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute focal loss.

        Args:
            probabilities: Predicted probabilities
            labels: Ground truth labels

        Returns:
            Focal loss value
        """
        eps = 1e-10
        probabilities = np.clip(probabilities, eps, 1 - eps)
        labels = np.asarray(labels).flatten()

        p_t = np.where(labels == 1, probabilities, 1 - probabilities)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = np.where(labels == 1, self.alpha, 1 - self.alpha)

        ce_loss = -np.log(p_t)
        focal_loss = alpha_weight * focal_weight * ce_loss

        return float(np.mean(focal_loss))


def calibrate_predictions(
    confidences: np.ndarray,
    labels: np.ndarray,
    method: str = "temperature"
) -> Tuple[np.ndarray, Dict]:
    """
    Calibrate predictions using the specified method.

    Convenience function that fits and applies calibration in one step.

    Args:
        confidences: Raw confidence scores [0, 1]
        labels: Ground truth binary labels
        method: Calibration method ('temperature', 'platt')

    Returns:
        Tuple of (calibrated_confidences, calibration_info)
    """
    # Convert confidences to logits
    eps = 1e-10
    confidences = np.clip(confidences, eps, 1 - eps)
    logits = np.log(confidences / (1 - confidences))

    if method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(logits, labels)
        calibrated = calibrator.transform(logits)
        info = {"method": "temperature", "temperature": calibrator.temperature}

    elif method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(logits, labels)
        calibrated = calibrator.transform(logits)
        info = {"method": "platt", "a": calibrator.a, "b": calibrator.b}

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return calibrated, info


def cross_validate_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    method: str = "temperature",
    n_folds: int = 5
) -> Dict:
    """
    Cross-validate calibration to estimate reliability.

    Args:
        logits: Model logits
        labels: Ground truth labels
        method: Calibration method
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import KFold

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    temperatures = []
    eces_before = []
    eces_after = []

    for train_idx, val_idx in kfold.split(logits):
        train_logits, val_logits = logits[train_idx], logits[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # Fit on train
        if method == "temperature":
            temp = compute_optimal_temperature(train_logits, train_labels)
            temperatures.append(temp)

            # Compute ECE before and after on validation
            if train_logits.ndim == 1:
                probs_before = expit(val_logits)
                probs_after = expit(val_logits / temp)
            else:
                probs_before = softmax(val_logits, axis=1)
                probs_after = softmax(val_logits / temp, axis=1)

    return {
        "temperatures": temperatures,
        "mean_temperature": float(np.mean(temperatures)),
        "std_temperature": float(np.std(temperatures)),
        "n_folds": n_folds,
    }
