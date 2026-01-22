"""
Abstract base class for model wrappers.

All model wrappers inherit from ModelWrapper and implement the required
abstract methods for prediction, uncertainty extraction, and calibration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import yaml


class ModelWrapper(ABC):
    """
    Abstract base class for vision model wrappers.

    Provides a unified interface for:
    - Model loading and inference
    - Uncertainty extraction
    - Calibration (temperature scaling, etc.)
    - Edge deployment optimization

    Subclasses must implement: load(), predict(), extract_uncertainty()
    """

    # Class-level model registry
    _registry: Dict[str, type] = {}

    def __init__(
        self,
        model_name: str,
        deployment_tier: str = "cloud",
        device: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the model wrapper.

        Args:
            model_name: Identifier for this model (e.g., "sam", "florence2_base")
            deployment_tier: One of 'edge', 'edge_plus', 'cloud'
                - edge: Jetson Nano 8GB equivalent (~4GB VRAM)
                - edge_plus: Jetson Orin 64GB equivalent
                - cloud: Full-precision on high-end GPUs (A100/H100)
            device: PyTorch device string (e.g., "cuda:0", "cpu").
                   Auto-detected if None.
            config_path: Path to models.yaml config file. Uses default if None.
        """
        self.model_name = model_name
        self.deployment_tier = deployment_tier
        self._device = device
        self._model = None
        self._processor = None
        self._is_loaded = False

        # Calibration state
        self._temperature = 1.0
        self._is_calibrated = False
        self._calibration_method = None

        # Load model config
        self._config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load model configuration from YAML file."""
        if config_path is None:
            # Default config path relative to this file
            config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"

        if Path(config_path).exists():
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)

            # Find this model's config
            model_key = self.model_name.lower().replace("-", "_")
            if "models" in full_config and model_key in full_config["models"]:
                return full_config["models"][model_key]

        # Return minimal default config
        return {
            "name": self.model_name,
            "category": "unknown",
            "deployment_tier": self.deployment_tier,
            "params": 0,
            "uncertainty_method": "unknown",
            "license": "unknown"
        }

    @property
    def device(self) -> str:
        """Get the device string, auto-detecting if necessary."""
        if self._device is None:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """
        Lazy load the model into memory.

        Should set self._model and self._is_loaded = True.
        May also set self._processor for models that need preprocessing.

        This method should handle:
        - Downloading model weights if necessary
        - Loading to the correct device
        - Applying quantization for edge deployment if applicable
        """
        pass

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded, loading it if necessary."""
        if not self._is_loaded:
            self.load()

    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[Any, float]:
        """
        Run inference on an image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR or RGB format.
                  Should be uint8 [0, 255] or float32 [0, 1].

        Returns:
            Tuple of:
                - prediction: Model output (format depends on model type)
                    - Detection models: List of (bbox, class, score)
                    - VLMs: Generated text string
                    - VLAs: Action vector or trajectory
                    - Self-supervised: Embedding vector
                - raw_confidence: Single float representing model confidence [0, 1]
        """
        pass

    @abstractmethod
    def extract_uncertainty(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract uncertainty metrics from model inference.

        Different model types use different uncertainty estimation methods:
        - VLMs/VLAs: Token entropy, sequence entropy
        - Self-supervised: Embedding distance to centroids
        - Detection: IoU prediction, objectness scores
        - Diffusion-based: Sampling variance

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Dictionary of uncertainty metrics, e.g.:
            {
                "token_entropy": 0.45,
                "sequence_entropy": 0.32,
                "max_logit_diff": 2.1,
                ...
            }
        """
        pass

    def predict_with_uncertainty(
        self,
        image: np.ndarray
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Combined prediction and uncertainty extraction.

        More efficient than calling predict() and extract_uncertainty()
        separately for models that can compute both in one forward pass.

        Args:
            image: Input image

        Returns:
            Tuple of (prediction, confidence, uncertainty_dict)
        """
        # Default implementation calls both separately
        # Subclasses can override for efficiency
        prediction, confidence = self.predict(image)
        uncertainty = self.extract_uncertainty(image)
        return prediction, confidence, uncertainty

    def calibrate(
        self,
        val_images: List[np.ndarray],
        val_labels: List[Any],
        method: str = "temperature_scaling"
    ) -> float:
        """
        Fit calibration parameters on a validation set.

        Args:
            val_images: List of validation images
            val_labels: List of ground truth labels
            method: Calibration method ('temperature_scaling' or 'platt_scaling')

        Returns:
            Optimal temperature or calibration loss
        """
        from ..uncertainty.calibration import (
            compute_optimal_temperature,
            fit_platt_scaling
        )

        self.ensure_loaded()

        # Collect logits/confidences on validation set
        logits_list = []
        labels_list = []

        for img, label in zip(val_images, val_labels):
            _, confidence = self.predict(img)
            logits_list.append(confidence)
            labels_list.append(label)

        logits = np.array(logits_list)
        labels = np.array(labels_list)

        if method == "temperature_scaling":
            self._temperature = compute_optimal_temperature(logits, labels)
            self._calibration_method = "temperature_scaling"
        elif method == "platt_scaling":
            self._platt_params = fit_platt_scaling(logits, labels)
            self._calibration_method = "platt_scaling"
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self._is_calibrated = True
        return self._temperature if method == "temperature_scaling" else 0.0

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """
        Apply calibration to a raw confidence score.

        Args:
            raw_confidence: Uncalibrated confidence [0, 1]

        Returns:
            Calibrated confidence [0, 1]
        """
        if not self._is_calibrated:
            return raw_confidence

        if self._calibration_method == "temperature_scaling":
            # Temperature scaling: softmax(logits / T)
            # For single confidence, approximate as sigmoid adjustment
            import numpy as np
            logit = np.log(raw_confidence / (1 - raw_confidence + 1e-10) + 1e-10)
            scaled_logit = logit / self._temperature
            return 1 / (1 + np.exp(-scaled_logit))

        elif self._calibration_method == "platt_scaling":
            a, b = self._platt_params
            logit = np.log(raw_confidence / (1 - raw_confidence + 1e-10) + 1e-10)
            return 1 / (1 + np.exp(-(a * logit + b)))

        return raw_confidence

    def to_edge(self) -> None:
        """
        Optimize model for edge deployment.

        Applies optimizations like:
        - INT8/INT4 quantization
        - TensorRT conversion
        - ONNX export
        - Model pruning

        The specific optimizations depend on the model and target platform.
        """
        # Default implementation - subclasses should override
        raise NotImplementedError(
            f"Edge optimization not implemented for {self.__class__.__name__}"
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Return model metadata.

        Returns:
            Dictionary containing:
            - name: Model identifier
            - full_name: Human-readable name
            - params: Number of parameters
            - license: License type
            - category: Model category (detection, vlm, vla, self_supervised)
            - deployment_tier: Current deployment tier
            - uncertainty_method: Method used for uncertainty estimation
            - is_loaded: Whether model is currently loaded
            - is_calibrated: Whether calibration has been applied
        """
        return {
            "name": self.model_name,
            "full_name": self._config.get("full_name", self.model_name),
            "params": self._config.get("params", 0),
            "license": self._config.get("license", "unknown"),
            "category": self._config.get("category", "unknown"),
            "deployment_tier": self.deployment_tier,
            "uncertainty_method": self._config.get("uncertainty_method", "unknown"),
            "model_id": self._config.get("model_id", ""),
            "is_loaded": self._is_loaded,
            "is_calibrated": self._is_calibrated,
            "calibration_method": self._calibration_method,
            "temperature": self._temperature if self._is_calibrated else None,
            "device": self._device,
        }

    def unload(self) -> None:
        """
        Unload model from memory to free resources.

        Useful when benchmarking multiple models sequentially.
        """
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._is_loaded = False

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        calib = ", calibrated" if self._is_calibrated else ""
        return f"{self.__class__.__name__}(name={self.model_name}, tier={self.deployment_tier}, {status}{calib})"

    # =========================================================================
    # Registry methods for model discovery
    # =========================================================================

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model wrapper class.

        Usage:
            @ModelWrapper.register("sam")
            class SAMWrapper(ModelWrapper):
                ...
        """
        def decorator(model_cls):
            cls._registry[name.lower()] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get_model_class(cls, name: str) -> type:
        """Get a registered model wrapper class by name."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(
                f"Unknown model: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]

    @classmethod
    def create(
        cls,
        name: str,
        deployment_tier: str = "cloud",
        **kwargs
    ) -> "ModelWrapper":
        """
        Factory method to create a model wrapper by name.

        Args:
            name: Model identifier (e.g., "sam", "florence2_base")
            deployment_tier: Deployment tier
            **kwargs: Additional arguments passed to constructor

        Returns:
            Instantiated model wrapper
        """
        model_cls = cls.get_model_class(name)
        return model_cls(model_name=name, deployment_tier=deployment_tier, **kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._registry.keys())

    @classmethod
    def list_models_by_category(cls, category: str) -> List[str]:
        """List models filtered by category."""
        result = []
        for name, model_cls in cls._registry.items():
            # Instantiate temporarily to check category
            try:
                temp = model_cls(model_name=name)
                if temp._config.get("category") == category:
                    result.append(name)
            except Exception:
                pass
        return result

    @classmethod
    def list_models_by_tier(cls, tier: str) -> List[str]:
        """List models filtered by deployment tier."""
        result = []
        for name, model_cls in cls._registry.items():
            try:
                temp = model_cls(model_name=name)
                if temp._config.get("deployment_tier") == tier:
                    result.append(name)
            except Exception:
                pass
        return result
