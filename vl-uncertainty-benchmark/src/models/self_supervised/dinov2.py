"""
DINOv2 model wrapper implementations.

Provides wrappers for:
- DINOv2-B: Edge-deployable (86M params)
- DINOv2-g: Cloud model (1.1B params)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..base import ModelWrapper
from ...uncertainty.extractors import embedding_distance


@ModelWrapper.register("dinov2_b")
class DINOv2B(ModelWrapper):
    """
    Wrapper for DINOv2 Base model.

    DINOv2 provides powerful visual representations through self-supervised
    learning. Uncertainty is computed as embedding distance to class centroids.

    Attributes:
        model_name: "dinov2_b"
        category: "self_supervised"
        deployment_tier: "edge" (86M params)
        uncertainty_method: embedding_distance
        license: Apache 2.0
    """

    def __init__(
        self,
        model_name: str = "dinov2_b",
        deployment_tier: str = "edge",
        **kwargs
    ):
        super().__init__(model_name, deployment_tier, **kwargs)
        self._centroids = None  # Class centroids for uncertainty

    def load(self) -> None:
        """Load DINOv2-B model."""
        # TODO: Implement DINOv2 loading
        # import torch
        # self._model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # or use transformers:
        # from transformers import AutoModel
        # self._model = AutoModel.from_pretrained('facebook/dinov2-base')
        raise NotImplementedError(
            "DINOv2-B wrapper not yet implemented. "
            "Install torch and implement loading."
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extract DINOv2 embeddings.

        Args:
            image: Input image (H, W, C)

        Returns:
            Tuple of:
                - embedding: Feature embedding vector
                - confidence: Distance-based confidence to nearest centroid
        """
        # TODO: Implement embedding extraction
        raise NotImplementedError("DINOv2-B prediction not implemented")

    def extract_uncertainty(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract uncertainty from embedding distance.

        Requires centroids to be set via set_centroids().
        """
        # TODO: Implement using embedding_distance()
        raise NotImplementedError("DINOv2-B uncertainty extraction not implemented")

    def set_centroids(self, centroids: np.ndarray) -> None:
        """
        Set class centroids for uncertainty computation.

        Args:
            centroids: Array of shape (n_classes, embed_dim)
        """
        self._centroids = np.asarray(centroids)

    def compute_centroids(
        self,
        images: List[np.ndarray],
        labels: List[int]
    ) -> np.ndarray:
        """
        Compute class centroids from labeled images.

        Args:
            images: List of images
            labels: Class labels for each image

        Returns:
            Centroids array of shape (n_classes, embed_dim)
        """
        # TODO: Implement centroid computation
        raise NotImplementedError("Centroid computation not implemented")


@ModelWrapper.register("dinov2_g")
class DINOv2G(DINOv2B):
    """
    Wrapper for DINOv2 Giant model.

    Larger variant with 1.1B parameters for cloud deployment.

    Attributes:
        model_name: "dinov2_g"
        category: "self_supervised"
        deployment_tier: "cloud" (1.1B params)
        license: Apache 2.0
    """

    def __init__(
        self,
        model_name: str = "dinov2_g",
        deployment_tier: str = "cloud",
        **kwargs
    ):
        ModelWrapper.__init__(self, model_name, deployment_tier, **kwargs)
        self._centroids = None

    def load(self) -> None:
        """Load DINOv2-g model."""
        # TODO: Implement loading
        # self._model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        raise NotImplementedError("DINOv2-g wrapper not yet implemented")
