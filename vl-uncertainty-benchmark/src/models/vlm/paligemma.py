"""
PaliGemma 2 model wrapper implementations.

Provides wrappers for:
- PaliGemma 2-3B: Edge-deployable
- PaliGemma 2-28B: Cloud model
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..base import ModelWrapper
from ...uncertainty.extractors import token_entropy


@ModelWrapper.register("paligemma2_3b")
class PaliGemma2_3B(ModelWrapper):
    """
    Wrapper for PaliGemma 2 3B model.

    PaliGemma combines PaLI's vision encoder with Gemma's language model.
    Uncertainty is extracted from location token confidence for detection tasks.

    Attributes:
        model_name: "paligemma2_3b"
        category: "vlm"
        deployment_tier: "edge" (3B params)
        uncertainty_method: location_confidence
        license: Gemma license
    """

    def __init__(
        self,
        model_name: str = "paligemma2_3b",
        deployment_tier: str = "edge",
        **kwargs
    ):
        super().__init__(model_name, deployment_tier, **kwargs)
        self._last_logits = None

    def load(self) -> None:
        """Load PaliGemma 2-3B model."""
        # TODO: Implement PaliGemma loading
        # from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        # self._model = PaliGemmaForConditionalGeneration.from_pretrained(
        #     "google/paligemma2-3b-pt-224"
        # )
        raise NotImplementedError(
            "PaliGemma 2-3B wrapper not yet implemented. "
            "Requires google/paligemma2-3b-pt-224 model access."
        )

    def predict(
        self,
        image: np.ndarray,
        prompt: str = "detect",
        max_new_tokens: int = 256
    ) -> Tuple[str, float]:
        """
        Run PaliGemma inference.

        Args:
            image: Input image (H, W, C)
            prompt: Task prompt (e.g., "detect", "caption", "answer: <question>")
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response, confidence)
        """
        # TODO: Implement prediction
        raise NotImplementedError("PaliGemma 2-3B prediction not implemented")

    def extract_uncertainty(self, image: np.ndarray) -> Dict[str, float]:
        """Extract uncertainty from token logits."""
        # TODO: Implement using token_entropy
        raise NotImplementedError("PaliGemma 2-3B uncertainty extraction not implemented")


@ModelWrapper.register("paligemma2_28b")
class PaliGemma2_28B(PaliGemma2_3B):
    """
    Wrapper for PaliGemma 2 28B model.

    Larger variant for cloud deployment.

    Attributes:
        model_name: "paligemma2_28b"
        category: "vlm"
        deployment_tier: "cloud" (28B params)
        license: Gemma license
    """

    def __init__(
        self,
        model_name: str = "paligemma2_28b",
        deployment_tier: str = "cloud",
        **kwargs
    ):
        ModelWrapper.__init__(self, model_name, deployment_tier, **kwargs)
        self._last_logits = None

    def load(self) -> None:
        """Load PaliGemma 2-28B model."""
        # TODO: Implement loading
        raise NotImplementedError("PaliGemma 2-28B wrapper not yet implemented")
