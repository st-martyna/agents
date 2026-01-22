"""
InternVL2.5 model wrapper implementation.

InternVL2.5-78B achieves highest benchmark scores among open VLMs.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..base import ModelWrapper
from ...uncertainty.extractors import token_entropy


@ModelWrapper.register("internvl25_78b")
class InternVL25_78B(ModelWrapper):
    """
    Wrapper for InternVL2.5-78B model.

    InternVL2.5 is a state-of-the-art vision-language model with
    highest benchmark scores among open-source VLMs.

    Attributes:
        model_name: "internvl25_78b"
        category: "vlm"
        deployment_tier: "cloud" (78B params)
        uncertainty_method: token_entropy
        license: Apache 2.0
    """

    def __init__(
        self,
        model_name: str = "internvl25_78b",
        deployment_tier: str = "cloud",
        **kwargs
    ):
        super().__init__(model_name, deployment_tier, **kwargs)
        self._last_logits = None

    def load(self) -> None:
        """Load InternVL2.5-78B model."""
        # TODO: Implement InternVL loading
        # from transformers import AutoModel, AutoTokenizer
        # self._model = AutoModel.from_pretrained(
        #     "OpenGVLab/InternVL2_5-78B",
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True
        # )
        raise NotImplementedError(
            "InternVL2.5-78B wrapper not yet implemented. "
            "Requires OpenGVLab/InternVL2_5-78B model access."
        )

    def predict(
        self,
        image: np.ndarray,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        """
        Run InternVL2.5 inference.

        Args:
            image: Input image (H, W, C)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response, confidence)
        """
        # TODO: Implement prediction
        raise NotImplementedError("InternVL2.5-78B prediction not implemented")

    def extract_uncertainty(self, image: np.ndarray) -> Dict[str, float]:
        """Extract uncertainty from token logits."""
        # TODO: Implement using token_entropy
        raise NotImplementedError("InternVL2.5-78B uncertainty extraction not implemented")
