"""
OpenVLA model wrapper implementation.

OpenVLA is a 7B vision-language-action model that uses token entropy
for uncertainty estimation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..base import ModelWrapper
from ...uncertainty.extractors import token_entropy


@ModelWrapper.register("openvla")
class OpenVLA(ModelWrapper):
    """
    Wrapper for OpenVLA 7B robot action model.

    OpenVLA is a large vision-language-action model that predicts
    robot actions as discrete tokens. Uncertainty is estimated from
    token entropy during action generation.

    Attributes:
        model_name: "openvla"
        category: "vla"
        deployment_tier: "cloud" (7B params)
        uncertainty_method: token_entropy
        license: Llama license
    """

    def __init__(
        self,
        model_name: str = "openvla",
        deployment_tier: str = "cloud",
        **kwargs
    ):
        super().__init__(model_name, deployment_tier, **kwargs)
        self._last_logits = None

    def load(self) -> None:
        """Load OpenVLA model."""
        # TODO: Implement OpenVLA loading
        # from transformers import AutoModelForVision2Seq, AutoProcessor
        # self._model = AutoModelForVision2Seq.from_pretrained(
        #     "openvla/openvla-7b",
        #     torch_dtype=torch.bfloat16
        # )
        raise NotImplementedError(
            "OpenVLA wrapper not yet implemented. "
            "Requires openvla/openvla-7b model."
        )

    def predict(
        self,
        image: np.ndarray,
        instruction: str = "",
        proprio_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Predict robot action.

        Args:
            image: Input image observation (H, W, C)
            instruction: Task instruction
            proprio_state: Optional proprioceptive state

        Returns:
            Tuple of:
                - action: Predicted action vector (decoded from tokens)
                - confidence: Based on token probabilities
        """
        # TODO: Implement action prediction
        raise NotImplementedError("OpenVLA prediction not implemented")

    def extract_uncertainty(
        self,
        image: np.ndarray,
        instruction: str = ""
    ) -> Dict[str, float]:
        """
        Extract uncertainty from action token entropy.

        Args:
            image: Input image
            instruction: Task instruction

        Returns:
            Dictionary with token entropy metrics
        """
        # TODO: Implement using token_entropy
        raise NotImplementedError("OpenVLA uncertainty extraction not implemented")

    def decode_actions(self, tokens: np.ndarray) -> np.ndarray:
        """
        Decode action tokens to continuous action space.

        Args:
            tokens: Predicted action tokens

        Returns:
            Continuous action vector
        """
        # TODO: Implement token decoding
        raise NotImplementedError("OpenVLA action decoding not implemented")
