"""
Octo model wrapper implementations.

Provides wrappers for:
- Octo-Small: Edge-deployable (27M params)
- Octo-Base: Cloud model (93M params)

Octo uses diffusion-based action prediction with variance for uncertainty.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..base import ModelWrapper
from ...uncertainty.extractors import diffusion_variance


@ModelWrapper.register("octo_small")
class OctoSmall(ModelWrapper):
    """
    Wrapper for Octo-Small robot action model.

    Octo is a generalist robot policy that uses diffusion for action
    prediction. Uncertainty is estimated by sampling multiple action
    trajectories and computing variance.

    Attributes:
        model_name: "octo_small"
        category: "vla"
        deployment_tier: "edge" (27M params)
        uncertainty_method: diffusion_variance
        license: Open
    """

    def __init__(
        self,
        model_name: str = "octo_small",
        deployment_tier: str = "edge",
        n_action_samples: int = 10,
        **kwargs
    ):
        super().__init__(model_name, deployment_tier, **kwargs)
        self.n_action_samples = n_action_samples

    def load(self) -> None:
        """Load Octo-Small model."""
        # TODO: Implement Octo loading
        # from octo.model.octo_model import OctoModel
        # self._model = OctoModel.load_pretrained("hf-octo/octo-small-1.5")
        raise NotImplementedError(
            "Octo-Small wrapper not yet implemented. "
            "Requires octo library installation."
        )

    def predict(
        self,
        image: np.ndarray,
        task: Optional[str] = None,
        proprio_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Predict robot action.

        Args:
            image: Input image observation (H, W, C)
            task: Optional task description
            proprio_state: Optional proprioceptive state

        Returns:
            Tuple of:
                - action: Predicted action vector
                - confidence: Based on action variance (lower = more confident)
        """
        # TODO: Implement action prediction
        raise NotImplementedError("Octo-Small prediction not implemented")

    def extract_uncertainty(
        self,
        image: np.ndarray,
        task: Optional[str] = None,
        proprio_state: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Extract uncertainty from diffusion sampling.

        Samples multiple action trajectories and computes variance.

        Args:
            image: Input image
            task: Optional task description
            proprio_state: Optional proprioceptive state
            n_samples: Number of samples (default: self.n_action_samples)

        Returns:
            Dictionary with diffusion variance metrics
        """
        # TODO: Implement using diffusion_variance
        # Sample n_samples trajectories, compute variance
        raise NotImplementedError("Octo-Small uncertainty extraction not implemented")

    def sample_actions(
        self,
        image: np.ndarray,
        n_samples: int = 10,
        task: Optional[str] = None
    ) -> np.ndarray:
        """
        Sample multiple action trajectories.

        Args:
            image: Input image
            n_samples: Number of trajectories to sample
            task: Optional task description

        Returns:
            Sampled actions of shape (n_samples, action_dim) or
            (n_samples, horizon, action_dim)
        """
        # TODO: Implement action sampling
        raise NotImplementedError("Octo-Small action sampling not implemented")


@ModelWrapper.register("octo_base")
class OctoBase(OctoSmall):
    """
    Wrapper for Octo-Base robot action model.

    Larger variant with 93M parameters for cloud deployment.

    Attributes:
        model_name: "octo_base"
        category: "vla"
        deployment_tier: "cloud" (93M params)
        license: Open
    """

    def __init__(
        self,
        model_name: str = "octo_base",
        deployment_tier: str = "cloud",
        **kwargs
    ):
        ModelWrapper.__init__(self, model_name, deployment_tier, **kwargs)
        self.n_action_samples = kwargs.get("n_action_samples", 10)

    def load(self) -> None:
        """Load Octo-Base model."""
        # TODO: Implement loading
        raise NotImplementedError("Octo-Base wrapper not yet implemented")
