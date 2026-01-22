"""
Model wrappers for vision models.

Provides a unified interface for loading, running inference, and extracting
uncertainty from various vision model architectures.
"""

from .base import ModelWrapper

# Import model implementations to register them
from .detection import sam, yolo
from .self_supervised import dinov2, vjepa
from .vlm import florence, paligemma, qwen, llava, internvl
from .vla import octo, smolvla, openvla

__all__ = [
    "ModelWrapper",
]


def get_model(name: str, **kwargs) -> ModelWrapper:
    """
    Get a model wrapper by name.

    Args:
        name: Model identifier (e.g., "sam", "florence2_base")
        **kwargs: Additional arguments passed to constructor

    Returns:
        Instantiated model wrapper
    """
    return ModelWrapper.create(name, **kwargs)


def list_models() -> list:
    """List all available model names."""
    return ModelWrapper.list_models()


def list_models_by_category(category: str) -> list:
    """List models filtered by category."""
    return ModelWrapper.list_models_by_category(category)


def list_models_by_tier(tier: str) -> list:
    """List models filtered by deployment tier."""
    return ModelWrapper.list_models_by_tier(tier)
