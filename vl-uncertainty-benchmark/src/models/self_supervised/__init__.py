"""Self-supervised model wrappers."""

from .dinov2 import DINOv2B, DINOv2G
from .vjepa import VJEPA2

__all__ = [
    "DINOv2B",
    "DINOv2G",
    "VJEPA2",
]
