"""
SAM (Segment Anything Model) wrapper implementations.

Provides wrappers for:
- SAM (base): Edge-deployable, uses IoU prediction as confidence
- SAM2-Large: Cloud model with IoU + occlusion scores

Reference implementation for detection model wrappers.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import warnings

from ..base import ModelWrapper
from ...uncertainty.extractors import iou_confidence


@ModelWrapper.register("sam")
class SAMWrapper(ModelWrapper):
    """
    Wrapper for Segment Anything Model (SAM).

    SAM provides a natural confidence measure through its IoU prediction head,
    which estimates the intersection-over-union of predicted masks.

    Attributes:
        model_name: "sam"
        category: "detection"
        deployment_tier: "edge" (94M params)
        uncertainty_method: IoU prediction
        license: Apache 2.0
    """

    def __init__(
        self,
        model_name: str = "sam",
        deployment_tier: str = "edge",
        model_type: str = "vit_b",
        **kwargs
    ):
        """
        Initialize SAM wrapper.

        Args:
            model_name: Model identifier
            deployment_tier: Deployment tier ('edge' or 'cloud')
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            **kwargs: Additional arguments for base class
        """
        super().__init__(model_name, deployment_tier, **kwargs)
        self.model_type = model_type
        self._predictor = None

    def load(self) -> None:
        """Load SAM model and predictor."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not installed. "
                "Install with: pip install segment-anything"
            )

        import torch

        # Get model checkpoint
        model_id = self._config.get("model_id", "facebook/sam-vit-base")

        # Try to load from HuggingFace hub or local checkpoint
        try:
            from transformers import SamModel, SamProcessor

            self._model = SamModel.from_pretrained(model_id)
            self._processor = SamProcessor.from_pretrained(model_id)

            # Move to device
            self._model.to(self.device)
            self._model.eval()

            self._use_hf = True

        except Exception as e:
            warnings.warn(
                f"Could not load from HuggingFace ({e}). "
                "Falling back to segment-anything library."
            )

            # Fallback to segment-anything library
            # This requires manual checkpoint download
            checkpoint_path = self._config.get("checkpoint")
            if checkpoint_path is None:
                raise ValueError(
                    "No checkpoint path specified and HuggingFace loading failed. "
                    "Please specify 'checkpoint' in config or install transformers."
                )

            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            sam.to(self.device)
            sam.eval()

            self._model = sam
            self._predictor = SamPredictor(sam)
            self._use_hf = False

        self._is_loaded = True

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is in correct format (RGB, uint8)."""
        # Convert BGR to RGB if needed (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

    def predict(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Tuple[Any, float]:
        """
        Run SAM inference.

        Args:
            image: Input image (H, W, C)
            point_coords: Optional point prompts of shape (N, 2)
            point_labels: Optional point labels (1 = foreground, 0 = background)
            box: Optional box prompt of shape (4,) [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks

        Returns:
            Tuple of:
                - masks: Predicted masks array (N, H, W)
                - confidence: IoU prediction score [0, 1]
        """
        self.ensure_loaded()

        image = self._preprocess_image(image)

        if self._use_hf:
            return self._predict_hf(
                image, point_coords, point_labels, box, multimask_output
            )
        else:
            return self._predict_sam(
                image, point_coords, point_labels, box, multimask_output
            )

    def _predict_hf(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        multimask_output: bool
    ) -> Tuple[Any, float]:
        """Prediction using HuggingFace transformers."""
        import torch

        # Prepare inputs
        inputs = self._processor(
            image,
            input_points=[point_coords.tolist()] if point_coords is not None else None,
            input_labels=[point_labels.tolist()] if point_labels is not None else None,
            input_boxes=[[box.tolist()]] if box is not None else None,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=multimask_output)

        # Extract masks and IoU predictions
        masks = outputs.pred_masks.squeeze().cpu().numpy()
        iou_scores = outputs.iou_scores.squeeze().cpu().numpy()

        # Get best mask based on IoU
        if masks.ndim == 3:  # Multiple masks
            best_idx = np.argmax(iou_scores)
            best_mask = masks[best_idx]
            best_iou = float(iou_scores[best_idx])
        else:
            best_mask = masks
            best_iou = float(iou_scores)

        return best_mask, best_iou

    def _predict_sam(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        multimask_output: bool
    ) -> Tuple[Any, float]:
        """Prediction using segment-anything library."""
        # Set image
        self._predictor.set_image(image)

        # Run prediction
        masks, iou_scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )

        # Get best mask
        if masks.ndim == 3:
            best_idx = np.argmax(iou_scores)
            best_mask = masks[best_idx]
            best_iou = float(iou_scores[best_idx])
        else:
            best_mask = masks
            best_iou = float(iou_scores)

        return best_mask, best_iou

    def extract_uncertainty(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract uncertainty from SAM's IoU prediction.

        Lower IoU prediction = higher uncertainty about segmentation quality.

        Args:
            image: Input image
            point_coords: Optional point prompts
            point_labels: Optional point labels
            box: Optional box prompt

        Returns:
            Dictionary with uncertainty metrics
        """
        self.ensure_loaded()

        image = self._preprocess_image(image)

        # Get all masks and their IoU scores
        if self._use_hf:
            import torch
            inputs = self._processor(
                image,
                input_points=[point_coords.tolist()] if point_coords is not None else None,
                input_labels=[point_labels.tolist()] if point_labels is not None else None,
                input_boxes=[[box.tolist()]] if box is not None else None,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, multimask_output=True)

            iou_scores = outputs.iou_scores.squeeze().cpu().numpy()
        else:
            self._predictor.set_image(image)
            _, iou_scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )

        # Use iou_confidence extractor
        return iou_confidence(iou_scores)

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Combined prediction and uncertainty extraction.

        More efficient than separate calls as it reuses the forward pass.
        """
        self.ensure_loaded()

        image = self._preprocess_image(image)

        if self._use_hf:
            import torch
            inputs = self._processor(
                image,
                input_points=[point_coords.tolist()] if point_coords is not None else None,
                input_labels=[point_labels.tolist()] if point_labels is not None else None,
                input_boxes=[[box.tolist()]] if box is not None else None,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, multimask_output=True)

            masks = outputs.pred_masks.squeeze().cpu().numpy()
            iou_scores = outputs.iou_scores.squeeze().cpu().numpy()
        else:
            self._predictor.set_image(image)
            masks, iou_scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )

        # Get best mask
        best_idx = np.argmax(iou_scores)
        best_mask = masks[best_idx] if masks.ndim == 3 else masks
        best_iou = float(iou_scores[best_idx] if iou_scores.ndim > 0 else iou_scores)

        # Compute uncertainty from all IoU scores
        uncertainty = iou_confidence(iou_scores)

        return best_mask, best_iou, uncertainty

    def to_edge(self) -> None:
        """
        Optimize SAM for edge deployment.

        Applies:
        - FP16 conversion
        - TensorRT optimization (if available)
        """
        import torch

        if not self._is_loaded:
            raise RuntimeError("Model must be loaded before edge optimization")

        # Convert to FP16
        self._model = self._model.half()

        # Try TensorRT conversion
        try:
            import torch_tensorrt

            # This is a simplified example - real implementation would
            # need proper input specifications
            warnings.warn(
                "TensorRT optimization requires specific input shapes. "
                "Using FP16 only for now."
            )
        except ImportError:
            pass


@ModelWrapper.register("sam2_large")
class SAM2LargeWrapper(ModelWrapper):
    """
    Wrapper for SAM 2 Large model.

    SAM 2 provides additional uncertainty signals:
    - IoU prediction
    - Occlusion scores
    - Stability scores

    Attributes:
        model_name: "sam2_large"
        category: "detection"
        deployment_tier: "cloud" (224M params)
        uncertainty_method: IoU + occlusion scores
        license: Apache 2.0
    """

    def __init__(
        self,
        model_name: str = "sam2_large",
        deployment_tier: str = "cloud",
        **kwargs
    ):
        """Initialize SAM2 Large wrapper."""
        super().__init__(model_name, deployment_tier, **kwargs)

    def load(self) -> None:
        """Load SAM2 Large model."""
        # TODO: Implement SAM2 loading when library is available
        # SAM2 has different architecture and API than SAM1
        raise NotImplementedError(
            "SAM2 Large wrapper not yet implemented. "
            "Waiting for official SAM2 library release."
        )

    def predict(self, image: np.ndarray) -> Tuple[Any, float]:
        """Run SAM2 inference."""
        raise NotImplementedError("SAM2 Large not yet implemented")

    def extract_uncertainty(self, image: np.ndarray) -> Dict[str, float]:
        """Extract uncertainty from SAM2's predictions."""
        raise NotImplementedError("SAM2 Large not yet implemented")
