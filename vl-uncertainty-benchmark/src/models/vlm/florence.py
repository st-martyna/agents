"""
Florence-2 model wrapper implementations.

Provides wrappers for:
- Florence-2-base: Edge-deployable (232M params)
- Florence-2-large: Cloud model (770M params)

Reference implementation for VLM model wrappers.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from ..base import ModelWrapper
from ...uncertainty.extractors import token_entropy


@ModelWrapper.register("florence2_base")
class Florence2Base(ModelWrapper):
    """
    Wrapper for Florence-2-base vision-language model.

    Florence-2 is a sequence-to-sequence model that can perform various
    vision-language tasks including captioning, object detection, and
    visual question answering.

    Uncertainty is extracted from token-level logits via entropy computation.

    Attributes:
        model_name: "florence2_base"
        category: "vlm"
        deployment_tier: "edge" (232M params)
        uncertainty_method: Token entropy
        license: MIT
    """

    # Task prompts for different capabilities
    TASK_PROMPTS = {
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "object_detection": "<OD>",
        "dense_region_caption": "<DENSE_REGION_CAPTION>",
        "region_proposal": "<REGION_PROPOSAL>",
        "caption_to_phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
        "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
        "region_to_category": "<REGION_TO_CATEGORY>",
        "region_to_description": "<REGION_TO_DESCRIPTION>",
        "ocr": "<OCR>",
        "ocr_with_region": "<OCR_WITH_REGION>",
    }

    def __init__(
        self,
        model_name: str = "florence2_base",
        deployment_tier: str = "edge",
        default_task: str = "caption",
        **kwargs
    ):
        """
        Initialize Florence-2-base wrapper.

        Args:
            model_name: Model identifier
            deployment_tier: Deployment tier
            default_task: Default task for prediction (see TASK_PROMPTS)
            **kwargs: Additional arguments
        """
        super().__init__(model_name, deployment_tier, **kwargs)
        self.default_task = default_task
        self._last_logits = None  # Cache for uncertainty extraction

    def load(self) -> None:
        """Load Florence-2 model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )

        import torch

        model_id = self._config.get("model_id", "microsoft/Florence-2-base")

        # Load model with trust_remote_code for Florence-2
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self._model.to(self.device)
        self._model.eval()

        self._is_loaded = True

    def _preprocess_image(self, image: np.ndarray) -> "PIL.Image.Image":
        """Convert numpy array to PIL Image."""
        from PIL import Image
        import cv2

        # Ensure RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return Image.fromarray(image)

    def predict(
        self,
        image: np.ndarray,
        task: Optional[str] = None,
        text_input: Optional[str] = None,
        max_new_tokens: int = 1024,
        num_beams: int = 3
    ) -> Tuple[str, float]:
        """
        Run Florence-2 inference.

        Args:
            image: Input image (H, W, C)
            task: Task type (see TASK_PROMPTS). Uses default_task if None.
            text_input: Additional text input for grounding tasks
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width

        Returns:
            Tuple of:
                - response: Generated text response
                - confidence: Mean token confidence [0, 1]
        """
        self.ensure_loaded()

        import torch

        # Get task prompt
        task = task or self.default_task
        if task not in self.TASK_PROMPTS:
            raise ValueError(
                f"Unknown task: {task}. "
                f"Available tasks: {list(self.TASK_PROMPTS.keys())}"
            )

        prompt = self.TASK_PROMPTS[task]
        if text_input:
            prompt = prompt + text_input

        # Preprocess
        pil_image = self._preprocess_image(image)
        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with output scores for uncertainty
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode response
        generated_ids = outputs.sequences
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Parse task-specific response
        parsed_response = self._processor.post_process_generation(
            response,
            task=prompt,
            image_size=pil_image.size
        )

        # Cache logits for uncertainty extraction
        if outputs.scores:
            # Stack scores: (num_tokens, vocab_size)
            self._last_logits = torch.stack(outputs.scores, dim=0).cpu().numpy()
        else:
            self._last_logits = None

        # Compute confidence from logits
        if self._last_logits is not None:
            confidence = self._compute_mean_confidence(self._last_logits)
        else:
            confidence = 0.5  # Default if no logits available

        return parsed_response, confidence

    def _compute_mean_confidence(self, logits: np.ndarray) -> float:
        """Compute mean token confidence from logits."""
        from scipy.special import softmax

        # logits shape: (num_tokens, vocab_size)
        probs = softmax(logits, axis=-1)

        # Mean of max probabilities
        max_probs = np.max(probs, axis=-1)
        return float(np.mean(max_probs))

    def extract_uncertainty(
        self,
        image: np.ndarray,
        task: Optional[str] = None,
        text_input: Optional[str] = None,
        recompute: bool = False
    ) -> Dict[str, float]:
        """
        Extract uncertainty from Florence-2 token predictions.

        Uses cached logits from last prediction if available,
        otherwise runs a new forward pass.

        Args:
            image: Input image
            task: Task type
            text_input: Additional text input
            recompute: Force recomputation even if cached logits exist

        Returns:
            Dictionary with uncertainty metrics:
            - token_entropy: Mean normalized entropy across tokens
            - max_entropy: Maximum token entropy in sequence
            - sequence_entropy: Entropy of mean token probabilities
        """
        self.ensure_loaded()

        # Use cached logits or recompute
        if self._last_logits is None or recompute:
            # Run prediction to get logits
            self.predict(image, task=task, text_input=text_input)

        if self._last_logits is None:
            warnings.warn("No logits available for uncertainty extraction")
            return {
                "token_entropy": 0.5,
                "max_entropy": 0.5,
                "min_entropy": 0.5,
                "sequence_entropy": 0.5,
            }

        # Use token_entropy extractor
        return token_entropy(self._last_logits, normalize=True)

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        task: Optional[str] = None,
        text_input: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Combined prediction and uncertainty extraction.

        Efficient as it reuses the forward pass logits.
        """
        # Predict (caches logits)
        prediction, confidence = self.predict(
            image, task=task, text_input=text_input, **kwargs
        )

        # Extract uncertainty from cached logits
        uncertainty = self.extract_uncertainty(
            image, task=task, text_input=text_input, recompute=False
        )

        return prediction, confidence, uncertainty

    def generate_with_full_output(
        self,
        image: np.ndarray,
        task: Optional[str] = None,
        text_input: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with full output including logits and attention.

        Useful for detailed analysis of model behavior.
        """
        self.ensure_loaded()

        import torch

        task = task or self.default_task
        prompt = self.TASK_PROMPTS[task]
        if text_input:
            prompt = prompt + text_input

        pil_image = self._preprocess_image(image)
        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                num_beams=kwargs.get("num_beams", 3),
                output_scores=True,
                output_attentions=kwargs.get("output_attentions", False),
                return_dict_in_generate=True,
            )

        # Decode
        response = self._processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )[0]

        parsed = self._processor.post_process_generation(
            response,
            task=prompt,
            image_size=pil_image.size
        )

        # Collect logits
        if outputs.scores:
            logits = torch.stack(outputs.scores, dim=0).cpu().numpy()
        else:
            logits = None

        self._last_logits = logits

        return {
            "response": parsed,
            "raw_response": response,
            "logits": logits,
            "sequences": outputs.sequences.cpu().numpy(),
            "uncertainty": token_entropy(logits) if logits is not None else None,
        }

    def to_edge(self) -> None:
        """
        Optimize Florence-2 for edge deployment.

        Applies:
        - FP16 quantization
        - Attention optimization
        """
        import torch

        if not self._is_loaded:
            raise RuntimeError("Model must be loaded before edge optimization")

        # Already loaded in FP16 if not CPU
        if self.device == "cpu":
            # Try INT8 quantization for CPU
            try:
                self._model = torch.quantization.quantize_dynamic(
                    self._model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as e:
                warnings.warn(f"INT8 quantization failed: {e}")

        # Enable memory efficient attention if available
        try:
            self._model.config.use_flash_attention_2 = True
        except AttributeError:
            pass


@ModelWrapper.register("florence2_large")
class Florence2Large(Florence2Base):
    """
    Wrapper for Florence-2-large vision-language model.

    Same interface as Florence-2-base but with larger model (770M params).

    Attributes:
        model_name: "florence2_large"
        category: "vlm"
        deployment_tier: "cloud" (770M params)
        uncertainty_method: Token entropy
        license: MIT
    """

    def __init__(
        self,
        model_name: str = "florence2_large",
        deployment_tier: str = "cloud",
        **kwargs
    ):
        """Initialize Florence-2-large wrapper."""
        # Don't call parent __init__ directly to avoid recursion
        ModelWrapper.__init__(self, model_name, deployment_tier, **kwargs)
        self.default_task = kwargs.get("default_task", "caption")
        self._last_logits = None

    def load(self) -> None:
        """Load Florence-2-large model."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )

        import torch

        model_id = self._config.get("model_id", "microsoft/Florence-2-large")

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self._model.to(self.device)
        self._model.eval()

        self._is_loaded = True
