"""
Abstract base class for deliberative layer model wrappers.

Provides a unified interface for running both VLMs and LLMs in
the deliberative reasoning experiment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class ModelResponse:
    """Response from a model."""
    raw_text: str
    parsed_json: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    inference_time_ms: float
    total_tokens: int
    error: Optional[str] = None


class DeliberativeModelWrapper(ABC):
    """
    Abstract base class for models in the deliberative experiment.

    Subclasses implement either VLM or LLM interfaces, with support
    for both text dump and query interface conditions.
    """

    def __init__(
        self,
        model_name: str,
        model_category: str,  # 'vlm' or 'llm'
        model_size: str,
        device: Optional[str] = None
    ):
        """
        Initialize the model wrapper.

        Args:
            model_name: Identifier for the model
            model_category: 'vlm' or 'llm'
            model_size: Size descriptor (e.g., '232M', '3B')
            device: Device to run on ('cuda', 'cpu', etc.)
        """
        self.model_name = model_name
        self.model_category = model_category
        self.model_size = model_size
        self._device = device
        self._model = None
        self._is_loaded = False

    @property
    def device(self) -> str:
        """Get device, auto-detecting if necessary."""
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
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    def ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self.load()

    @abstractmethod
    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """
        Run the model with a text dump prompt.

        Args:
            prompt: Full prompt with scene description
            image: Optional image for VLMs
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ModelResponse with raw text and parsed JSON
        """
        pass

    @abstractmethod
    def run_query_interface(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Any,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_tool_calls: int = 20
    ) -> ModelResponse:
        """
        Run the model with query interface (tool calling).

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            tools: Tool definitions
            tool_executor: Object with execute_tool(name, args) method
            image: Optional image for VLMs
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_tool_calls: Maximum number of tool calls allowed

        Returns:
            ModelResponse with raw text, parsed JSON, and tool calls
        """
        pass

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._is_loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "category": self.model_category,
            "size": self.model_size,
            "device": self._device,
            "is_loaded": self._is_loaded,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"{self.__class__.__name__}({self.model_name}, {self.model_size}, {status})"


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Parse JSON from a model response.

    Args:
        text: Raw response text

    Returns:
        Parsed JSON dict or empty dict if parsing fails
    """
    import json
    import re

    # Try to find JSON in code block
    code_block = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}
