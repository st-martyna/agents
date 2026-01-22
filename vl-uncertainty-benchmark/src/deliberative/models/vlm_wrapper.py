"""
VLM wrappers for the deliberative layer experiment.

Reuses existing VLM wrappers from src/models/vlm/ where possible,
adapting them for the deliberative reasoning interface.
"""

import time
import json
import re
from typing import Dict, Any, List, Optional
import numpy as np

from .base import DeliberativeModelWrapper, ModelResponse, parse_json_response


class Florence2DeliberativeWrapper(DeliberativeModelWrapper):
    """
    Wrapper for Florence-2 models in deliberative mode.

    Note: Florence-2 is primarily designed for visual tasks, not
    complex reasoning. This tests whether VLMs can handle deliberative
    reasoning at all.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        model_size: str = "232M",
        device: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            model_category="vlm",
            model_size=model_size,
            device=device
        )
        self._processor = None

    def load(self) -> None:
        """Load Florence-2 model."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("transformers required")

        import torch

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True

    def _generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        max_tokens: int = 1024
    ) -> str:
        """Generate text from prompt."""
        import torch
        from PIL import Image

        # Florence-2 requires an image
        if image is None:
            # Create a dummy image
            image = Image.new('RGB', (224, 224), color='white')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=3
            )

        return self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Run Florence-2 with text dump prompt."""
        self.ensure_loaded()

        start_time = time.time()

        try:
            # Florence-2 uses task tokens, but we'll try free-form
            raw_text = self._generate(prompt, image, max_tokens)
            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],
                inference_time_ms=inference_time,
                total_tokens=0
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                error=str(e)
            )

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
        Run Florence-2 with query interface.

        Note: Florence-2 is not designed for multi-turn tool calling,
        so this is a best-effort implementation.
        """
        self.ensure_loaded()

        start_time = time.time()

        try:
            # Combine system and user prompts
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            raw_text = self._generate(combined_prompt, image, max_tokens)
            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],  # Florence-2 doesn't do tool calling
                inference_time_ms=inference_time,
                total_tokens=0
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                error=str(e)
            )


class Qwen2VLDeliberativeWrapper(DeliberativeModelWrapper):
    """
    Wrapper for Qwen2.5-VL models in deliberative mode.

    Qwen2.5-VL has better reasoning capabilities and can handle
    complex multi-turn interactions.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        model_size: str = "3B",
        quantization: str = "4bit",
        device: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            model_category="vlm",
            model_size=model_size,
            device=device
        )
        self.quantization = quantization
        self._processor = None

    def load(self) -> None:
        """Load Qwen2.5-VL model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("transformers required with Qwen2-VL support")

        import torch

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                pass

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._is_loaded = True

    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Run Qwen2.5-VL with text dump prompt."""
        self.ensure_loaded()

        start_time = time.time()

        try:
            import torch
            from PIL import Image

            messages = [{"role": "user", "content": []}]

            if image is not None:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                messages[0]["content"].append({"type": "image", "image": image})

            messages[0]["content"].append({"type": "text", "text": prompt})

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            if image is not None:
                inputs = self._processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                )
            else:
                inputs = self._processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True
                )

            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0
                )

            raw_text = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],
                inference_time_ms=inference_time,
                total_tokens=outputs[0].shape[0]
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                error=str(e)
            )

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
        """Run Qwen2.5-VL with ReAct-style tool calling."""
        # For simplicity, combine prompts and do single-turn
        # A full implementation would do multi-turn with tool execution
        combined = f"{system_prompt}\n\n{user_prompt}"
        return self.run_text_dump(combined, image, temperature, max_tokens)


class LlamaVisionWrapper(DeliberativeModelWrapper):
    """
    Wrapper for Llama-3.2-Vision models.

    Llama Vision models have good reasoning capabilities and can
    handle complex multi-turn interactions with images.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        model_size: str = "11B",
        quantization: str = "4bit",
        device: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            model_category="vlm",
            model_size=model_size,
            device=device
        )
        self.quantization = quantization
        self._processor = None

    def load(self) -> None:
        """Load Llama Vision model."""
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("transformers required with Llama Vision support")

        import torch

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                pass

        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._is_loaded = True

    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Run Llama Vision with text dump prompt."""
        self.ensure_loaded()

        start_time = time.time()

        try:
            import torch
            from PIL import Image

            messages = [{"role": "user", "content": []}]

            if image is not None:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                messages[0]["content"].append({"type": "image"})

            messages[0]["content"].append({"type": "text", "text": prompt})

            text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            if image is not None:
                inputs = self._processor(
                    images=image,
                    text=text,
                    return_tensors="pt"
                ).to(self._model.device)
            else:
                inputs = self._processor(
                    text=text,
                    return_tensors="pt"
                ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0
                )

            raw_text = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],
                inference_time_ms=inference_time,
                total_tokens=outputs[0].shape[0]
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                error=str(e)
            )

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
        """Run Llama Vision with combined prompts."""
        combined = f"{system_prompt}\n\n{user_prompt}"
        return self.run_text_dump(combined, image, temperature, max_tokens)


class PixtralWrapper(DeliberativeModelWrapper):
    """
    Wrapper for Mistral Pixtral models.

    Pixtral is Mistral's multimodal model with vision capabilities.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Pixtral-12B-2409",
        model_size: str = "12B",
        quantization: str = "4bit",
        device: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            model_category="vlm",
            model_size=model_size,
            device=device
        )
        self.quantization = quantization
        self._processor = None

    def load(self) -> None:
        """Load Pixtral model."""
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("transformers required with Pixtral support")

        import torch

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                pass

        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._is_loaded = True

    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Run Pixtral with text dump prompt."""
        self.ensure_loaded()

        start_time = time.time()

        try:
            import torch
            from PIL import Image

            if image is not None:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            else:
                conversation = [
                    {"role": "user", "content": prompt}
                ]

            text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            if image is not None:
                inputs = self._processor(
                    text=text,
                    images=[image],
                    return_tensors="pt"
                ).to(self._model.device)
            else:
                inputs = self._processor(
                    text=text,
                    return_tensors="pt"
                ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0
                )

            raw_text = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],
                inference_time_ms=inference_time,
                total_tokens=outputs[0].shape[0]
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                error=str(e)
            )

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
        """Run Pixtral with combined prompts."""
        combined = f"{system_prompt}\n\n{user_prompt}"
        return self.run_text_dump(combined, image, temperature, max_tokens)


class Idefics3Wrapper(DeliberativeModelWrapper):
    """
    Wrapper for HuggingFace Idefics3 models.

    Idefics3 is a strong open-source VLM with good reasoning.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceM4/Idefics3-8B-Llama3",
        model_size: str = "8B",
        quantization: str = "4bit",
        device: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            model_category="vlm",
            model_size=model_size,
            device=device
        )
        self.quantization = quantization
        self._processor = None

    def load(self) -> None:
        """Load Idefics3 model."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError("transformers required with Idefics3 support")

        import torch

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                pass

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._is_loaded = True

    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Run Idefics3 with text dump prompt."""
        self.ensure_loaded()

        start_time = time.time()

        try:
            import torch
            from PIL import Image

            if image is not None:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            else:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]

            text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            if image is not None:
                inputs = self._processor(
                    text=text,
                    images=[image],
                    return_tensors="pt"
                ).to(self._model.device)
            else:
                inputs = self._processor(
                    text=text,
                    return_tensors="pt"
                ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0
                )

            raw_text = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],
                inference_time_ms=inference_time,
                total_tokens=outputs[0].shape[0]
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                error=str(e)
            )

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
        """Run Idefics3 with combined prompts."""
        combined = f"{system_prompt}\n\n{user_prompt}"
        return self.run_text_dump(combined, image, temperature, max_tokens)


# Factory function to create VLM wrappers
def create_vlm_wrapper(
    model_name: str,
    model_size: str,
    device: Optional[str] = None,
    quantization: str = "4bit"
) -> DeliberativeModelWrapper:
    """
    Create a VLM wrapper by model name.

    Args:
        model_name: Model identifier
        model_size: Size descriptor
        device: Device to run on
        quantization: Quantization method

    Returns:
        Appropriate VLM wrapper
    """
    model_lower = model_name.lower()

    if "florence" in model_lower:
        return Florence2DeliberativeWrapper(
            model_name=model_name,
            model_size=model_size,
            device=device
        )
    elif "qwen" in model_lower and "vl" in model_lower:
        return Qwen2VLDeliberativeWrapper(
            model_name=model_name,
            model_size=model_size,
            quantization=quantization,
            device=device
        )
    elif "llama" in model_lower and "vision" in model_lower:
        return LlamaVisionWrapper(
            model_name=model_name,
            model_size=model_size,
            quantization=quantization,
            device=device
        )
    elif "pixtral" in model_lower:
        return PixtralWrapper(
            model_name=model_name,
            model_size=model_size,
            quantization=quantization,
            device=device
        )
    elif "idefics" in model_lower:
        return Idefics3Wrapper(
            model_name=model_name,
            model_size=model_size,
            quantization=quantization,
            device=device
        )
    else:
        raise ValueError(f"Unknown VLM model: {model_name}")
