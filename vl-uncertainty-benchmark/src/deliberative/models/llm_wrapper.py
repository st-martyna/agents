"""
LLM wrappers for the deliberative layer experiment.

Provides wrappers for:
- Claude (via Anthropic API)
- Qwen2.5-Instruct (local, quantized)
"""

import os
import time
import json
import re
from typing import Dict, Any, List, Optional

from .base import DeliberativeModelWrapper, ModelResponse, parse_json_response


class ClaudeWrapper(DeliberativeModelWrapper):
    """
    Wrapper for Claude models via Anthropic API.

    Uses native tool_use for the query interface condition.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        """
        Initialize Claude wrapper.

        Args:
            model_name: Claude model ID
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        super().__init__(
            model_name=model_name,
            model_category="llm",
            model_size="unknown",  # Claude doesn't disclose size
            device="api"
        )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def load(self) -> None:
        """Initialize the Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        if not self._api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")

        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._is_loaded = True

    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Run Claude with text dump prompt."""
        self.ensure_loaded()

        start_time = time.time()

        try:
            messages = [{"role": "user", "content": prompt}]

            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )

            raw_text = response.content[0].text
            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],
                inference_time_ms=inference_time,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
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
        """Run Claude with native tool use."""
        self.ensure_loaded()

        start_time = time.time()
        all_tool_calls = []
        total_tokens = 0

        try:
            messages = [{"role": "user", "content": user_prompt}]

            # Agentic loop
            for _ in range(max_tool_calls):
                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    tools=tools,
                    messages=messages
                )

                total_tokens += response.usage.input_tokens + response.usage.output_tokens

                # Check for tool use
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                if not tool_use_blocks:
                    # No more tool calls, extract final response
                    text_blocks = [b for b in response.content if b.type == "text"]
                    raw_text = text_blocks[0].text if text_blocks else ""
                    break

                # Process tool calls
                tool_results = []
                for tool_use in tool_use_blocks:
                    tool_name = tool_use.name
                    tool_input = tool_use.input

                    all_tool_calls.append({
                        "name": tool_name,
                        "arguments": tool_input
                    })

                    # Execute tool
                    result = tool_executor.execute_tool(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result) if result is not None else "null"
                    })

                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # Max tool calls reached
                raw_text = "Max tool calls reached"

            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=all_tool_calls,
                inference_time_ms=inference_time,
                total_tokens=total_tokens
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=all_tool_calls,
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=total_tokens,
                error=str(e)
            )


class QwenInstructWrapper(DeliberativeModelWrapper):
    """
    Wrapper for Qwen2.5-Instruct models (local, quantized).

    Uses ReAct-style prompting for tool calling since local models
    may not support native tool use.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        model_size: str = "7B",
        quantization: str = "4bit",
        device: Optional[str] = None
    ):
        """
        Initialize Qwen wrapper.

        Args:
            model_name: HuggingFace model ID
            model_size: Size descriptor
            quantization: Quantization method ('4bit', '8bit', 'none')
            device: Device to run on
        """
        super().__init__(
            model_name=model_name,
            model_category="llm",
            model_size=model_size,
            device=device
        )
        self.quantization = quantization
        self._tokenizer = None

    def load(self) -> None:
        """Load Qwen model with quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers required. Install with: pip install transformers")

        import torch

        # Set up quantization
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
                print("Warning: bitsandbytes not available, using fp16")

        elif self.quantization == "8bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                print("Warning: bitsandbytes not available, using fp16")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self._is_loaded = True

    def _generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> str:
        """Generate text from prompt."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def run_text_dump(
        self,
        prompt: str,
        image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> ModelResponse:
        """Run Qwen with text dump prompt."""
        self.ensure_loaded()

        start_time = time.time()

        try:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            raw_text = self._generate(formatted, temperature, max_tokens)
            inference_time = (time.time() - start_time) * 1000

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=[],
                inference_time_ms=inference_time,
                total_tokens=0  # Would need to count
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
        """Run Qwen with ReAct-style tool calling."""
        self.ensure_loaded()

        start_time = time.time()
        all_tool_calls = []

        try:
            # Build ReAct prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            full_response = formatted

            # ReAct loop
            for _ in range(max_tool_calls):
                response_text = self._generate(full_response, temperature, max_tokens)
                full_response += response_text

                # Check for Action:
                action_match = re.search(r'Action:\s*(\w+)\((.*?)\)', response_text)
                if not action_match:
                    # No more actions, done
                    break

                tool_name = action_match.group(1)
                args_str = action_match.group(2)

                # Parse arguments
                from ..tools import parse_react_tool_call
                _, args = parse_react_tool_call(f"{tool_name}({args_str})")

                all_tool_calls.append({
                    "name": tool_name,
                    "arguments": args
                })

                # Execute tool
                try:
                    result = tool_executor.execute_tool(tool_name, args)
                    observation = json.dumps(result) if result is not None else "null"
                except Exception as e:
                    observation = f"Error: {e}"

                full_response += f"\nObservation: {observation}\nThought:"

            inference_time = (time.time() - start_time) * 1000

            # Extract final answer
            final_match = re.search(r'Final Answer:\s*(\{[\s\S]*\})', full_response)
            if final_match:
                raw_text = final_match.group(0)
            else:
                raw_text = full_response

            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parse_json_response(raw_text),
                tool_calls=all_tool_calls,
                inference_time_ms=inference_time,
                total_tokens=0
            )

        except Exception as e:
            return ModelResponse(
                raw_text="",
                parsed_json={},
                tool_calls=all_tool_calls,
                inference_time_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                error=str(e)
            )
