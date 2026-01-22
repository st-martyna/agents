#!/usr/bin/env python3
"""
Run the deliberative layer experiment.

Tests whether VLMs and LLMs behave differently when given a typed query interface
vs a text dump of scene information.

Usage:
    python scripts/run_deliberative_experiment.py --models claude-sonnet --scenarios baseline
    python scripts/run_deliberative_experiment.py --all
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deliberative import (
    SCENARIOS,
    SceneToolkit,
    create_text_dump_prompt,
    create_query_interface_prompt,
    CLAUDE_TOOLS,
    TrialResult,
    evaluate_trial,
    generate_report,
    create_scene_image,
)
from src.deliberative.models import (
    ClaudeWrapper,
    QwenInstructWrapper,
    Florence2DeliberativeWrapper,
    Qwen2VLDeliberativeWrapper,
    create_vlm_wrapper,
)


# Model configurations
MODEL_CONFIGS = {
    # LLMs
    "claude-sonnet": {
        "class": ClaudeWrapper,
        "kwargs": {"model_name": "claude-sonnet-4-20250514"},
        "category": "llm",
    },
    "qwen-7b-instruct": {
        "class": QwenInstructWrapper,
        "kwargs": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "model_size": "7B",
            "quantization": "4bit",
        },
        "category": "llm",
    },
    "qwen-3b-instruct": {
        "class": QwenInstructWrapper,
        "kwargs": {
            "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "model_size": "3B",
            "quantization": "4bit",
        },
        "category": "llm",
    },
    # VLMs
    "florence-2-base": {
        "class": Florence2DeliberativeWrapper,
        "kwargs": {
            "model_name": "microsoft/Florence-2-base",
            "model_size": "232M",
        },
        "category": "vlm",
    },
    "florence-2-large": {
        "class": Florence2DeliberativeWrapper,
        "kwargs": {
            "model_name": "microsoft/Florence-2-large",
            "model_size": "770M",
        },
        "category": "vlm",
    },
    "qwen2-vl-3b": {
        "class": Qwen2VLDeliberativeWrapper,
        "kwargs": {
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_size": "3B",
            "quantization": "4bit",
        },
        "category": "vlm",
    },
    "qwen2-vl-7b": {
        "class": Qwen2VLDeliberativeWrapper,
        "kwargs": {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_size": "7B",
            "quantization": "4bit",
        },
        "category": "vlm",
    },
}

# Default configurations
DEFAULT_MODELS = ["claude-sonnet"]
DEFAULT_SCENARIOS = list(SCENARIOS.keys())
RUNS_PER_CONDITION = 3
TEMPERATURE = 0.0
MAX_TOKENS = 2048
TIMEOUT_SECONDS = 120


def load_model(model_name: str):
    """Load a model by name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model = config["class"](**config["kwargs"])
    model.load()
    return model, config["category"]


def run_text_dump_trial(
    model,
    model_name: str,
    model_category: str,
    scenario_name: str,
    scenario,
    image=None,
    run_idx: int = 0,
) -> TrialResult:
    """Run a single text dump condition trial."""
    # Create tool executor for evaluation
    toolkit = SceneToolkit(scenario)

    # Format prompt
    prompt = create_text_dump_prompt(scenario)

    # Run model
    start_time = time.time()
    try:
        response = model.run_text_dump(
            prompt=prompt,
            image=image if model_category == "vlm" else None,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # Check timeout
        if time.time() - start_time > TIMEOUT_SECONDS:
            response.error = "Timeout exceeded"

    except Exception as e:
        from src.deliberative.models import ModelResponse
        response = ModelResponse(
            raw_text="",
            parsed_json={},
            tool_calls=[],
            inference_time_ms=(time.time() - start_time) * 1000,
            total_tokens=0,
            error=str(e),
        )

    # Evaluate trial
    result = evaluate_trial(
        response=response,
        scenario=scenario,
        scenario_name=scenario_name,
        model_name=model_name,
        model_category=model_category,
        condition="text_dump",
        toolkit=toolkit,
    )

    return result


def run_query_interface_trial(
    model,
    model_name: str,
    model_category: str,
    scenario_name: str,
    scenario,
    image=None,
    run_idx: int = 0,
) -> TrialResult:
    """Run a single query interface condition trial."""
    # Create tool executor
    toolkit = SceneToolkit(scenario)

    # Format prompts
    prompts = create_query_interface_prompt(scenario, use_react=False)
    system_prompt = prompts["system"]
    user_prompt = prompts["user"]

    # Run model
    start_time = time.time()
    try:
        response = model.run_query_interface(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=CLAUDE_TOOLS,
            tool_executor=toolkit,
            image=image if model_category == "vlm" else None,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            max_tool_calls=20,
        )

        # Check timeout
        if time.time() - start_time > TIMEOUT_SECONDS:
            response.error = "Timeout exceeded"

    except Exception as e:
        from src.deliberative.models import ModelResponse
        response = ModelResponse(
            raw_text="",
            parsed_json={},
            tool_calls=[],
            inference_time_ms=(time.time() - start_time) * 1000,
            total_tokens=0,
            error=str(e),
        )

    # Evaluate trial
    result = evaluate_trial(
        response=response,
        scenario=scenario,
        scenario_name=scenario_name,
        model_name=model_name,
        model_category=model_category,
        condition="query_interface",
        toolkit=toolkit,
    )

    return result


def run_experiment(
    models: List[str],
    scenarios: List[str],
    runs_per_condition: int = RUNS_PER_CONDITION,
    output_dir: str = "results/deliberative",
    verbose: bool = True,
) -> List[TrialResult]:
    """
    Run the full deliberative experiment.

    Args:
        models: List of model names to test
        scenarios: List of scenario names to test
        runs_per_condition: Number of runs per (model, condition, scenario)
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        List of all trial results
    """
    all_results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate scenario images
    if verbose:
        print("Generating scenario images...")
    scenario_images = {}
    for scenario_name in scenarios:
        if scenario_name in SCENARIOS:
            scenario = SCENARIOS[scenario_name]
            scenario_images[scenario_name] = create_scene_image(
                scenario_name, scenario.objects
            )

    total_trials = len(models) * len(scenarios) * 2 * runs_per_condition
    trial_count = 0

    for model_name in models:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading model: {model_name}")
            print(f"{'='*60}")

        try:
            model, model_category = load_model(model_name)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        for scenario_name in scenarios:
            if scenario_name not in SCENARIOS:
                print(f"Unknown scenario: {scenario_name}")
                continue

            scenario = SCENARIOS[scenario_name]
            image = scenario_images.get(scenario_name)

            for condition in ["text_dump", "query_interface"]:
                for run_idx in range(runs_per_condition):
                    trial_count += 1
                    if verbose:
                        print(
                            f"\n[{trial_count}/{total_trials}] "
                            f"{model_name} / {scenario_name} / {condition} / run {run_idx + 1}"
                        )

                    try:
                        if condition == "text_dump":
                            result = run_text_dump_trial(
                                model=model,
                                model_name=model_name,
                                model_category=model_category,
                                scenario_name=scenario_name,
                                scenario=scenario,
                                image=image,
                                run_idx=run_idx,
                            )
                        else:
                            result = run_query_interface_trial(
                                model=model,
                                model_name=model_name,
                                model_category=model_category,
                                scenario_name=scenario_name,
                                scenario=scenario,
                                image=image,
                                run_idx=run_idx,
                            )

                        all_results.append(result)

                        if verbose:
                            print(f"  Hallucination: {result.hallucination_detected}")
                            print(f"  Correct: {result.correct_action}")
                            print(f"  Schema valid: {result.schema_valid}")
                            print(f"  Time: {result.inference_time_ms:.0f}ms")
                            if result.error:
                                print(f"  Error: {result.error}")

                    except Exception as e:
                        print(f"  Trial failed: {e}")

                    # Save intermediate results
                    intermediate_path = output_path / "intermediate_results.json"
                    with open(intermediate_path, "w") as f:
                        json.dump(
                            [r.to_dict() for r in all_results],
                            f,
                            indent=2,
                            default=str,
                        )

        # Unload model to free memory
        if hasattr(model, "unload"):
            model.unload()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the deliberative layer experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to test. Available: {list(MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=f"Scenarios to test. Available: {list(SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models and scenarios",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_CONDITION,
        help=f"Runs per (model, condition, scenario). Default: {RUNS_PER_CONDITION}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/deliberative",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, config in MODEL_CONFIGS.items():
            category = config["category"]
            print(f"  {name} ({category})")
        return

    if args.list_scenarios:
        print("Available scenarios:")
        for name, scenario in SCENARIOS.items():
            print(f"  {name}: {scenario.description[:60]}...")
        return

    # Determine models and scenarios
    if args.all:
        models = list(MODEL_CONFIGS.keys())
        scenarios = list(SCENARIOS.keys())
    else:
        models = args.models or DEFAULT_MODELS
        scenarios = args.scenarios or DEFAULT_SCENARIOS

    print(f"Deliberative Layer Experiment")
    print(f"==============================")
    print(f"Models: {models}")
    print(f"Scenarios: {scenarios}")
    print(f"Runs per condition: {args.runs}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Check for API key if using Claude
    if any("claude" in m for m in models):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Warning: ANTHROPIC_API_KEY not set. Claude models will fail.")
            print("Set it with: export ANTHROPIC_API_KEY=your-key")
            print()

    # Run experiment
    start_time = time.time()
    results = run_experiment(
        models=models,
        scenarios=scenarios,
        runs_per_condition=args.runs,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"{'='*60}")
    print(f"Total trials: {len(results)}")
    print(f"Total time: {elapsed:.1f}s")

    # Generate report
    if results:
        experiment_config = {
            "models": models,
            "scenarios": scenarios,
            "runs_per_condition": args.runs,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "timestamp": datetime.now().isoformat(),
        }

        report_path = generate_report(
            results=results,
            output_dir=args.output_dir,
            experiment_config=experiment_config,
        )
        print(f"Report saved to: {report_path}")

        # Print summary
        print("\nQuick Summary:")
        hallucinations = sum(r.hallucination_detected for r in results)
        correct = sum(r.correct_action for r in results)
        print(f"  Hallucination rate: {hallucinations}/{len(results)} ({hallucinations/len(results):.1%})")
        print(f"  Correct action rate: {correct}/{len(results)} ({correct/len(results):.1%})")

        # By condition
        text_dump = [r for r in results if r.condition == "text_dump"]
        query = [r for r in results if r.condition == "query_interface"]

        if text_dump:
            td_hall = sum(r.hallucination_detected for r in text_dump) / len(text_dump)
            td_corr = sum(r.correct_action for r in text_dump) / len(text_dump)
            print(f"  Text dump: {td_hall:.1%} hallucination, {td_corr:.1%} correct")

        if query:
            qi_hall = sum(r.hallucination_detected for r in query) / len(query)
            qi_corr = sum(r.correct_action for r in query) / len(query)
            print(f"  Query interface: {qi_hall:.1%} hallucination, {qi_corr:.1%} correct")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
