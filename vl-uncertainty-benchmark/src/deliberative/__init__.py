"""
Deliberative layer experiment module.

Tests whether VLMs and LLMs behave differently when given a typed query interface
vs a text dump of scene information, to validate constrained program synthesis
for uncertainty-aware robotics middleware.
"""

from .scenarios import Scenario, SCENARIOS
from .prompts import (
    TEXT_DUMP_TEMPLATE,
    QUERY_INTERFACE_SYSTEM_PROMPT,
    QUERY_INTERFACE_USER_PROMPT,
    QUERY_CONSTRAINED_SYSTEM_PROMPT,
    QUERY_CONSTRAINED_USER_PROMPT,
    QUERY_VERIFY_FIRST_SYSTEM_PROMPT,
    QUERY_VERIFY_FIRST_USER_PROMPT,
    CLAUDE_TOOLS,
    CLAUDE_TOOLS_EXTENDED,
    REACT_STYLE_PROMPT,
    REACT_CONSTRAINED_PROMPT,
    REACT_VERIFY_FIRST_PROMPT,
    create_text_dump_prompt,
    create_query_interface_prompt,
    extract_json_from_response,
    parse_react_response
)
from .tools import SceneToolkit, parse_react_tool_call
from .metrics import (
    TrialResult,
    UncertaintyMetrics,
    detect_hallucinations,
    compute_query_discipline,
    compute_uncertainty_propagation,
    compute_uncertainty_metrics,
    check_correct_action,
    compute_trial_metrics,
    aggregate_results,
    validate_response_schema,
    extract_referenced_objects,
    evaluate_trial
)
from .models import (
    DeliberativeModelWrapper,
    ModelResponse,
    parse_json_response,
    ClaudeWrapper,
    QwenInstructWrapper,
    Florence2DeliberativeWrapper,
    Qwen2VLDeliberativeWrapper,
    create_vlm_wrapper
)
from .images import create_scene_image, generate_scenario_images, create_dummy_image
from .analysis import generate_report, generate_markdown_report, compare_conditions

__all__ = [
    # Scenarios
    "Scenario",
    "SCENARIOS",
    # Prompts
    "TEXT_DUMP_TEMPLATE",
    "QUERY_INTERFACE_SYSTEM_PROMPT",
    "QUERY_INTERFACE_USER_PROMPT",
    "CLAUDE_TOOLS",
    "REACT_STYLE_PROMPT",
    "create_text_dump_prompt",
    "create_query_interface_prompt",
    "extract_json_from_response",
    "parse_react_response",
    # Tools
    "SceneToolkit",
    "parse_react_tool_call",
    # Metrics
    "TrialResult",
    "detect_hallucinations",
    "compute_query_discipline",
    "compute_uncertainty_propagation",
    "check_correct_action",
    "compute_trial_metrics",
    "aggregate_results",
    "validate_response_schema",
    "extract_referenced_objects",
    "evaluate_trial",
    # Models
    "DeliberativeModelWrapper",
    "ModelResponse",
    "parse_json_response",
    "ClaudeWrapper",
    "QwenInstructWrapper",
    "Florence2DeliberativeWrapper",
    "Qwen2VLDeliberativeWrapper",
    "create_vlm_wrapper",
    # Images
    "create_scene_image",
    "generate_scenario_images",
    "create_dummy_image",
    # Analysis
    "generate_report",
    "generate_markdown_report",
    "compare_conditions",
]
