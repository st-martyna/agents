"""
Prompt templates for the deliberative layer experiment.

Two conditions:
- Condition A (text_dump): Full scene description provided upfront
- Condition B (query_interface): Model must query tools to discover scene
"""

from typing import Dict, Any, List
from .scenarios import Scenario, SceneObject


# =============================================================================
# SHARED PROBLEM STATEMENT
# =============================================================================

PROBLEM_STATEMENT = """You are a robot's deliberative reasoning system. Your task is to help improve perception confidence for a color classification task.

SITUATION:
- You are holding a pill that needs to be sorted into either a "pink" bin or a "red" bin
- The current color classification confidence is {color_confidence:.0%} (too low to act reliably)
- You need to find a way to improve the lighting conditions to get better color classification

GOAL:
Determine the best action to improve perception confidence, or recognize if it's impossible.

VALID ACTIONS:
1. move_to(target_id) - Move to a light source to improve lighting
2. cannot_solve(reason) - Indicate the problem cannot be solved with available resources

IMPORTANT CONSTRAINTS:
- Only reference objects that actually exist in the scene
- Only choose reachable targets
- Acknowledge uncertainty when confidence is low
- Do not invent or assume objects that weren't detected"""


# =============================================================================
# CONDITION A: TEXT DUMP
# =============================================================================

TEXT_DUMP_TEMPLATE = """{problem_statement}

SCENE STATE:
{scene_description}

Based on this scene, determine the best action. Respond with a JSON object:
{{
    "reasoning": "your step-by-step reasoning",
    "action": "move_to" or "cannot_solve",
    "target": "object_id if move_to, null otherwise",
    "reason": "reason if cannot_solve, null otherwise",
    "confidence": 0.0-1.0 (your confidence in this decision)
}}"""


def format_scene_for_text_dump(scenario: Scenario) -> str:
    """Format a scenario's scene as a text dump."""
    lines = []
    lines.append(f"Color Classification Confidence: {scenario.color_classification_confidence:.0%}")
    lines.append("")
    lines.append("Detected Objects:")

    for obj_id, props in scenario.objects.items():
        lines.append(f"\n  {obj_id}:")
        lines.append(f"    type: {props.get('type', 'unknown')}")
        lines.append(f"    confidence: {props.get('confidence', 0.5):.0%}")

        if 'position' in props:
            pos = props['position']
            lines.append(f"    position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

        if 'reachable' in props:
            lines.append(f"    reachable: {props['reachable']}")

        if 'brightness' in props:
            lines.append(f"    brightness: {props['brightness']:.0%}")

        if 'color_distribution' in props:
            dist = props['color_distribution']
            dist_str = ", ".join(f"{k}: {v:.0%}" for k, v in dist.items())
            lines.append(f"    color_distribution: {{{dist_str}}}")

        if 'grasped' in props:
            lines.append(f"    grasped: {props['grasped']}")

        if 'label' in props:
            lines.append(f"    label: {props['label']}")

    return "\n".join(lines)


def create_text_dump_prompt(scenario: Scenario) -> str:
    """Create the full prompt for Condition A (text dump)."""
    problem = PROBLEM_STATEMENT.format(
        color_confidence=scenario.color_classification_confidence
    )
    scene = format_scene_for_text_dump(scenario)

    return TEXT_DUMP_TEMPLATE.format(
        problem_statement=problem,
        scene_description=scene
    )


# =============================================================================
# CONDITION B: QUERY INTERFACE
# =============================================================================

QUERY_INTERFACE_SYSTEM_PROMPT = """You are a robot's deliberative reasoning system. You have access to tools to query the scene state.

IMPORTANT: You must use the tools to discover what objects exist in the scene. Do not assume or invent objects.

Your goal is to determine the best action to improve perception confidence for color classification, or recognize when it's impossible."""


QUERY_INTERFACE_USER_PROMPT = """{problem_statement}

You have access to the following tools to query the scene:

1. find_objects(type: str) -> List[{{id, type, confidence}}]
   Find all objects of a given type. Types: "light_source", "pill", "bin", "surface"

2. get_object_details(id: str) -> {{full object properties}} or null
   Get detailed properties of a specific object by ID

3. check_reachable(id: str) -> bool
   Check if an object is reachable by the robot

4. get_affecting_factors(property: str) -> List[str]
   Get factors that affect a property. E.g., "color_classification" -> ["lighting", "distance"]

5. estimate_effect(action_type: str, target_id: str, property: str) -> {{delta: float, confidence: float}}
   Estimate the effect of an action on a property

Use these tools to explore the scene, then provide your decision as JSON:
{{
    "reasoning": "your step-by-step reasoning",
    "action": "move_to" or "cannot_solve",
    "target": "object_id if move_to, null otherwise",
    "reason": "reason if cannot_solve, null otherwise",
    "confidence": 0.0-1.0 (your confidence in this decision),
    "queries_made": ["list of queries you made"]
}}"""


# Tool definitions for Claude's native tool_use
CLAUDE_TOOLS = [
    {
        "name": "find_objects",
        "description": "Find all objects of a given type in the scene. Returns a list of objects with their IDs, types, and detection confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "The type of object to search for. Valid types: 'light_source', 'pill', 'bin', 'surface'"
                }
            },
            "required": ["type"]
        }
    },
    {
        "name": "get_object_details",
        "description": "Get detailed properties of a specific object by its ID. Returns all known properties or null if object doesn't exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The unique identifier of the object"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "check_reachable",
        "description": "Check if an object is physically reachable by the robot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The unique identifier of the object to check"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "get_affecting_factors",
        "description": "Get the factors that affect a given property. For example, 'color_classification' is affected by lighting and distance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "property": {
                    "type": "string",
                    "description": "The property to query factors for. E.g., 'color_classification', 'visibility'"
                }
            },
            "required": ["property"]
        }
    },
    {
        "name": "estimate_effect",
        "description": "Estimate the effect of performing an action on a property. Returns the expected change (delta) and confidence in that estimate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": "The type of action. E.g., 'move_to'"
                },
                "target_id": {
                    "type": "string",
                    "description": "The ID of the target object"
                },
                "property": {
                    "type": "string",
                    "description": "The property to estimate the effect on. E.g., 'color_classification_confidence'"
                }
            },
            "required": ["action_type", "target_id", "property"]
        }
    }
]


# ReAct-style prompt for local models that don't support native tool calling
REACT_STYLE_PROMPT = """{problem_statement}

You have access to the following tools to query the scene:

Tools:
- find_objects(type): Find all objects of a given type. Types: "light_source", "pill", "bin", "surface"
- get_object_details(id): Get detailed properties of an object
- check_reachable(id): Check if an object is reachable
- get_affecting_factors(property): Get factors affecting a property
- estimate_effect(action_type, target_id, property): Estimate effect of an action

Use the following format:

Thought: I need to think about what to do
Action: tool_name(arg1, arg2, ...)
Observation: [result will be provided]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to decide
Final Answer: {{
    "reasoning": "your reasoning",
    "action": "move_to" or "cannot_solve",
    "target": "object_id or null",
    "reason": "reason or null",
    "confidence": 0.0-1.0
}}

Begin!

Thought:"""


def create_query_interface_prompt(
    scenario: Scenario,
    use_react: bool = False
) -> Dict[str, Any]:
    """
    Create prompts for Condition B (query interface).

    Args:
        scenario: The scenario to create prompts for
        use_react: Whether to use ReAct-style (for local models) vs native tools

    Returns:
        Dictionary with 'system', 'user', and optionally 'tools' keys
    """
    problem = PROBLEM_STATEMENT.format(
        color_confidence=scenario.color_classification_confidence
    )

    if use_react:
        return {
            "system": QUERY_INTERFACE_SYSTEM_PROMPT,
            "user": REACT_STYLE_PROMPT.format(problem_statement=problem),
            "tools": None
        }
    else:
        return {
            "system": QUERY_INTERFACE_SYSTEM_PROMPT,
            "user": QUERY_INTERFACE_USER_PROMPT.format(problem_statement=problem),
            "tools": CLAUDE_TOOLS
        }


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON object from a response string."""
    import json
    import re

    # Try to find JSON in the response
    # First, try to find a code block
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object directly
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Return empty dict if nothing found
    return {}


def parse_react_response(response: str) -> tuple:
    """
    Parse a ReAct-style response.

    Returns:
        Tuple of (queries_made, final_answer_dict)
    """
    import re

    queries = []

    # Find all Action: lines
    action_pattern = r'Action:\s*(\w+)\((.*?)\)'
    for match in re.finditer(action_pattern, response):
        tool_name = match.group(1)
        args = match.group(2)
        queries.append(f"{tool_name}({args})")

    # Find final answer
    final_answer = {}
    final_match = re.search(r'Final Answer:\s*(\{[\s\S]*\})', response)
    if final_match:
        try:
            import json
            final_answer = json.loads(final_match.group(1))
        except json.JSONDecodeError:
            final_answer = extract_json_from_response(response)

    return queries, final_answer
