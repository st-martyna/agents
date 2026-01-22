"""
Mock tool implementations for the query interface condition.

These tools simulate a robot's perception and planning APIs, allowing
the deliberative layer to query the scene state.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from .scenarios import Scenario


@dataclass
class ToolCallRecord:
    """Record of a tool call made during reasoning."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any


class SceneToolkit:
    """
    Toolkit for querying a scenario's scene state.

    Implements the five tools available to the deliberative layer:
    - find_objects
    - get_object_details
    - check_reachable
    - get_affecting_factors
    - estimate_effect
    """

    def __init__(self, scenario: Scenario):
        """
        Initialize the toolkit with a scenario.

        Args:
            scenario: The scenario providing the scene state
        """
        self.scenario = scenario
        self.call_history: List[ToolCallRecord] = []

    def reset_history(self):
        """Clear the call history."""
        self.call_history = []

    def get_call_history(self) -> List[ToolCallRecord]:
        """Get the history of tool calls."""
        return self.call_history.copy()

    def get_queries_made(self) -> List[str]:
        """Get a list of query strings made."""
        queries = []
        for record in self.call_history:
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in record.arguments.items())
            queries.append(f"{record.tool_name}({args_str})")
        return queries

    def _record_call(self, tool_name: str, arguments: Dict[str, Any], result: Any):
        """Record a tool call."""
        self.call_history.append(ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result
        ))

    def find_objects(self, type: str) -> List[Dict[str, Any]]:
        """
        Find all objects of a given type.

        Args:
            type: Object type to search for ("light_source", "pill", "bin", "surface")

        Returns:
            List of objects with id, type, and confidence
        """
        result = []
        for obj_id, props in self.scenario.objects.items():
            if props.get("type") == type:
                result.append({
                    "id": obj_id,
                    "type": props.get("type"),
                    "confidence": props.get("confidence", 0.5)
                })

        self._record_call("find_objects", {"type": type}, result)
        return result

    def get_object_details(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed properties of an object.

        Args:
            id: Object identifier

        Returns:
            Full object properties or None if not found
        """
        if id not in self.scenario.objects:
            result = None
        else:
            result = {"id": id, **self.scenario.objects[id]}

        self._record_call("get_object_details", {"id": id}, result)
        return result

    def check_reachable(self, id: str) -> Optional[bool]:
        """
        Check if an object is reachable.

        Args:
            id: Object identifier

        Returns:
            True/False if object exists, None if object not found
        """
        if id not in self.scenario.objects:
            result = None
        else:
            result = self.scenario.objects[id].get("reachable", True)

        self._record_call("check_reachable", {"id": id}, result)
        return result

    def get_affecting_factors(self, property: str) -> List[str]:
        """
        Get factors that affect a property.

        Args:
            property: Property name (e.g., "color_classification")

        Returns:
            List of affecting factors
        """
        # Static mapping of properties to affecting factors
        factors_map = {
            "color_classification": ["lighting", "distance", "camera_angle", "object_surface"],
            "color_classification_confidence": ["lighting", "distance", "camera_angle"],
            "visibility": ["lighting", "occlusion", "distance"],
            "detection_confidence": ["lighting", "distance", "object_size"],
            "grasp_success": ["object_position", "gripper_state", "object_size"],
        }

        result = factors_map.get(property, ["unknown"])
        self._record_call("get_affecting_factors", {"property": property}, result)
        return result

    def estimate_effect(
        self,
        action_type: str,
        target_id: str,
        property: str
    ) -> Dict[str, float]:
        """
        Estimate the effect of an action on a property.

        Args:
            action_type: Type of action (e.g., "move_to")
            target_id: Target object ID
            property: Property to estimate effect on

        Returns:
            Dict with "delta" (expected change) and "confidence" (estimate reliability)
        """
        result = {"delta": 0.0, "confidence": 0.0}

        if target_id not in self.scenario.objects:
            self._record_call("estimate_effect", {
                "action_type": action_type,
                "target_id": target_id,
                "property": property
            }, result)
            return result

        obj = self.scenario.objects[target_id]

        if action_type == "move_to" and obj.get("type") == "light_source":
            if property in ["color_classification", "color_classification_confidence"]:
                # Moving to a light source improves color classification
                brightness = obj.get("brightness", 0.5)
                reachable = obj.get("reachable", True)
                obj_confidence = obj.get("confidence", 0.5)

                if reachable:
                    # Delta based on brightness
                    delta = brightness * 0.5  # Max 50% improvement
                    # Confidence based on object detection confidence
                    confidence = obj_confidence * 0.8
                    result = {"delta": delta, "confidence": confidence}
                else:
                    result = {"delta": 0.0, "confidence": 0.9}  # Confident it won't help

        self._record_call("estimate_effect", {
            "action_type": action_type,
            "target_id": target_id,
            "property": property
        }, result)
        return result

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool result
        """
        tool_map = {
            "find_objects": self.find_objects,
            "get_object_details": self.get_object_details,
            "check_reachable": self.check_reachable,
            "get_affecting_factors": self.get_affecting_factors,
            "estimate_effect": self.estimate_effect,
        }

        if tool_name not in tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        return tool_map[tool_name](**arguments)


def parse_react_tool_call(action_str: str) -> tuple:
    """
    Parse a ReAct-style tool call string.

    Args:
        action_str: String like "find_objects(type='light_source')"

    Returns:
        Tuple of (tool_name, arguments_dict)
    """
    import re
    import ast

    # Match tool_name(args)
    match = re.match(r'(\w+)\((.*)\)', action_str.strip())
    if not match:
        return None, {}

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    if not args_str:
        return tool_name, {}

    # Parse arguments
    args = {}

    # Try to parse as keyword arguments
    # Handle both key=value and key="value" formats
    kwarg_pattern = r'(\w+)\s*=\s*([^,]+)'
    for kwmatch in re.finditer(kwarg_pattern, args_str):
        key = kwmatch.group(1)
        value_str = kwmatch.group(2).strip()

        # Try to evaluate the value
        try:
            # Remove quotes if present
            if (value_str.startswith('"') and value_str.endswith('"')) or \
               (value_str.startswith("'") and value_str.endswith("'")):
                value = value_str[1:-1]
            else:
                value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str

        args[key] = value

    # If no kwargs found, try positional argument
    if not args and args_str:
        # Assume it's the first required argument based on tool
        try:
            if (args_str.startswith('"') and args_str.endswith('"')) or \
               (args_str.startswith("'") and args_str.endswith("'")):
                value = args_str[1:-1]
            else:
                value = ast.literal_eval(args_str)
        except (ValueError, SyntaxError):
            value = args_str

        # Map tool to first argument name
        first_arg_map = {
            "find_objects": "type",
            "get_object_details": "id",
            "check_reachable": "id",
            "get_affecting_factors": "property",
        }
        if tool_name in first_arg_map:
            args[first_arg_map[tool_name]] = value

    return tool_name, args


def format_tool_result(result: Any) -> str:
    """Format a tool result for display in ReAct-style prompts."""
    import json

    if result is None:
        return "null (object not found)"
    elif isinstance(result, bool):
        return str(result).lower()
    elif isinstance(result, (dict, list)):
        return json.dumps(result, indent=2)
    else:
        return str(result)
