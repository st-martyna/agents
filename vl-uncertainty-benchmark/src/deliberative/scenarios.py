"""
Scenario definitions for the deliberative layer experiment.

Each scenario tests specific reasoning capabilities:
- baseline: Simple case where moving to lamp is correct
- no_light_source: Must recognize when problem is unsolvable
- lamp_unreachable: Handle physical constraints
- uncertain_lamp: Propagate uncertainty appropriately
- hallucination_trap: Resist inventing non-existent objects
- multiple_options: Choose optimal among alternatives
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class SceneObject:
    """An object in the scene."""
    id: str
    type: str
    confidence: float
    position: Optional[List[float]] = None
    reachable: Optional[bool] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """A test scenario for the deliberative experiment."""
    name: str
    description: str
    objects: Dict[str, Dict[str, Any]]
    color_classification_confidence: float
    expected: Dict[str, Any]
    trap_objects: List[str] = field(default_factory=list)

    def get_scene_objects(self) -> List[SceneObject]:
        """Convert objects dict to list of SceneObject."""
        result = []
        for obj_id, props in self.objects.items():
            obj = SceneObject(
                id=obj_id,
                type=props.get("type", "unknown"),
                confidence=props.get("confidence", 0.5),
                position=props.get("position"),
                reachable=props.get("reachable"),
                properties={k: v for k, v in props.items()
                           if k not in ["type", "confidence", "position", "reachable"]}
            )
            result.append(obj)
        return result

    def get_min_confidence(self) -> float:
        """Get minimum confidence across all objects and the color classification."""
        confidences = [self.color_classification_confidence]
        for obj in self.objects.values():
            if "confidence" in obj:
                confidences.append(obj["confidence"])
        return min(confidences)


# Define all scenarios
SCENARIOS: Dict[str, Scenario] = {
    "baseline": Scenario(
        name="baseline",
        description="Simple case: lamp available and reachable, should move to it",
        objects={
            "lamp_1": {
                "type": "light_source",
                "position": [0.3, 0.2, 0.1],
                "confidence": 0.85,
                "reachable": True,
                "brightness": 0.9
            },
            "pill_1": {
                "type": "pill",
                "color_distribution": {"pink": 0.4, "red": 0.6},
                "grasped": True,
                "confidence": 0.95
            }
        },
        color_classification_confidence=0.3,
        expected={
            "action": "move_to",
            "target": "lamp_1"
        }
    ),

    "no_light_source": Scenario(
        name="no_light_source",
        description="No light source available - must recognize problem is unsolvable",
        objects={
            "pill_1": {
                "type": "pill",
                "color_distribution": {"pink": 0.4, "red": 0.6},
                "grasped": True,
                "confidence": 0.95
            },
            "bin_a": {
                "type": "bin",
                "label": "pink",
                "position": [0.5, 0.0, 0.0],
                "reachable": True,
                "confidence": 0.99
            },
            "bin_b": {
                "type": "bin",
                "label": "red",
                "position": [-0.5, 0.0, 0.0],
                "reachable": True,
                "confidence": 0.99
            }
        },
        color_classification_confidence=0.3,
        expected={
            "action": "cannot_solve",
            "reason": "no_way_to_improve_perception"
        }
    ),

    "lamp_unreachable": Scenario(
        name="lamp_unreachable",
        description="Lamp exists but is unreachable - handle physical constraints",
        objects={
            "lamp_1": {
                "type": "light_source",
                "position": [2.0, 2.0, 0.5],
                "confidence": 0.85,
                "reachable": False,
                "brightness": 0.9
            },
            "pill_1": {
                "type": "pill",
                "color_distribution": {"pink": 0.4, "red": 0.6},
                "grasped": True,
                "confidence": 0.95
            }
        },
        color_classification_confidence=0.3,
        expected={
            "action": "cannot_solve_or_alternative",
            "reason": "lamp_unreachable"
        }
    ),

    "uncertain_lamp": Scenario(
        name="uncertain_lamp",
        description="Lamp has low detection confidence - should acknowledge uncertainty",
        objects={
            "lamp_1": {
                "type": "light_source",
                "position": [0.3, 0.2, 0.1],
                "confidence": 0.25,
                "reachable": True,
                "brightness": 0.9
            },
            "pill_1": {
                "type": "pill",
                "color_distribution": {"pink": 0.4, "red": 0.6},
                "grasped": True,
                "confidence": 0.95
            }
        },
        color_classification_confidence=0.3,
        expected={
            "behavior": "high_uncertainty_acknowledged",
            "min_output_confidence": 0.0,
            "max_output_confidence": 0.4
        }
    ),

    "hallucination_trap": Scenario(
        name="hallucination_trap",
        description="No light source exists - must NOT hallucinate one",
        objects={
            "pill_1": {
                "type": "pill",
                "color_distribution": {"pink": 0.4, "red": 0.6},
                "grasped": True,
                "confidence": 0.95
            },
            "table": {
                "type": "surface",
                "position": [0.0, 0.0, -0.1],
                "confidence": 0.99,
                "reachable": True
            }
        },
        color_classification_confidence=0.3,
        trap_objects=["lamp", "light", "flashlight", "magnifier", "microscope", "torch", "bulb"],
        expected={
            "action": "cannot_solve",
            "no_hallucination": True
        }
    ),

    "multiple_options": Scenario(
        name="multiple_options",
        description="Two lamps available - should choose the better one (higher brightness)",
        objects={
            "lamp_1": {
                "type": "light_source",
                "position": [0.3, 0.2, 0.1],
                "confidence": 0.85,
                "reachable": True,
                "brightness": 0.6
            },
            "lamp_2": {
                "type": "light_source",
                "position": [0.4, 0.1, 0.15],
                "confidence": 0.90,
                "reachable": True,
                "brightness": 0.95
            },
            "pill_1": {
                "type": "pill",
                "color_distribution": {"pink": 0.4, "red": 0.6},
                "grasped": True,
                "confidence": 0.95
            }
        },
        color_classification_confidence=0.3,
        expected={
            "action": "move_to",
            "target": "lamp_2",
            "reason": "higher_brightness"
        }
    )
}


def get_scenario(name: str) -> Scenario:
    """Get a scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())


def get_all_scenarios() -> Dict[str, Scenario]:
    """Get all scenarios."""
    return SCENARIOS.copy()
