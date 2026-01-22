"""
Metrics and measurement functions for the deliberative layer experiment.

Tracks hallucination, query discipline, uncertainty propagation, and correctness.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import re


@dataclass
class TrialResult:
    """Result of a single experimental trial."""

    # Identifiers
    model: str
    model_category: str  # 'vlm' or 'llm'
    model_size: str  # e.g., '232M', '3B', '72B'
    condition: str  # 'text_dump' or 'query_interface'
    scenario: str
    run_number: int

    # Hallucination detection
    hallucination_detected: bool = False
    hallucinated_entities: List[str] = field(default_factory=list)

    # Query discipline (for query_interface condition)
    query_discipline: float = 0.0  # % of referenced entities that were queried first
    queries_made: List[str] = field(default_factory=list)

    # Uncertainty propagation
    uncertainty_propagation: float = 0.0  # How well does output confidence reflect input uncertainty
    input_min_confidence: float = 0.0
    output_stated_confidence: float = 0.0

    # Response quality
    schema_valid: bool = False  # Did response follow expected JSON schema
    appropriate_failure: bool = False  # Did model correctly identify unsolvable cases
    correct_action: bool = False  # Did model choose correct action

    # Performance
    inference_time_ms: float = 0.0
    total_tokens: int = 0

    # Raw data
    raw_response: str = ""
    parsed_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrialResult":
        """Create from dictionary."""
        return cls(**d)


def detect_hallucinations(
    response: str,
    parsed_response: Dict[str, Any],
    scenario_objects: Dict[str, Any],
    trap_objects: List[str]
) -> tuple:
    """
    Detect if the response references non-existent objects.

    Args:
        response: Raw response text
        parsed_response: Parsed JSON response
        scenario_objects: Dict of actual objects in the scenario
        trap_objects: List of object types that would indicate hallucination

    Returns:
        Tuple of (hallucination_detected, list_of_hallucinated_entities)
    """
    hallucinated = []

    # Check for trap object mentions in the response
    response_lower = response.lower()
    for trap in trap_objects:
        if trap.lower() in response_lower:
            # Make sure it's not just in a negation context
            negation_patterns = [
                f"no {trap}",
                f"not.*{trap}",
                f"cannot find.*{trap}",
                f"don't see.*{trap}",
                f"no.*{trap}.*found",
                f"no.*{trap}.*detected",
                f"no.*{trap}.*available",
            ]
            is_negation = any(re.search(p, response_lower) for p in negation_patterns)
            if not is_negation:
                hallucinated.append(trap)

    # Check if target references a non-existent object
    target = parsed_response.get("target")
    if target and target not in scenario_objects:
        # Check if it looks like a made-up object ID
        if not any(target.startswith(obj_id.split("_")[0]) for obj_id in scenario_objects):
            hallucinated.append(f"target:{target}")

    return len(hallucinated) > 0, hallucinated


def compute_query_discipline(
    queries_made: List[str],
    referenced_objects: List[str],
    scenario_objects: Dict[str, Any]
) -> float:
    """
    Compute query discipline: % of referenced objects that were queried first.

    Args:
        queries_made: List of query strings
        referenced_objects: List of object IDs referenced in the final answer
        scenario_objects: Dict of actual objects in the scenario

    Returns:
        Query discipline score (0.0 - 1.0)
    """
    if not referenced_objects:
        return 1.0  # No references = perfect discipline

    # Extract object IDs from queries
    queried_objects = set()
    for query in queries_made:
        # Look for object IDs in the query
        for obj_id in scenario_objects.keys():
            if obj_id in query:
                queried_objects.add(obj_id)

        # Also check for type queries that would reveal objects
        if "find_objects" in query:
            # find_objects queries count as discovering objects of that type
            pass

    # Count how many referenced objects were properly queried
    properly_queried = sum(1 for obj in referenced_objects if obj in queried_objects)

    return properly_queried / len(referenced_objects) if referenced_objects else 1.0


def compute_uncertainty_propagation(
    input_min_confidence: float,
    output_stated_confidence: float
) -> float:
    """
    Compute uncertainty propagation score.

    A well-calibrated system should not output higher confidence than its
    lowest input confidence. This metric measures how well the model
    propagates uncertainty.

    Args:
        input_min_confidence: Minimum confidence in input data
        output_stated_confidence: Confidence stated in output

    Returns:
        Score from 0.0 (poor) to 1.0 (good)
        - 1.0 if output confidence <= input confidence (good propagation)
        - Decreases as output confidence exceeds input confidence
    """
    if output_stated_confidence <= input_min_confidence:
        return 1.0  # Good: not overconfident

    # Penalize overconfidence
    overconfidence = output_stated_confidence - input_min_confidence
    return max(0.0, 1.0 - overconfidence)


def validate_response_schema(parsed_response: Dict[str, Any]) -> bool:
    """
    Validate that the response follows the expected schema.

    Expected schema:
    {
        "reasoning": str,
        "action": "move_to" | "cannot_solve",
        "target": str | null,
        "reason": str | null,
        "confidence": float
    }
    """
    required_keys = ["action", "confidence"]
    for key in required_keys:
        if key not in parsed_response:
            return False

    action = parsed_response.get("action")
    if action not in ["move_to", "cannot_solve"]:
        return False

    confidence = parsed_response.get("confidence")
    if not isinstance(confidence, (int, float)):
        return False
    if not 0.0 <= confidence <= 1.0:
        return False

    if action == "move_to" and not parsed_response.get("target"):
        return False

    return True


def check_correct_action(
    parsed_response: Dict[str, Any],
    expected: Dict[str, Any],
    scenario_objects: Dict[str, Any]
) -> tuple:
    """
    Check if the model chose the correct action.

    Args:
        parsed_response: The model's parsed response
        expected: The expected outcome from the scenario
        scenario_objects: Objects in the scenario

    Returns:
        Tuple of (correct_action: bool, appropriate_failure: bool)
    """
    action = parsed_response.get("action")
    target = parsed_response.get("target")

    expected_action = expected.get("action")
    expected_target = expected.get("target")

    correct_action = False
    appropriate_failure = False

    if expected_action == "cannot_solve":
        # Model should recognize this is unsolvable
        if action == "cannot_solve":
            correct_action = True
            appropriate_failure = True
    elif expected_action == "cannot_solve_or_alternative":
        # Either cannot_solve or a valid alternative is acceptable
        if action == "cannot_solve":
            correct_action = True
            appropriate_failure = True
        elif action == "move_to" and target in scenario_objects:
            # Check if target is a valid alternative (e.g., reachable light)
            obj = scenario_objects.get(target, {})
            if obj.get("type") == "light_source" and obj.get("reachable", True):
                correct_action = True
    elif expected_action == "move_to":
        if action == "move_to":
            if expected_target and target == expected_target:
                correct_action = True
            elif not expected_target and target in scenario_objects:
                # Any valid target is acceptable
                obj = scenario_objects.get(target, {})
                if obj.get("type") == "light_source" and obj.get("reachable", True):
                    correct_action = True

    # Check for uncertainty acknowledgment
    if expected.get("behavior") == "high_uncertainty_acknowledged":
        confidence = parsed_response.get("confidence", 1.0)
        min_conf = expected.get("min_output_confidence", 0.0)
        max_conf = expected.get("max_output_confidence", 1.0)
        if min_conf <= confidence <= max_conf:
            correct_action = True

    return correct_action, appropriate_failure


def extract_referenced_objects(
    parsed_response: Dict[str, Any],
    raw_response: str
) -> List[str]:
    """
    Extract object IDs referenced in the response.

    Args:
        parsed_response: Parsed JSON response
        raw_response: Raw response text

    Returns:
        List of object IDs mentioned
    """
    referenced = []

    # Check target
    target = parsed_response.get("target")
    if target:
        referenced.append(target)

    # Look for object IDs in reasoning
    reasoning = parsed_response.get("reasoning", "")
    text_to_search = reasoning + " " + raw_response

    # Common object ID patterns
    patterns = [
        r'lamp_\d+',
        r'pill_\d+',
        r'bin_[a-z]',
        r'table',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_to_search, re.IGNORECASE)
        referenced.extend(matches)

    return list(set(referenced))


def compute_trial_metrics(
    result: TrialResult,
    scenario_objects: Dict[str, Any],
    expected: Dict[str, Any],
    trap_objects: List[str]
) -> TrialResult:
    """
    Compute all metrics for a trial result.

    Args:
        result: TrialResult with raw data filled in
        scenario_objects: Objects in the scenario
        expected: Expected outcome
        trap_objects: List of trap object types

    Returns:
        Updated TrialResult with metrics computed
    """
    # Detect hallucinations
    result.hallucination_detected, result.hallucinated_entities = detect_hallucinations(
        result.raw_response,
        result.parsed_response,
        scenario_objects,
        trap_objects
    )

    # Validate schema
    result.schema_valid = validate_response_schema(result.parsed_response)

    # Check correctness
    result.correct_action, result.appropriate_failure = check_correct_action(
        result.parsed_response,
        expected,
        scenario_objects
    )

    # Compute query discipline (for query_interface condition)
    if result.condition == "query_interface":
        referenced = extract_referenced_objects(result.parsed_response, result.raw_response)
        result.query_discipline = compute_query_discipline(
            result.queries_made,
            referenced,
            scenario_objects
        )

    # Compute uncertainty propagation
    output_conf = result.parsed_response.get("confidence", 1.0)
    if isinstance(output_conf, (int, float)):
        result.output_stated_confidence = float(output_conf)
        result.uncertainty_propagation = compute_uncertainty_propagation(
            result.input_min_confidence,
            result.output_stated_confidence
        )

    return result


def aggregate_results(results: List[TrialResult]) -> Dict[str, Any]:
    """
    Aggregate results across multiple trials.

    Args:
        results: List of trial results

    Returns:
        Dictionary with aggregated statistics
    """
    if not results:
        return {}

    import numpy as np

    # Group by model and condition
    by_model_condition = {}
    for r in results:
        key = (r.model, r.condition)
        if key not in by_model_condition:
            by_model_condition[key] = []
        by_model_condition[key].append(r)

    aggregated = {
        "total_trials": len(results),
        "by_model_condition": {},
        "overall": {}
    }

    # Compute per-model-condition stats
    for (model, condition), group in by_model_condition.items():
        n = len(group)
        aggregated["by_model_condition"][f"{model}_{condition}"] = {
            "n_trials": n,
            "hallucination_rate": sum(r.hallucination_detected for r in group) / n,
            "schema_valid_rate": sum(r.schema_valid for r in group) / n,
            "correct_action_rate": sum(r.correct_action for r in group) / n,
            "appropriate_failure_rate": sum(r.appropriate_failure for r in group) / n,
            "mean_query_discipline": np.mean([r.query_discipline for r in group]),
            "mean_uncertainty_propagation": np.mean([r.uncertainty_propagation for r in group]),
            "mean_inference_time_ms": np.mean([r.inference_time_ms for r in group]),
        }

    # Compute overall stats
    n_total = len(results)
    aggregated["overall"] = {
        "hallucination_rate": sum(r.hallucination_detected for r in results) / n_total,
        "schema_valid_rate": sum(r.schema_valid for r in results) / n_total,
        "correct_action_rate": sum(r.correct_action for r in results) / n_total,
        "mean_uncertainty_propagation": np.mean([r.uncertainty_propagation for r in results]),
    }

    return aggregated


def evaluate_trial(
    response,
    scenario,
    scenario_name: str,
    model_name: str,
    model_category: str,
    condition: str,
    toolkit,
    run_number: int = 0
) -> TrialResult:
    """
    Evaluate a single trial and return a TrialResult.

    Args:
        response: ModelResponse object from the model wrapper
        scenario: Scenario object
        scenario_name: Name of the scenario
        model_name: Name of the model
        model_category: Category ('vlm' or 'llm')
        condition: Condition ('text_dump' or 'query_interface')
        toolkit: SceneToolkit instance
        run_number: Run index

    Returns:
        TrialResult with all metrics computed
    """
    # Get minimum confidence from scenario objects
    min_confidence = 1.0
    for obj in scenario.objects.values():
        conf = obj.get("confidence", 1.0)
        if conf < min_confidence:
            min_confidence = conf

    # Create initial result
    result = TrialResult(
        model=model_name,
        model_category=model_category,
        model_size="unknown",
        condition=condition,
        scenario=scenario_name,
        run_number=run_number,
        inference_time_ms=response.inference_time_ms,
        total_tokens=response.total_tokens,
        raw_response=response.raw_text,
        parsed_response=response.parsed_json,
        error=response.error,
        input_min_confidence=min_confidence,
        queries_made=[f"{tc['name']}({tc.get('arguments', {})})" for tc in response.tool_calls],
    )

    # Compute metrics if we have a valid response
    if response.parsed_json:
        result = compute_trial_metrics(
            result=result,
            scenario_objects=scenario.objects,
            expected=scenario.expected,
            trap_objects=scenario.trap_objects
        )
    else:
        # No valid JSON response
        result.schema_valid = False

    return result
