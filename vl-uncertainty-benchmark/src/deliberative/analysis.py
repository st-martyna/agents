"""
Results analysis for the deliberative layer experiment.

Generates reports comparing text dump vs query interface conditions,
analyzing hallucination rates, uncertainty propagation, and correctness.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

from .metrics import TrialResult, aggregate_results


def generate_report(
    results: List[TrialResult],
    output_dir: str,
    experiment_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a comprehensive analysis report.

    Args:
        results: List of trial results
        output_dir: Directory to save report
        experiment_config: Optional experiment configuration

    Returns:
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw results
    raw_results = [r.to_dict() for r in results]
    with open(output_path / "raw_results.json", "w") as f:
        json.dump(raw_results, f, indent=2, default=str)

    # Aggregate results
    aggregated = aggregate_results(results)

    # Generate markdown report
    report = generate_markdown_report(results, aggregated, experiment_config)

    report_path = output_path / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    return str(report_path)


def generate_markdown_report(
    results: List[TrialResult],
    aggregated: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Generate markdown report content."""

    # Group results by condition
    text_dump_results = [r for r in results if r.condition == "text_dump"]
    query_results = [r for r in results if r.condition == "query_interface"]

    # Group by model
    by_model = {}
    all_conditions = ["text_dump", "query_interface", "query_constrained", "query_verify_first"]
    for r in results:
        if r.model not in by_model:
            by_model[r.model] = {c: [] for c in all_conditions}
        if r.condition in by_model[r.model]:
            by_model[r.model][r.condition].append(r)

    # Build report
    report = f"""# Deliberative Layer Experiment Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This experiment tests whether vision-language models and LLMs behave differently
when given a **typed query interface** vs a **text dump** of scene information.

**Key Question:** Is constrained program synthesis viable for uncertainty-aware
robotics middleware?

### Overall Results

| Metric | Text Dump | Query Interface | Difference |
|--------|-----------|-----------------|------------|
"""

    # Compute per-condition stats
    if text_dump_results:
        td_hallucination = sum(r.hallucination_detected for r in text_dump_results) / len(text_dump_results)
        td_correct = sum(r.correct_action for r in text_dump_results) / len(text_dump_results)
        td_schema = sum(r.schema_valid for r in text_dump_results) / len(text_dump_results)
        td_uncertainty = np.mean([r.uncertainty_propagation for r in text_dump_results])
    else:
        td_hallucination = td_correct = td_schema = td_uncertainty = 0

    if query_results:
        qi_hallucination = sum(r.hallucination_detected for r in query_results) / len(query_results)
        qi_correct = sum(r.correct_action for r in query_results) / len(query_results)
        qi_schema = sum(r.schema_valid for r in query_results) / len(query_results)
        qi_uncertainty = np.mean([r.uncertainty_propagation for r in query_results])
        qi_discipline = np.mean([r.query_discipline for r in query_results])
    else:
        qi_hallucination = qi_correct = qi_schema = qi_uncertainty = qi_discipline = 0

    report += f"| Hallucination Rate | {td_hallucination:.1%} | {qi_hallucination:.1%} | {qi_hallucination - td_hallucination:+.1%} |\n"
    report += f"| Correct Action | {td_correct:.1%} | {qi_correct:.1%} | {qi_correct - td_correct:+.1%} |\n"
    report += f"| Schema Valid | {td_schema:.1%} | {qi_schema:.1%} | {qi_schema - td_schema:+.1%} |\n"
    report += f"| Uncertainty Propagation | {td_uncertainty:.2f} | {qi_uncertainty:.2f} | {qi_uncertainty - td_uncertainty:+.2f} |\n"

    if query_results:
        report += f"| Query Discipline | N/A | {qi_discipline:.1%} | N/A |\n"

    report += f"""

### Key Findings

"""

    # Analyze findings
    findings = []

    # Hallucination analysis
    if td_hallucination > qi_hallucination + 0.1:
        findings.append("**Text dump leads to more hallucinations** - Models are more likely to invent objects when given all information upfront.")
    elif qi_hallucination > td_hallucination + 0.1:
        findings.append("**Query interface leads to more hallucinations** - Surprising result that may indicate issues with tool use.")
    else:
        findings.append("**Hallucination rates are similar** between conditions.")

    # Correctness analysis
    if qi_correct > td_correct + 0.1:
        findings.append("**Query interface improves correctness** - Structured discovery leads to better reasoning.")
    elif td_correct > qi_correct + 0.1:
        findings.append("**Text dump achieves higher correctness** - Full context helps decision-making.")
    else:
        findings.append("**Correctness is similar** between conditions.")

    # Uncertainty analysis
    if qi_uncertainty > td_uncertainty + 0.1:
        findings.append("**Query interface improves uncertainty propagation** - Models are better calibrated with structured queries.")
    elif td_uncertainty > qi_uncertainty + 0.1:
        findings.append("**Text dump achieves better uncertainty propagation** - Unexpected finding.")

    for finding in findings:
        report += f"- {finding}\n"

    # Recommendation
    report += """

### Recommendation for Robotics Middleware

"""

    if qi_hallucination <= td_hallucination and qi_correct >= td_correct:
        report += """**The query interface approach is viable.** Models show equal or better performance
when required to explicitly query for information. This supports the use of
constrained program synthesis for uncertainty-aware control.
"""
    else:
        report += """**Results are mixed.** Consider hybrid approaches that provide some context upfront
while still requiring explicit queries for critical decisions.
"""

    # Per-model breakdown
    report += """

## Per-Model Analysis

"""

    for model, conditions in by_model.items():
        report += f"### {model}\n\n"

        for condition, trials in conditions.items():
            if not trials:
                continue

            n = len(trials)
            hallucination = sum(t.hallucination_detected for t in trials) / n if n > 0 else 0
            correct = sum(t.correct_action for t in trials) / n if n > 0 else 0
            schema = sum(t.schema_valid for t in trials) / n if n > 0 else 0
            avg_time = np.mean([t.inference_time_ms for t in trials]) if n > 0 else 0

            report += f"**{condition.replace('_', ' ').title()}** ({n} trials)\n"
            report += f"- Hallucination: {hallucination:.1%}\n"
            report += f"- Correct Action: {correct:.1%}\n"
            report += f"- Schema Valid: {schema:.1%}\n"
            report += f"- Avg Inference: {avg_time:.0f}ms\n\n"

    # Per-scenario breakdown
    report += """

## Per-Scenario Analysis

"""

    by_scenario = {}
    for r in results:
        if r.scenario not in by_scenario:
            by_scenario[r.scenario] = []
        by_scenario[r.scenario].append(r)

    for scenario, trials in by_scenario.items():
        n = len(trials)
        hallucination = sum(t.hallucination_detected for t in trials) / n if n > 0 else 0
        correct = sum(t.correct_action for t in trials) / n if n > 0 else 0
        appropriate_fail = sum(t.appropriate_failure for t in trials) / n if n > 0 else 0

        report += f"### {scenario}\n\n"
        report += f"- Trials: {n}\n"
        report += f"- Hallucination: {hallucination:.1%}\n"
        report += f"- Correct Action: {correct:.1%}\n"
        report += f"- Appropriate Failure: {appropriate_fail:.1%}\n"

        # Show hallucinated entities if any
        all_hallucinated = []
        for t in trials:
            all_hallucinated.extend(t.hallucinated_entities)
        if all_hallucinated:
            unique = list(set(all_hallucinated))
            report += f"- Hallucinated entities: {', '.join(unique)}\n"

        report += "\n"

    # Hallucination analysis
    report += """

## Hallucination Analysis

"""

    hallucination_trials = [r for r in results if r.hallucination_detected]
    if hallucination_trials:
        report += f"Total hallucination instances: {len(hallucination_trials)}\n\n"

        # Group by entity type
        entity_counts = {}
        for t in hallucination_trials:
            for entity in t.hallucinated_entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1

        if entity_counts:
            report += "Most common hallucinated entities:\n"
            for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
                report += f"- {entity}: {count} times\n"
    else:
        report += "No hallucinations detected in any trial.\n"

    # CANNOT_SOLVE analysis
    report += """

## CANNOT_SOLVE Behavior Analysis

Testing whether models appropriately recognize unsolvable situations.

"""

    unsolvable_scenarios = ["no_light_source", "hallucination_trap"]
    for scenario in unsolvable_scenarios:
        scenario_trials = [r for r in results if r.scenario == scenario]
        if scenario_trials:
            appropriate = sum(r.appropriate_failure for r in scenario_trials)
            total = len(scenario_trials)
            report += f"**{scenario}**: {appropriate}/{total} ({appropriate/total:.0%}) correctly identified as unsolvable\n"

    # Configuration
    if config:
        report += """

## Experiment Configuration

```json
"""
        report += json.dumps(config, indent=2)
        report += "\n```\n"

    report += """

---

*Generated by VL-Uncertainty-Benchmark Deliberative Experiment*
"""

    return report


def compare_conditions(
    results: List[TrialResult]
) -> Dict[str, Any]:
    """
    Statistical comparison between conditions.

    Args:
        results: List of trial results

    Returns:
        Dictionary with comparison statistics
    """
    text_dump = [r for r in results if r.condition == "text_dump"]
    query = [r for r in results if r.condition == "query_interface"]

    comparison = {
        "text_dump": {
            "n": len(text_dump),
            "hallucination_rate": sum(r.hallucination_detected for r in text_dump) / len(text_dump) if text_dump else 0,
            "correct_rate": sum(r.correct_action for r in text_dump) / len(text_dump) if text_dump else 0,
            "schema_valid_rate": sum(r.schema_valid for r in text_dump) / len(text_dump) if text_dump else 0,
            "mean_uncertainty_propagation": np.mean([r.uncertainty_propagation for r in text_dump]) if text_dump else 0,
        },
        "query_interface": {
            "n": len(query),
            "hallucination_rate": sum(r.hallucination_detected for r in query) / len(query) if query else 0,
            "correct_rate": sum(r.correct_action for r in query) / len(query) if query else 0,
            "schema_valid_rate": sum(r.schema_valid for r in query) / len(query) if query else 0,
            "mean_uncertainty_propagation": np.mean([r.uncertainty_propagation for r in query]) if query else 0,
            "mean_query_discipline": np.mean([r.query_discipline for r in query]) if query else 0,
        }
    }

    # Compute differences
    comparison["difference"] = {
        "hallucination_rate": comparison["query_interface"]["hallucination_rate"] - comparison["text_dump"]["hallucination_rate"],
        "correct_rate": comparison["query_interface"]["correct_rate"] - comparison["text_dump"]["correct_rate"],
        "schema_valid_rate": comparison["query_interface"]["schema_valid_rate"] - comparison["text_dump"]["schema_valid_rate"],
        "uncertainty_propagation": comparison["query_interface"]["mean_uncertainty_propagation"] - comparison["text_dump"]["mean_uncertainty_propagation"],
    }

    return comparison
