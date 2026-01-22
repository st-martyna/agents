"""
Composable degradation pipeline for chaining multiple transforms.

Provides DegradationPipeline class and utilities for generating test matrices
with all combinations of degradations and severity levels.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Iterator
from dataclasses import dataclass, field
from itertools import product
import copy

from .transforms import (
    apply_degradation,
    DEGRADATION_FUNCTIONS,
    SEVERITY_CONFIGS,
    get_degradation_info,
)


@dataclass
class DegradationStep:
    """A single degradation step in a pipeline."""
    degradation_type: str
    severity: int
    kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.degradation_type not in DEGRADATION_FUNCTIONS:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")
        if not 1 <= self.severity <= 5:
            raise ValueError(f"Severity must be 1-5, got {self.severity}")

    def __repr__(self):
        return f"{self.degradation_type}(severity={self.severity})"


class DegradationPipeline:
    """
    Composable pipeline for chaining multiple image degradations.

    Example:
        >>> pipeline = DegradationPipeline([
        ...     ("lighting", 2),
        ...     ("blur", 3),
        ... ])
        >>> degraded = pipeline(image)

        >>> # Or build incrementally
        >>> pipeline = DegradationPipeline()
        >>> pipeline.add("noise", severity=2)
        >>> pipeline.add("jpeg", severity=3)
        >>> degraded = pipeline(image)
    """

    def __init__(
        self,
        steps: Optional[List[Union[Tuple[str, int], DegradationStep]]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the pipeline.

        Args:
            steps: List of (degradation_type, severity) tuples or DegradationStep objects
            seed: Random seed for reproducible degradations
        """
        self.steps: List[DegradationStep] = []
        self.seed = seed

        if steps:
            for step in steps:
                if isinstance(step, DegradationStep):
                    self.steps.append(step)
                elif isinstance(step, (tuple, list)):
                    if len(step) == 2:
                        self.steps.append(DegradationStep(step[0], step[1]))
                    elif len(step) == 3:
                        self.steps.append(DegradationStep(step[0], step[1], step[2]))
                    else:
                        raise ValueError(f"Invalid step format: {step}")
                else:
                    raise ValueError(f"Invalid step type: {type(step)}")

    def add(
        self,
        degradation_type: str,
        severity: int,
        **kwargs
    ) -> "DegradationPipeline":
        """
        Add a degradation step to the pipeline.

        Args:
            degradation_type: Type of degradation
            severity: Severity level 1-5
            **kwargs: Additional arguments for the degradation

        Returns:
            Self for method chaining
        """
        self.steps.append(DegradationStep(degradation_type, severity, kwargs))
        return self

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all degradation steps to an image.

        Args:
            image: Input image (H, W, C)

        Returns:
            Degraded image
        """
        result = image.copy()

        for i, step in enumerate(self.steps):
            # Set seed if provided (offset by step index for variety)
            kwargs = step.kwargs.copy()
            if self.seed is not None and "seed" not in kwargs:
                kwargs["seed"] = self.seed + i

            result = apply_degradation(
                result,
                step.degradation_type,
                step.severity,
                **kwargs
            )

        return result

    def __repr__(self):
        steps_str = " -> ".join(str(s) for s in self.steps)
        return f"DegradationPipeline([{steps_str}])"

    def __len__(self):
        return len(self.steps)

    def copy(self) -> "DegradationPipeline":
        """Create a deep copy of this pipeline."""
        return DegradationPipeline(
            steps=[copy.deepcopy(s) for s in self.steps],
            seed=self.seed
        )

    def to_dict(self) -> Dict:
        """Serialize pipeline to dictionary."""
        return {
            "steps": [
                {
                    "type": s.degradation_type,
                    "severity": s.severity,
                    "kwargs": s.kwargs
                }
                for s in self.steps
            ],
            "seed": self.seed
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DegradationPipeline":
        """Create pipeline from dictionary."""
        steps = [
            DegradationStep(s["type"], s["severity"], s.get("kwargs", {}))
            for s in d["steps"]
        ]
        return cls(steps=steps, seed=d.get("seed"))

    def get_name(self) -> str:
        """Get a descriptive name for this pipeline."""
        if not self.steps:
            return "clean"
        return "_".join(f"{s.degradation_type}{s.severity}" for s in self.steps)


@dataclass
class TestCase:
    """A test case with degradation configuration and metadata."""
    pipeline: DegradationPipeline
    name: str
    degradation_type: str  # "single", "combination", or "clean"
    severity: Optional[int] = None  # For single degradations

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the degradation pipeline to an image."""
        return self.pipeline(image)


def generate_single_degradation_cases(
    degradation_types: Optional[List[str]] = None,
    severity_levels: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> List[TestCase]:
    """
    Generate test cases for all single degradations at all severity levels.

    Args:
        degradation_types: List of degradation types to include (default: all)
        severity_levels: List of severity levels to include (default: 1-5)
        seed: Random seed for reproducibility

    Returns:
        List of TestCase objects
    """
    if degradation_types is None:
        # Use canonical names only (not aliases)
        degradation_types = list(SEVERITY_CONFIGS.keys())

    if severity_levels is None:
        severity_levels = [1, 2, 3, 4, 5]

    cases = []

    for deg_type in degradation_types:
        for severity in severity_levels:
            pipeline = DegradationPipeline(
                [(deg_type, severity)],
                seed=seed
            )
            cases.append(TestCase(
                pipeline=pipeline,
                name=f"{deg_type}_s{severity}",
                degradation_type="single",
                severity=severity
            ))

    return cases


def generate_combination_cases(
    combinations: List[List[Tuple[str, int]]],
    seed: Optional[int] = None
) -> List[TestCase]:
    """
    Generate test cases for specific degradation combinations.

    Args:
        combinations: List of combinations, where each combination is a list
                     of (degradation_type, severity) tuples
        seed: Random seed for reproducibility

    Returns:
        List of TestCase objects
    """
    cases = []

    for combo in combinations:
        pipeline = DegradationPipeline(combo, seed=seed)
        name = "_".join(f"{d}{s}" for d, s in combo)
        cases.append(TestCase(
            pipeline=pipeline,
            name=name,
            degradation_type="combination"
        ))

    return cases


def generate_test_matrix(
    include_clean: bool = True,
    include_single: bool = True,
    include_combinations: bool = True,
    degradation_types: Optional[List[str]] = None,
    severity_levels: Optional[List[int]] = None,
    combinations: Optional[List[List[Tuple[str, int]]]] = None,
    seed: Optional[int] = None
) -> List[TestCase]:
    """
    Generate a comprehensive test matrix of degradation cases.

    Args:
        include_clean: Include clean (no degradation) case
        include_single: Include all single degradations at all severity levels
        include_combinations: Include predefined combinations
        degradation_types: List of degradation types for single cases
        severity_levels: List of severity levels for single cases
        combinations: Custom combinations to include. If None and include_combinations
                     is True, uses default combinations.
        seed: Random seed for reproducibility

    Returns:
        List of TestCase objects

    Example:
        >>> cases = generate_test_matrix(
        ...     include_combinations=True,
        ...     combinations=[
        ...         [("lighting", 3), ("blur", 2)],
        ...         [("noise", 2), ("jpeg", 3)],
        ...     ]
        ... )
    """
    cases = []

    # Clean case (no degradation)
    if include_clean:
        cases.append(TestCase(
            pipeline=DegradationPipeline([], seed=seed),
            name="clean",
            degradation_type="clean"
        ))

    # Single degradation cases
    if include_single:
        cases.extend(generate_single_degradation_cases(
            degradation_types=degradation_types,
            severity_levels=severity_levels,
            seed=seed
        ))

    # Combination cases
    if include_combinations:
        if combinations is None:
            # Default realistic combinations
            combinations = [
                # Low light + blur (common in robotics)
                [("lighting", 2), ("gaussian_blur", 2)],
                [("lighting", 3), ("gaussian_blur", 3)],
                [("lighting", 4), ("gaussian_blur", 2)],

                # Motion + blur
                [("motion_blur", 2), ("gaussian_blur", 2)],
                [("motion_blur", 3), ("gaussian_blur", 2)],

                # Sensor noise + compression
                [("gaussian_noise", 2), ("jpeg_compression", 3)],
                [("gaussian_noise", 3), ("jpeg_compression", 2)],

                # Low light + noise (common sensor behavior)
                [("lighting", 3), ("gaussian_noise", 2)],
                [("lighting", 4), ("gaussian_noise", 3)],

                # Outdoor harsh conditions
                [("lighting", 4), ("motion_blur", 2), ("jpeg_compression", 3)],

                # Partial occlusion scenarios
                [("occlusion", 2), ("gaussian_blur", 1)],
                [("occlusion", 3), ("lighting", 2)],

                # Compressed stream + pixelation (low bandwidth)
                [("jpeg_compression", 3), ("pixelation", 2)],
                [("jpeg_compression", 4), ("pixelation", 3)],
            ]

        cases.extend(generate_combination_cases(combinations, seed=seed))

    return cases


def generate_severity_sweep(
    degradation_type: str,
    seed: Optional[int] = None
) -> List[TestCase]:
    """
    Generate test cases sweeping all severity levels for a single degradation.

    Useful for analyzing how a specific degradation affects model uncertainty
    as severity increases.

    Args:
        degradation_type: Type of degradation to sweep
        seed: Random seed

    Returns:
        List of TestCase objects for severity 1-5
    """
    return generate_single_degradation_cases(
        degradation_types=[degradation_type],
        severity_levels=[1, 2, 3, 4, 5],
        seed=seed
    )


def iterate_degraded_images(
    image: np.ndarray,
    test_cases: List[TestCase]
) -> Iterator[Tuple[np.ndarray, TestCase]]:
    """
    Iterate over degraded versions of an image.

    Args:
        image: Input image
        test_cases: List of test cases to apply

    Yields:
        Tuples of (degraded_image, test_case)
    """
    for case in test_cases:
        degraded = case.apply(image)
        yield degraded, case


def get_test_matrix_summary(cases: List[TestCase]) -> Dict:
    """
    Get summary statistics for a test matrix.

    Args:
        cases: List of test cases

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_cases": len(cases),
        "clean_cases": sum(1 for c in cases if c.degradation_type == "clean"),
        "single_cases": sum(1 for c in cases if c.degradation_type == "single"),
        "combination_cases": sum(1 for c in cases if c.degradation_type == "combination"),
        "degradation_types": {},
    }

    # Count by degradation type
    for case in cases:
        if case.degradation_type == "single":
            deg_type = case.pipeline.steps[0].degradation_type
            if deg_type not in summary["degradation_types"]:
                summary["degradation_types"][deg_type] = 0
            summary["degradation_types"][deg_type] += 1

    return summary
