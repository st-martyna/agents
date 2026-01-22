#!/usr/bin/env python3
"""
Main benchmark runner for VL-Uncertainty-Benchmark.

Runs vision models on degraded images and collects uncertainty metrics.

Usage:
    python scripts/run_benchmark.py \
        --models florence2_base sam \
        --tier edge \
        --dataset /path/to/images \
        --degradations blur noise lighting \
        --output_dir ./results

    python scripts/run_benchmark.py \
        --models all \
        --tier all \
        --degradations all \
        --dataset /path/to/images \
        --output_dir ./results
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelWrapper, get_model, list_models, list_models_by_tier
from src.degradation import (
    DegradationPipeline,
    generate_test_matrix,
    generate_single_degradation_cases,
    TestCase,
)
from src.metrics import (
    expected_calibration_error,
    compute_calibration_metrics,
    reliability_diagram,
)
from src.uncertainty import TemperatureScaling


def load_dataset(dataset_path: str, max_images: Optional[int] = None) -> List[Dict]:
    """
    Load images from dataset directory.

    Expects either:
    - Directory of images
    - JSON file with image paths and labels

    Args:
        dataset_path: Path to dataset
        max_images: Maximum number of images to load

    Returns:
        List of dicts with 'image_path', 'image', 'label' keys
    """
    dataset_path = Path(dataset_path)
    samples = []

    if dataset_path.is_file() and dataset_path.suffix == '.json':
        # JSON manifest
        with open(dataset_path) as f:
            manifest = json.load(f)

        for item in manifest.get('images', manifest):
            img_path = item.get('path', item.get('image_path'))
            if img_path:
                samples.append({
                    'image_path': img_path,
                    'label': item.get('label', item.get('class', 'unknown')),
                })

    elif dataset_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for img_path in sorted(dataset_path.rglob('*')):
            if img_path.suffix.lower() in image_extensions:
                # Try to infer label from parent directory
                label = img_path.parent.name if img_path.parent != dataset_path else 'unknown'
                samples.append({
                    'image_path': str(img_path),
                    'label': label,
                })

    else:
        raise ValueError(f"Dataset path must be a directory or JSON file: {dataset_path}")

    if max_images:
        samples = samples[:max_images]

    # Load images
    for sample in samples:
        img = cv2.imread(sample['image_path'])
        if img is not None:
            sample['image'] = img
        else:
            print(f"Warning: Could not load {sample['image_path']}")
            sample['image'] = None

    # Filter out failed loads
    samples = [s for s in samples if s['image'] is not None]

    return samples


def run_single_evaluation(
    model: ModelWrapper,
    image: np.ndarray,
    test_case: TestCase,
    ground_truth: str
) -> Dict[str, Any]:
    """
    Run evaluation on a single image with a specific degradation.

    Args:
        model: Model wrapper
        image: Original image
        test_case: Degradation test case
        ground_truth: Ground truth label

    Returns:
        Dictionary with evaluation results
    """
    # Apply degradation
    degraded_image = test_case.apply(image)

    # Measure inference time
    start_time = time.time()

    try:
        # Get prediction with uncertainty
        prediction, confidence, uncertainty = model.predict_with_uncertainty(degraded_image)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Determine if prediction is correct
        # This depends on the model type and prediction format
        if isinstance(prediction, str):
            is_correct = ground_truth.lower() in prediction.lower()
        elif isinstance(prediction, dict):
            is_correct = prediction.get('label', '') == ground_truth
        else:
            is_correct = False  # Can't determine

        result = {
            'prediction': str(prediction)[:200],  # Truncate long outputs
            'raw_confidence': confidence,
            'uncertainty': uncertainty.get('token_entropy', uncertainty.get('iou_confidence', 0)),
            'uncertainty_full': uncertainty,
            'is_correct': int(is_correct),
            'inference_time_ms': inference_time,
            'error': None,
        }

    except Exception as e:
        result = {
            'prediction': None,
            'raw_confidence': 0.0,
            'uncertainty': 1.0,
            'uncertainty_full': {},
            'is_correct': 0,
            'inference_time_ms': 0,
            'error': str(e),
        }

    return result


@click.command()
@click.option('--models', '-m', multiple=True, default=['all'],
              help='Model names to benchmark (or "all")')
@click.option('--tier', '-t', default='all',
              type=click.Choice(['edge', 'edge_plus', 'cloud', 'all']),
              help='Deployment tier filter')
@click.option('--dataset', '-d', required=True,
              type=click.Path(exists=True),
              help='Path to dataset (directory or JSON)')
@click.option('--degradations', '-g', multiple=True, default=['all'],
              help='Degradation types (or "all")')
@click.option('--severities', '-s', multiple=True, default=[1, 2, 3, 4, 5],
              type=int, help='Severity levels to test')
@click.option('--output_dir', '-o', required=True,
              type=click.Path(),
              help='Output directory for results')
@click.option('--max_images', type=int, default=None,
              help='Maximum images to process')
@click.option('--calibrate/--no-calibrate', default=True,
              help='Apply temperature scaling calibration')
@click.option('--calibration_split', type=float, default=0.2,
              help='Fraction of data for calibration')
@click.option('--seed', type=int, default=42,
              help='Random seed')
def main(
    models: tuple,
    tier: str,
    dataset: str,
    degradations: tuple,
    severities: tuple,
    output_dir: str,
    max_images: Optional[int],
    calibrate: bool,
    calibration_split: float,
    seed: int
):
    """
    Run uncertainty calibration benchmark on vision models.

    This benchmark evaluates how well model confidence aligns with actual
    accuracy under various image degradations, which is critical for
    robotics applications that need to know when to switch from reactive
    to deliberative control.
    """
    np.random.seed(seed)

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Determine models to run
    if 'all' in models:
        if tier == 'all':
            model_names = list_models()
        else:
            model_names = list_models_by_tier(tier)
    else:
        model_names = list(models)

    print(f"Models to benchmark: {model_names}")

    # Generate test cases
    if 'all' in degradations:
        degradation_types = None  # Use all
    else:
        degradation_types = list(degradations)

    test_cases = generate_test_matrix(
        include_clean=True,
        include_single=True,
        include_combinations=True,
        degradation_types=degradation_types,
        severity_levels=list(severities),
        seed=seed
    )

    print(f"Test cases: {len(test_cases)}")

    # Load dataset
    print(f"Loading dataset from {dataset}...")
    samples = load_dataset(dataset, max_images)
    print(f"Loaded {len(samples)} images")

    if len(samples) == 0:
        print("Error: No images loaded")
        return

    # Split for calibration
    n_cal = int(len(samples) * calibration_split)
    cal_samples = samples[:n_cal]
    test_samples = samples[n_cal:]

    print(f"Calibration samples: {n_cal}, Test samples: {len(test_samples)}")

    # Results storage
    all_results = []

    # Run benchmark for each model
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")

        try:
            model = get_model(model_name)
            model.load()
            model_info = model.get_info()

            print(f"  Loaded: {model_info['params']:,} params, {model_info['deployment_tier']} tier")

        except NotImplementedError as e:
            print(f"  Skipping {model_name}: {e}")
            continue
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            continue

        # Calibration phase
        if calibrate and len(cal_samples) > 0:
            print("  Calibrating on validation set...")
            try:
                cal_images = [s['image'] for s in cal_samples]
                cal_labels = [s['label'] for s in cal_samples]
                temp = model.calibrate(cal_images, cal_labels)
                print(f"  Optimal temperature: {temp:.3f}")
            except Exception as e:
                print(f"  Calibration failed: {e}")

        # Evaluation phase
        print("  Running evaluation...")
        model_results = []

        for sample in tqdm(test_samples, desc=f"  {model_name}"):
            for test_case in test_cases:
                result = run_single_evaluation(
                    model,
                    sample['image'],
                    test_case,
                    sample['label']
                )

                result.update({
                    'model': model_name,
                    'tier': model_info['deployment_tier'],
                    'category': model_info['category'],
                    'params': model_info['params'],
                    'image_path': sample['image_path'],
                    'ground_truth': sample['label'],
                    'degradation_type': test_case.pipeline.steps[0].degradation_type if test_case.pipeline.steps else 'clean',
                    'severity': test_case.pipeline.steps[0].severity if test_case.pipeline.steps else 0,
                    'test_case_name': test_case.name,
                })

                # Add calibrated confidence
                result['calibrated_confidence'] = model.get_calibrated_confidence(
                    result['raw_confidence']
                )

                model_results.append(result)

        all_results.extend(model_results)

        # Unload model to free memory
        model.unload()

        # Save intermediate results
        df = pd.DataFrame(model_results)
        df.to_csv(run_dir / f"{model_name}_results.csv", index=False)

    # Save all results
    print("\nSaving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(run_dir / "all_results.csv", index=False)

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary = []

    for model_name in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model_name]

        # Raw calibration metrics
        raw_metrics = compute_calibration_metrics(
            model_df['raw_confidence'].values,
            model_df['is_correct'].values,
            model_df['uncertainty'].values
        )

        # Calibrated metrics
        cal_metrics = compute_calibration_metrics(
            model_df['calibrated_confidence'].values,
            model_df['is_correct'].values,
            model_df['uncertainty'].values
        )

        summary.append({
            'model': model_name,
            'tier': model_df['tier'].iloc[0],
            'category': model_df['category'].iloc[0],
            'params': model_df['params'].iloc[0],
            'n_samples': len(model_df),
            'accuracy': model_df['is_correct'].mean(),
            'mean_confidence': model_df['raw_confidence'].mean(),
            'raw_ece': raw_metrics['ece'],
            'raw_mce': raw_metrics['mce'],
            'calibrated_ece': cal_metrics['ece'],
            'calibrated_mce': cal_metrics['mce'],
            'uncertainty_auroc': raw_metrics.get('auroc_uncertainty', 0),
            'mean_inference_ms': model_df['inference_time_ms'].mean(),
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('uncertainty_auroc', ascending=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))

    # Generate reliability diagrams
    print("\nGenerating reliability diagrams...")
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    for model_name in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model_name]

        # Raw reliability diagram
        fig = reliability_diagram(
            model_df['raw_confidence'].values,
            model_df['is_correct'].values,
            title=f"{model_name} - Raw Confidence"
        )
        fig.savefig(figs_dir / f"{model_name}_reliability_raw.png", dpi=150)
        plt.close(fig)

        # Calibrated reliability diagram
        fig = reliability_diagram(
            model_df['calibrated_confidence'].values,
            model_df['is_correct'].values,
            title=f"{model_name} - Calibrated Confidence"
        )
        fig.savefig(figs_dir / f"{model_name}_reliability_calibrated.png", dpi=150)
        plt.close(fig)

    # Save run config
    config = {
        'models': list(models),
        'tier': tier,
        'dataset': dataset,
        'degradations': list(degradations),
        'severities': list(severities),
        'n_test_cases': len(test_cases),
        'n_samples': len(samples),
        'calibrate': calibrate,
        'seed': seed,
        'timestamp': timestamp,
    }

    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    print("\nKey findings:")

    # Answer the core research question
    if len(summary_df) > 0:
        edge_models = summary_df[summary_df['tier'] == 'edge']
        cloud_models = summary_df[summary_df['tier'] == 'cloud']

        if len(edge_models) > 0 and len(cloud_models) > 0:
            edge_auroc = edge_models['uncertainty_auroc'].mean()
            cloud_auroc = cloud_models['uncertainty_auroc'].mean()

            print(f"\n  Edge model avg AUROC: {edge_auroc:.3f}")
            print(f"  Cloud model avg AUROC: {cloud_auroc:.3f}")
            print(f"  Difference: {cloud_auroc - edge_auroc:.3f}")

            if cloud_auroc - edge_auroc < 0.05:
                print("\n  CONCLUSION: Edge models provide similar uncertainty quality.")
                print("  They may be sufficient for reactive→deliberative mode switching.")
            else:
                print("\n  CONCLUSION: Cloud models provide meaningfully better uncertainty.")
                print("  Consider using cloud models for critical decisions.")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
