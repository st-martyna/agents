#!/usr/bin/env python3
"""
Generate comprehensive benchmark report with visualizations.

Usage:
    python scripts/generate_report.py \
        --results_dir ./results/run_20240101_120000 \
        --output ./report
"""

import sys
import json
from pathlib import Path

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    reliability_diagram,
    plot_calibration_comparison,
    compute_calibration_by_degradation,
)
from src.analysis import (
    find_high_confidence_failures,
    cluster_failures_by_degradation,
    generate_failure_report,
    plot_pareto_frontier,
    plot_edge_vs_cloud_comparison,
    generate_pareto_report,
)


@click.command()
@click.option('--results_dir', '-r', required=True,
              type=click.Path(exists=True),
              help='Path to benchmark results directory')
@click.option('--output', '-o', required=True,
              type=click.Path(),
              help='Output directory for report')
def main(results_dir: str, output: str):
    """Generate comprehensive benchmark report."""

    results_path = Path(results_dir)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    figs_dir = output_path / "figures"
    figs_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading benchmark results...")

    summary_df = pd.read_csv(results_path / "summary.csv")
    print(f"  Loaded {len(summary_df)} model summaries")

    all_results_path = results_path / "all_results.csv"
    if all_results_path.exists():
        results_df = pd.read_csv(all_results_path)
        print(f"  Loaded {len(results_df)} evaluation results")
    else:
        results_df = None

    config_path = results_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Generate figures
    print("\nGenerating visualizations...")

    # 1. Overall calibration comparison
    if results_df is not None:
        print("  - Calibration comparison across models")
        model_calibrations = {}
        for model in results_df['model'].unique():
            model_df = results_df[results_df['model'] == model]
            model_calibrations[model] = (
                model_df['calibrated_confidence'].values,
                model_df['is_correct'].values
            )

        if model_calibrations:
            fig = plot_calibration_comparison(
                model_calibrations,
                title="Calibration Comparison Across Models"
            )
            fig.savefig(figs_dir / "calibration_comparison.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    # 2. Pareto frontier
    print("  - Pareto frontier (params vs AUROC)")
    if 'params' in summary_df.columns and 'uncertainty_auroc' in summary_df.columns:
        fig = plot_pareto_frontier(
            summary_df,
            x='params',
            y='uncertainty_auroc',
            title="Compute vs. Uncertainty Quality"
        )
        fig.savefig(figs_dir / "pareto_frontier.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 3. Edge vs cloud comparison
    print("  - Edge vs cloud comparison")
    if 'tier' in summary_df.columns:
        fig = plot_edge_vs_cloud_comparison(summary_df)
        fig.savefig(figs_dir / "edge_vs_cloud.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 4. Calibration by degradation type
    if results_df is not None and 'degradation_type' in results_df.columns:
        print("  - Calibration by degradation")
        for model in results_df['model'].unique():
            model_df = results_df[results_df['model'] == model]

            # Create degradation heatmap
            pivot_data = model_df.groupby(
                ['degradation_type', 'severity']
            ).agg({
                'is_correct': 'mean',
                'calibrated_confidence': 'mean'
            }).reset_index()

            if len(pivot_data) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Accuracy heatmap
                acc_pivot = pivot_data.pivot(
                    index='degradation_type',
                    columns='severity',
                    values='is_correct'
                )
                if not acc_pivot.empty:
                    sns.heatmap(acc_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                               ax=ax1, vmin=0, vmax=1)
                    ax1.set_title(f'{model}: Accuracy by Degradation')

                # Confidence heatmap
                conf_pivot = pivot_data.pivot(
                    index='degradation_type',
                    columns='severity',
                    values='calibrated_confidence'
                )
                if not conf_pivot.empty:
                    sns.heatmap(conf_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                               ax=ax2, vmin=0, vmax=1)
                    ax2.set_title(f'{model}: Confidence by Degradation')

                plt.tight_layout()
                fig.savefig(figs_dir / f"{model}_degradation_heatmap.png", dpi=150)
                plt.close(fig)

    # 5. Failure analysis
    if results_df is not None:
        print("  - Failure analysis")
        failures = find_high_confidence_failures(
            results_df,
            confidence_threshold=0.9,
            confidence_col='calibrated_confidence'
        )

        if len(failures) > 0:
            failure_report = generate_failure_report(
                failures,
                total_samples=len(results_df)
            )

            with open(output_path / "failure_report.json", 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_types(v) for v in obj]
                    return obj

                json.dump(convert_types(failure_report), f, indent=2)

    # 6. Summary statistics table
    print("  - Summary statistics")

    # Create a nice summary table
    summary_table = summary_df[[
        'model', 'tier', 'params', 'accuracy',
        'calibrated_ece', 'uncertainty_auroc', 'mean_inference_ms'
    ]].copy()

    summary_table['params'] = summary_table['params'].apply(lambda x: f"{x/1e6:.1f}M")
    summary_table['accuracy'] = summary_table['accuracy'].apply(lambda x: f"{x:.1%}")
    summary_table['calibrated_ece'] = summary_table['calibrated_ece'].apply(lambda x: f"{x:.3f}")
    summary_table['uncertainty_auroc'] = summary_table['uncertainty_auroc'].apply(lambda x: f"{x:.3f}")
    summary_table['mean_inference_ms'] = summary_table['mean_inference_ms'].apply(lambda x: f"{x:.1f}")

    summary_table.columns = ['Model', 'Tier', 'Params', 'Accuracy', 'ECE', 'AUROC', 'Inference (ms)']

    # Generate Pareto report
    pareto_report = generate_pareto_report(summary_df)

    # Write markdown report
    print("\nWriting report...")

    report_md = f"""# VL-Uncertainty-Benchmark Report

## Configuration

- **Dataset:** {config.get('dataset', 'N/A')}
- **Models tested:** {config.get('n_models', len(summary_df))}
- **Test cases:** {config.get('n_test_cases', 'N/A')}
- **Samples:** {config.get('n_samples', 'N/A')}

## Key Findings

### Does scaling improve uncertainty calibration?

{pareto_report['scaling_analysis'].get('uncertainty_auroc', {}).get('interpretation', 'Analysis pending')}

### Recommendations

"""

    for rec in pareto_report.get('recommendations', []):
        report_md += f"- {rec}\n"

    report_md += f"""

## Model Performance Summary

{summary_table.to_markdown(index=False)}

## Visualizations

### Pareto Frontier

![Pareto Frontier](figures/pareto_frontier.png)

### Edge vs Cloud Comparison

![Edge vs Cloud](figures/edge_vs_cloud.png)

### Calibration Comparison

![Calibration Comparison](figures/calibration_comparison.png)

## Failure Analysis

High-confidence failures (>90% confidence but incorrect) are particularly
dangerous for robotics applications.

See `failure_report.json` for detailed failure analysis.

## Methodology

This benchmark evaluates uncertainty calibration using:

1. **Expected Calibration Error (ECE):** Measures average gap between confidence and accuracy
2. **AUROC for Uncertainty:** Measures how well uncertainty predicts errors
3. **Reliability Diagrams:** Visual comparison of confidence vs accuracy

Models are evaluated across 7 degradation types at 5 severity levels:
- Lighting changes (gamma correction)
- Gaussian blur
- Gaussian noise
- Motion blur
- Occlusion (random patches)
- JPEG compression artifacts
- Pixelation

---

*Generated by VL-Uncertainty-Benchmark*
"""

    with open(output_path / "report.md", 'w') as f:
        f.write(report_md)

    # Also save summary CSV
    summary_df.to_csv(output_path / "summary.csv", index=False)

    print(f"\nReport generated: {output_path}")
    print(f"  - report.md: Main report")
    print(f"  - figures/: Visualizations")
    print(f"  - failure_report.json: Failure analysis")
    print(f"  - summary.csv: Summary statistics")


if __name__ == '__main__':
    main()
