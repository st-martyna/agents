#!/usr/bin/env python3
"""
Compare edge vs cloud model uncertainty calibration.

Analyzes benchmark results to compare edge-deployable models against
their cloud counterparts within the same model family.

Usage:
    python scripts/compare_edge_cloud.py \
        --results_dir ./results/run_20240101_120000 \
        --output ./comparison_report.html
"""

import sys
from pathlib import Path

import click
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pareto import (
    plot_edge_vs_cloud_comparison,
    plot_pareto_frontier,
    compute_scaling_efficiency,
    generate_pareto_report,
)
from src.analysis.failure_modes import compare_failure_rates


@click.command()
@click.option('--results_dir', '-r', required=True,
              type=click.Path(exists=True),
              help='Path to benchmark results directory')
@click.option('--output', '-o', required=True,
              type=click.Path(),
              help='Output path for comparison report')
@click.option('--format', '-f', default='html',
              type=click.Choice(['html', 'pdf', 'markdown']),
              help='Output format')
def main(results_dir: str, output: str, format: str):
    """Compare edge vs cloud model uncertainty calibration."""

    results_path = Path(results_dir)
    output_path = Path(output)

    # Load summary results
    summary_path = results_path / "summary.csv"
    if not summary_path.exists():
        print(f"Error: summary.csv not found in {results_dir}")
        return

    summary_df = pd.read_csv(summary_path)

    # Load detailed results
    all_results_path = results_path / "all_results.csv"
    if all_results_path.exists():
        all_results_df = pd.read_csv(all_results_path)
    else:
        all_results_df = None

    print(f"Loaded {len(summary_df)} model summaries")

    # Create figures directory
    figs_dir = output_path.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Generate Pareto frontier plot
    print("Generating Pareto frontier plot...")
    if 'params' in summary_df.columns and 'uncertainty_auroc' in summary_df.columns:
        fig = plot_pareto_frontier(
            summary_df,
            x='params',
            y='uncertainty_auroc',
            model_col='model',
            tier_col='tier' if 'tier' in summary_df.columns else None,
            title='Compute vs. Uncertainty Quality Tradeoff'
        )
        fig.savefig(figs_dir / "pareto_frontier.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Generate edge vs cloud comparison
    print("Generating edge vs cloud comparison...")
    fig = plot_edge_vs_cloud_comparison(
        summary_df,
        metric_col='uncertainty_auroc'
    )
    fig.savefig(figs_dir / "edge_vs_cloud.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Compute scaling efficiency
    print("Computing scaling efficiency...")
    scaling = compute_scaling_efficiency(
        summary_df,
        param_col='params',
        quality_col='uncertainty_auroc'
    )

    # Compare failure rates
    print("Comparing failure rates...")
    if all_results_df is not None:
        failure_comparison = compare_failure_rates(
            all_results_df,
            model_col='model',
            confidence_thresholds=[0.5, 0.7, 0.9, 0.95]
        )
    else:
        failure_comparison = None

    # Generate comprehensive report
    print("Generating comprehensive report...")
    report = generate_pareto_report(
        summary_df,
        metrics=['uncertainty_auroc', 'calibrated_ece'],
        cost_metrics=['params']
    )

    # Generate output
    if format == 'html':
        generate_html_report(
            output_path,
            summary_df,
            scaling,
            failure_comparison,
            report,
            figs_dir
        )
    elif format == 'markdown':
        generate_markdown_report(
            output_path,
            summary_df,
            scaling,
            failure_comparison,
            report
        )
    else:
        print(f"Format {format} not yet implemented")

    print(f"\nReport saved to: {output_path}")


def generate_html_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    scaling: dict,
    failure_comparison: pd.DataFrame,
    report: dict,
    figs_dir: Path
):
    """Generate HTML comparison report."""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Edge vs Cloud Model Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .highlight {{ background-color: #ffffcc; }}
        .recommendation {{ background-color: #e7f3ff; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .metric-good {{ color: green; font-weight: bold; }}
        .metric-bad {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Edge vs Cloud Model Uncertainty Comparison</h1>

    <h2>Executive Summary</h2>
    <div class="recommendation">
        <strong>Key Finding:</strong> {scaling.get('interpretation', 'Analysis pending')}
        <br><br>
        <strong>Scaling Slope:</strong> {scaling.get('slope', 0):.4f}
        (R² = {scaling.get('r_squared', 0):.3f})
    </div>

    <h2>Model Performance Summary</h2>
    {summary_df.to_html(classes='summary-table', index=False)}

    <h2>Pareto Frontier Analysis</h2>
    <p>The Pareto frontier shows models that are optimal for their compute budget.</p>
    <img src="{figs_dir / 'pareto_frontier.png'}" alt="Pareto Frontier">

    <h2>Edge vs Cloud Comparison by Model Family</h2>
    <img src="{figs_dir / 'edge_vs_cloud.png'}" alt="Edge vs Cloud Comparison">

    <h2>Recommendations</h2>
    <ul>
"""

    for rec in report.get('recommendations', []):
        html += f"        <li>{rec}</li>\n"

    html += """
    </ul>

    <h2>Methodology</h2>
    <p>This comparison evaluates uncertainty calibration using:</p>
    <ul>
        <li><strong>AUROC for Uncertainty:</strong> Can uncertainty predict model errors?</li>
        <li><strong>ECE (Expected Calibration Error):</strong> How well does confidence match accuracy?</li>
        <li><strong>MCE (Maximum Calibration Error):</strong> Worst-case calibration gap</li>
    </ul>

</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)


def generate_markdown_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    scaling: dict,
    failure_comparison: pd.DataFrame,
    report: dict
):
    """Generate Markdown comparison report."""

    md = f"""# Edge vs Cloud Model Uncertainty Comparison

## Executive Summary

**Key Finding:** {scaling.get('interpretation', 'Analysis pending')}

- Scaling Slope: {scaling.get('slope', 0):.4f}
- R²: {scaling.get('r_squared', 0):.3f}

## Model Performance Summary

{summary_df.to_markdown(index=False)}

## Recommendations

"""

    for rec in report.get('recommendations', []):
        md += f"- {rec}\n"

    md += """

## Methodology

This comparison evaluates uncertainty calibration using:

- **AUROC for Uncertainty:** Can uncertainty predict model errors?
- **ECE (Expected Calibration Error):** How well does confidence match accuracy?
- **MCE (Maximum Calibration Error):** Worst-case calibration gap
"""

    with open(output_path, 'w') as f:
        f.write(md)


if __name__ == '__main__':
    main()
