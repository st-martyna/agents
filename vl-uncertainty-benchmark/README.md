# VL-Uncertainty-Benchmark

A systematic benchmarking framework for vision model uncertainty calibration under degraded conditions. Designed for robotics middleware that needs to determine when to switch from reactive to deliberative control based on perception uncertainty.

## Overview

This benchmark evaluates **20 vision models** across 4 categories:
- **Detection**: SAM, SAM2-Large, YOLO-World-S, YOLO-World-X
- **Self-Supervised**: DINOv2-B, DINOv2-g, V-JEPA 2
- **Vision-Language Models (VLMs)**: Florence-2, PaliGemma 2, Qwen2.5-VL, LLaVA-OneVision, InternVL2.5
- **Vision-Language-Action (VLAs)**: Octo, SmolVLA, OpenVLA

### Core Research Question

> Does scaling up models give meaningfully better uncertainty calibration, or are edge models sufficient for reactive→deliberative mode switching?

## Features

- **Degradation Pipeline**: 7 degradation types with 5 severity levels each
  - Lighting (gamma), Gaussian blur, Gaussian noise, Motion blur
  - Occlusion (random patches), JPEG compression, Pixelation

- **Uncertainty Extraction**: Model-specific uncertainty metrics
  - Token entropy (VLMs/VLAs)
  - Embedding distance to centroids (self-supervised)
  - Diffusion/flow variance (action models)
  - IoU prediction (SAM), detection confidence (YOLO)

- **Calibration Methods**: Temperature scaling, Platt scaling

- **Metrics**: ECE, MCE, Brier score, reliability diagrams, AUROC for error prediction

- **Analysis**: Failure mode clustering, Pareto frontier (compute vs. uncertainty quality)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vl-uncertainty-benchmark.git
cd vl-uncertainty-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Hardware Requirements

- **Edge tier** (`edge`): Models run on Jetson Nano 8GB equivalent (~4GB VRAM)
- **Edge+ tier** (`edge_plus`): Models run on Jetson Orin 64GB equivalent
- **Cloud tier** (`cloud`): Full-precision models requiring high-end GPUs (A100/H100)

## Quick Start

### Run a basic benchmark

```bash
# Benchmark edge-deployable models on a dataset
python scripts/run_benchmark.py \
    --models florence2_base sam \
    --tier edge \
    --dataset /path/to/images \
    --degradations blur noise lighting \
    --output_dir ./results

# Run all models on all degradations
python scripts/run_benchmark.py \
    --models all \
    --tier all \
    --dataset /path/to/images \
    --degradations all \
    --output_dir ./results
```

### Compare edge vs cloud models

```bash
python scripts/compare_edge_cloud.py \
    --results_dir ./results \
    --output ./comparison_report.html
```

### Generate a full report

```bash
python scripts/generate_report.py \
    --results_dir ./results \
    --output ./report
```

## Usage Examples

### Using the degradation pipeline

```python
from src.degradation.transforms import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_lighting_change,
)
from src.degradation.pipeline import DegradationPipeline, generate_test_matrix

# Single degradation
degraded = apply_gaussian_blur(image, severity=3)  # severity 1-5

# Composable pipeline
pipeline = DegradationPipeline([
    ("lighting", 2),
    ("blur", 3),
])
degraded = pipeline(image)

# Generate full test matrix
test_cases = generate_test_matrix(
    include_combinations=True,
    combinations=[("lighting", "blur"), ("noise", "motion_blur")]
)
```

### Using model wrappers

```python
from src.models.vlm.florence import Florence2Base

# Initialize and load model
model = Florence2Base(deployment_tier="edge")
model.load()

# Get prediction with confidence
prediction, confidence = model.predict(image)

# Extract uncertainty metrics
uncertainty = model.extract_uncertainty(image)
# {'token_entropy': 0.23, 'sequence_entropy': 0.45}

# Calibrate on validation set
model.calibrate(val_images, val_labels)

# Get model info
info = model.get_info()
# {'params': 232_000_000, 'license': 'MIT', 'category': 'vlm', ...}
```

### Computing calibration metrics

```python
from src.metrics.calibration import (
    expected_calibration_error,
    reliability_diagram,
    auroc_uncertainty,
)

# Compute ECE
ece = expected_calibration_error(confidences, accuracies, n_bins=15)

# Generate reliability diagram
fig = reliability_diagram(confidences, accuracies, n_bins=15)
fig.savefig("reliability.png")

# Can uncertainty predict errors?
auroc = auroc_uncertainty(uncertainties, is_correct)
```

### Failure mode analysis

```python
from src.analysis.failure_modes import (
    find_high_confidence_failures,
    cluster_failures_by_degradation,
    generate_failure_report,
)

# Find confident but wrong predictions
failures = find_high_confidence_failures(results_df, confidence_threshold=0.9)

# Cluster by degradation type
clusters = cluster_failures_by_degradation(failures)

# Generate summary report
report = generate_failure_report(failures)
```

## Project Structure

```
vl-uncertainty-benchmark/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/
│   └── models.yaml          # Model configs, paths, deployment tiers
├── src/
│   ├── degradation/         # Image degradation transforms
│   ├── models/              # Model wrappers (detection, VLM, VLA, etc.)
│   ├── uncertainty/         # Uncertainty extraction and calibration
│   ├── metrics/             # ECE, MCE, Brier score, etc.
│   └── analysis/            # Failure modes, Pareto analysis
├── scripts/
│   ├── run_benchmark.py     # Main benchmark runner
│   ├── compare_edge_cloud.py
│   └── generate_report.py
├── notebooks/
│   └── analysis.ipynb
└── tests/
    └── test_degradation.py
```

## Model Registry

| Model | Category | Tier | Params | Uncertainty Method | License |
|-------|----------|------|--------|-------------------|---------|
| SAM | detection | edge | 94M | IoU prediction | Apache 2.0 |
| SAM2-Large | detection | cloud | 224M | IoU + occlusion | Apache 2.0 |
| YOLO-World-S | detection | edge | 13M | obj * class prob | GPL-3.0 |
| YOLO-World-X | detection | cloud | 97M | obj * class prob | GPL-3.0 |
| DINOv2-B | self_supervised | edge | 86M | embedding distance | Apache 2.0 |
| DINOv2-g | self_supervised | cloud | 1.1B | embedding distance | Apache 2.0 |
| V-JEPA 2 | self_supervised | cloud | 1.2B | latent variance | MIT |
| Florence-2-base | vlm | edge | 232M | token entropy | MIT |
| Florence-2-large | vlm | cloud | 770M | token entropy | MIT |
| PaliGemma 2-3B | vlm | edge | 3B | location confidence | Gemma |
| PaliGemma 2-28B | vlm | cloud | 28B | location confidence | Gemma |
| Qwen2.5-VL-3B | vlm | edge | 3B | token entropy | Apache 2.0 |
| Qwen2.5-VL-72B | vlm | cloud | 72B | token entropy | Apache 2.0 |
| LLaVA-OneVision-0.5B | vlm | edge | 0.5B | token entropy | Apache 2.0 |
| LLaVA-OneVision-72B | vlm | cloud | 72B | token entropy | Apache 2.0 |
| InternVL2.5-78B | vlm | cloud | 78B | token entropy | Apache 2.0 |
| Octo-Small | vla | edge | 27M | diffusion variance | Open |
| Octo-Base | vla | cloud | 93M | diffusion variance | Open |
| SmolVLA | vla | edge | 450M | flow variance | Open |
| OpenVLA | vla | cloud | 7B | token entropy | Llama |

## Interpreting Results

### Calibration Quality

- **ECE < 0.05**: Well-calibrated, suitable for direct threshold-based switching
- **ECE 0.05-0.15**: Moderate calibration, requires temperature scaling
- **ECE > 0.15**: Poor calibration, uncertainty signals unreliable

### AUROC for Error Prediction

- **AUROC > 0.8**: Uncertainty reliably predicts errors
- **AUROC 0.6-0.8**: Moderate predictive power
- **AUROC < 0.6**: Uncertainty not useful for error detection

### Reactive→Deliberative Switching

For robotics middleware:
1. Models with high AUROC can trigger deliberative mode when uncertainty exceeds threshold
2. Edge models with good calibration may suffice for real-time switching
3. Cloud models useful for deliberative mode verification

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Implement your changes with tests
4. Submit a pull request

## License

Apache 2.0

## Citation

```bibtex
@software{vl_uncertainty_benchmark,
  title={VL-Uncertainty-Benchmark: Vision Model Uncertainty Calibration for Robotics},
  year={2025},
  url={https://github.com/your-org/vl-uncertainty-benchmark}
}
```
