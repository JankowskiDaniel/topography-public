# Synthetic Topography Dataset Generation

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive framework for generating synthetic topography datasets and training neural networks to predict epsilon values from interference pattern images. This repository implements multiple noise generation techniques to create realistic synthetic datasets that closely match real-world topographic measurements.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Generation](#dataset-generation)
- [Project Structure](#project-structure)
- [Neural Network Training](#neural-network-training)
- [Analytical Methods](#analytical-methods)
- [Examples](#examples)
- [License](#license)

## 🔬 Overview

This project addresses the challenge of epsilon value prediction in topographic interference patterns through synthetic data generation and deep learning. The framework generates realistic synthetic topography images using mathematical models combined with various noise types, then trains convolutional neural networks (ResNet-based) to predict epsilon values.

### Key Components

- **Pure Image Generation**: Mathematical interference pattern generation based on epsilon values
- **Noise Controllers**: Seven different noise types to enhance realism
- **Dataset Generator**: Unified interface for creating large-scale datasets
- **Neural Network Training**: ResNet-based models for epsilon prediction
- **Analytical Methods**: Exact method implementations for comparison

### Scientific Context

Topographic measurements using interference patterns require accurate epsilon value determination. This project provides tools to:
- Generate synthetic training data at scale
- Train models on steel and ceramic material patterns
- Compare ML predictions against analytical methods
- Evaluate model performance across different noise conditions

## ✨ Features

### Noise Generation Types

1. **Pure**: Mathematical interference patterns without noise
2. **Perlin**: Perlin noise with CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Average**: Real material-based averaged noise (steel/ceramic)
4. **Fourier**: Frequency/amplitude domain noise from real images
5. **Bubble**: Gaussian-distributed bubble artifacts
6. **Triangular**: Triangular brightness variations
7. **Blackbox**: Rectangular occlusions

### Data Generation

- Configurable epsilon ranges (0.0 to 1.0)
- Multiple copies per epsilon value for data augmentation
- Seed-based reproducibility
- Parallel processing support
- ZIP archive export option
- Comprehensive metadata tracking

### Model Training

- ResNet18/ResNet50 architectures
- Transfer learning from ImageNet
- Material-specific models (steel, ceramic)
- Cyclic Mean Absolute Error (CMAE) metric
- Training notebooks with visualization

## 🚀 Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for training)
- 10+ GB free disk space for datasets

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/topography-public.git
cd topography-public
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package** (optional, for cleaner imports)
```bash
pip install -e .
```

### Dependencies

Key packages:
- `torch>=2.2.2` - Deep learning framework
- `torchvision>=0.17.2` - Pre-trained models and transforms
- `opencv-python>=4.9.0` - Image processing
- `numpy>=1.26.3` - Numerical computations
- `pandas>=2.2.1` - Data manipulation
- `scikit-learn>=1.4.1` - Machine learning utilities
- `noise>=1.2.2` - Perlin noise generation
- `pydantic>=2.7.1` - Data validation

See [requirements.txt](requirements.txt) for complete list.

## 🎯 Quick Start

### Generate Your First Dataset

```bash
# Generate 100 images with Perlin noise
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/test \
    --epsilon-range 0.0 0.1 \
    --epsilon-step 0.01 \
    --n-copies 10 \
    --seed 23
```

### Use as Python Library

```python
from src.data_generation.datasets.generator import generate_dataset

# Generate dataset with Perlin noise
generate_dataset(
    noise_type="perlin",
    path="data/generated/my_dataset",
    n_copies=50,
    name_prefix="img",
    epsilon_range=(0.0, 1.0),
    epsilon_step=0.001,
    size=(640, 480),
    seed=23,
    alpha=0.45,  # Perlin-specific parameter
)
```

### Train a Model

```python
# See notebooks/02_model_training/ for complete examples
import torch
from torchvision import models

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Modify for single-channel input
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Custom regression head
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Train on your dataset
# (see notebooks for complete training loops)
```

## 📊 Dataset Generation

### Command-Line Interface

The main dataset generation script provides a unified interface for all noise types:

```bash
python scripts/generate_dataset.py --help
```

### Basic Usage

#### Pure Images (No Noise)
```bash
python scripts/generate_dataset.py \
    --noise pure \
    --output data/generated/pure \
    --n-copies 50 \
    --seed 23
```

#### Perlin Noise
```bash
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/perlin \
    --n-copies 50 \
    --alpha 0.45 \
    --seed 23
```

#### Average Noise (Requires Pre-computed Noise Images)
```bash
python scripts/generate_dataset.py \
    --noise average \
    --output data/generated/average \
    --path-average-noise data/average_noise/steel \
    --n-copies 50 \
    --seed 23
```

#### Combined Noise Types
```bash
python scripts/generate_dataset.py \
    --noise perlin triangular \
    --output data/generated/combined \
    --n-copies 50 \
    --alpha 0.45 \
    --nr-of-triangulars 3 8 \
    --triangular-strength 20 30 \
    --seed 23
```

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--noise` | Noise type(s) to apply | Required |
| `--output` | Output directory path | Required |
| `--epsilon-range` | Min and max epsilon values | 0.0 1.0 |
| `--epsilon-step` | Step size for epsilon | 0.001 |
| `--n-copies` | Copies per epsilon value | 50 |
| `--size` | Image dimensions (W H) | 640 480 |
| `--brightness` | Brightness range (min max) | 80 210 |
| `--seed` | Random seed | None |
| `--zipfile` | Save to ZIP archive | False |

### Noise-Specific Parameters

#### Perlin Noise
```bash
--alpha 0.45                    # Blending weight (0.0-1.0)
--scale 100.0                   # Noise scale
--octaves 6                     # Number of octaves
--persistence-range 0.7 0.9     # Persistence range
--lacunarity 2.0                # Lacunarity parameter
--clip-limit-range 1.0 2.0      # CLAHE clip limit
```

#### Bubble Noise
```bash
--spray-particles 800           # Number of particles
--spray-diameter 8              # Spray diameter
--range-of-blobs 30 40          # Number of blobs
```

#### Triangular Noise
```bash
--nr-of-triangulars 3 8              # Number of segments
--triangular-strength 10 15          # Brightness change
```

#### Fourier Noise
```bash
--fourier-domain ampl           # Domain: ampl or freq
--path-fourier-noise-ampl PATH  # Amplitude noise path
--path-fourier-noise-freq PATH  # Frequency noise path
--noise-proportion 0.5          # Noise proportion
```

### Parallel Processing

For large datasets, split generation across multiple processes:

```bash
# See scripts/example_parallel_generation.sh for complete example

# Process 1
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/parallel \
    --epsilon-range 0.0 0.25 \
    --n-copies 50 \
    --name-prefix p1 \
    --seed 23 &

# Process 2
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/parallel \
    --epsilon-range 0.25 0.5 \
    --n-copies 50 \
    --name-prefix p2 \
    --seed 43 &

wait
```

### Output Format

Generated datasets include:

1. **Images**: PNG files named with specified prefix (e.g., `img_00000.png`)
2. **Metadata CSV** (`parameters.csv`):
   - `filename`: Image filename
   - `width`, `height`: Image dimensions
   - `epsilon`: Ground truth epsilon value
   - `ring_center_width`, `ring_center_height`: Ring center coordinates
   - `min_brightness`, `max_brightness`: Brightness range
   - `used_noise`: Noise index (for average noise)

## 📁 Project Structure

```
topography-public/
├── src/                              # Source code
│   ├── data_generation/              # Data generation modules
│   │   ├── image/                    # Pure image generation
│   │   │   ├── image_interface.py    # Abstract generator interface
│   │   │   └── image_generator.py    # Pure image implementation
│   │   ├── noise_controllers/        # Noise implementations
│   │   │   ├── decorator.py          # Noise controller interface
│   │   │   ├── builder.py            # Controller factory
│   │   │   ├── noise_perlin.py       # Perlin noise
│   │   │   ├── noise_average.py      # Average noise
│   │   │   ├── noise_fourier.py      # Fourier noise
│   │   │   ├── noise_bubble.py       # Bubble noise
│   │   │   ├── noise_triangular.py        # Triangular noise
│   │   │   └── noise_blackbox.py     # Blackbox noise
│   │   └── datasets/                 # Dataset generation
│   │       ├── generator.py          # Main generator
│   │       ├── generate_utils.py     # Utilities
│   │       └── generator_noise_*.py  # Specialized generators
│   ├── data_preparation/             # Data utilities
│   │   └── preparation_utils.py      # CSV, ZIP utilities
│   ├── methods/                      # Analytical methods
│   │   └── analytic/
│   │       └── exact.py              # Exact epsilon calculation
│   └── models/                       # Data models
│       └── image_models.py           # Pydantic models
│
├── scripts/                          # Generation scripts
│   ├── generate_dataset.py           # Main CLI script
│   ├── test_perlin_integration.py    # Integration test
│   ├── example_perlin_generation.sh  # Example: Perlin
│   ├── example_parallel_generation.sh # Example: Parallel
│   └── example_combined_noise.sh     # Example: Combined
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01-dataset_generation.ipynb   # Dataset generation examples
│   ├── 02-model_training.ipynb       # Model training
│   ├── 03-feature_maps_resnet50.ipynb # Feature visualization
│   ├── 10-exact-method.ipynb         # Analytical method
│   ├── 11-raw-images-inference.ipynb # Raw data inference
│   └── steel/                        # Steel-specific experiments
│       ├── 03-perlin-training.ipynb  # Perlin noise training
│       ├── 08-models-results.ipynb   # Results analysis
│       ├── 09-final-model-testing.ipynb # Final evaluation
│       └── utils.py                  # Plotting utilities
│
├── data/                             # Data directory (not in repo)
│   ├── average_noise/                # Pre-computed average noise
│   │   ├── ceramic/
│   │   └── steel/
│   ├── generated/                    # Generated datasets
│   └── results/                      # Model predictions
│
├── tests/                            # Unit tests
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
└── README.md                         # This file
```

## 🧠 Neural Network Training

### Architecture

Models use **ResNet18** or **ResNet50** as backbone:
- Modified first convolutional layer for single-channel grayscale input
- Custom regression head (fully connected layer outputting single epsilon value)
- Transfer learning from ImageNet pre-trained weights

### Training Process

1. **Data Preparation**
   ```python
   from torch.utils.data import Dataset, DataLoader
   import torchvision.transforms as transforms

   transform = transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5], std=[0.5])
   ])
   ```

2. **Loss Function**
   - Cyclic Mean Absolute Error (CMAE) accounts for epsilon's cyclic nature [0, 1]
   - Standard: `torch.nn.BCEWithLogitsLoss()` or custom CMAE implementation

3. **Training Loop**
   - See `notebooks/02_model_training/` for complete examples
   - Typical: 10-50 epochs, learning rate 1e-4, batch size 32

4. **Evaluation**
   - Compare against analytical exact method
   - Metrics: CMAE, MAE, distribution plots
   - See `notebooks/03_evaluation/` for analysis

### Model Variations

- **Steel models**: Trained on steel material patterns
- **Ceramic models**: Trained on ceramic material patterns
- **General models**: Trained on mixed material data
- **Noise-specific models**: Optimized for specific noise types

Training notebooks available in `notebooks/02_model_training/`:
- `06-perlin-training.ipynb`
- `01-steel-scratch-training.ipynb`
- `08-ceramic-model-fine-tuning.ipynb`

## 🔍 Analytical Methods

The repository includes an exact analytical method for epsilon calculation based on peak detection and linear regression.

### Algorithm Overview

1. **Circle Detection**: Locate ring center using Hough circles or peak symmetry
2. **Peak Detection**: Find brightness maxima (ring locations)
3. **Symmetry Analysis**: Identify consistent ring spacing patterns
4. **Linear Regression**: Fit line to distance² vs ring number
5. **Epsilon Calculation**: Compute from regression parameters

### Usage

```python
from src.methods.analytic.exact import AnalyticalMathodNew

# Initialize method
method = AnalyticalMathodNew()

# Load image
import cv2
img = cv2.imread("path/to/image.png", cv2.IMREAD_GRAYSCALE)

# Calculate epsilon
epsilon = method.calculate_epsilon(img)
print(f"Predicted epsilon: {epsilon:.3f}")
```

See `notebooks/10-exact-method.ipynb` for detailed examples.

## 📚 Examples

### Example 1: Generate Test Dataset

```bash
# 1000 images for quick testing
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/test \
    --epsilon-range 0.0 1.0 \
    --epsilon-step 0.01 \
    --n-copies 10 \
    --seed 23
```

### Example 2: Full Production Dataset

```bash
# 50,000 images for training
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/steel_perlin \
    --epsilon-range 0.0 1.0 \
    --epsilon-step 0.001 \
    --n-copies 50 \
    --alpha 0.45 \
    --scale 100.0 \
    --seed 23
```

### Example 3: Multi-Noise Realistic Dataset

```bash
# Combined noise for maximum realism
python scripts/generate_dataset.py \
    --noise perlin triangular bubble \
    --output data/generated/realistic \
    --n-copies 50 \
    --alpha 0.45 \
    --nr-of-triangulars 3 5 \
    --triangular-strength 20 30 \
    --range-of-blobs 30 40 \
    --seed 23
```

### Example 4: Use Pre-built Scripts

```bash
# Run example generation script
bash scripts/example_perlin_generation.sh

# Run parallel generation
bash scripts/example_parallel_generation.sh

# Run combined noise generation
bash scripts/example_combined_noise.sh
```