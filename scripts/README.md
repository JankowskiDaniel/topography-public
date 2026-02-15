# Dataset Generation Scripts

This directory contains scripts for generating synthetic topography datasets with various noise types.

## Main Script: `generate_dataset.py`

A unified script that provides a simple interface to generate datasets with any combination of noise types.

### Basic Usage

```bash
# Show help
python scripts/generate_dataset.py --help

# Generate pure images (no noise)
python scripts/generate_dataset.py \
    --noise pure \
    --output data/generated/pure \
    --n-copies 50

# Generate with Perlin noise
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/perlin \
    --n-copies 50 \
    --alpha 0.45

# Generate with average noise (requires pre-computed average noise images)
python scripts/generate_dataset.py \
    --noise average \
    --output data/generated/average \
    --path-average-noise data/average_noise/steel \
    --n-copies 50

# Generate with multiple noise types (Perlin + Pizza)
python scripts/generate_dataset.py \
    --noise perlin pizza \
    --output data/generated/perlin_pizza \
    --n-copies 50 \
    --alpha 0.45 \
    --nr-of-pizzas 3 8 \
    --pizza-strength 20 30
```

### Available Noise Types

1. **pure** - No noise, just mathematical interference patterns
2. **average** - Uses averaged real topography images
3. **perlin** - Perlin noise with CLAHE enhancement
4. **bubble** - Gaussian bubble artifacts
5. **pizza** - Triangular brightness variations
6. **blackbox** - Rectangular occlusions
7. **fourier** - Frequency/amplitude domain noise

### Common Parameters

```
--output PATH           Output directory (required)
--noise TYPE [TYPE ...] Noise type(s) to apply (required)
--epsilon-range MIN MAX Epsilon value range (default: 0.0 1.0)
--epsilon-step STEP     Step size for epsilon (default: 0.001)
--n-copies N            Copies per epsilon value (default: 50)
--size WIDTH HEIGHT     Image dimensions (default: 640 480)
--seed SEED             Random seed for reproducibility
```

### Noise-Specific Parameters

#### Perlin Noise
```
--alpha FLOAT              Blending weight (default: 0.45)
--scale FLOAT              Noise scale (default: 100.0)
--octaves INT              Number of octaves (default: 6)
--persistence-range MIN MAX Persistence range (default: 0.7 0.9)
--lacunarity FLOAT         Lacunarity parameter (default: 2.0)
--clip-limit-range MIN MAX CLAHE clip limit (default: 1.0 2.0)
```

#### Average Noise
```
--path-average-noise PATH  Path to average noise images (required)
```

#### Bubble Noise
```
--spray-particles INT      Number of particles (default: auto)
--spray-diameter INT       Spray diameter (default: auto)
--range-of-blobs MIN MAX   Number of blobs (default: 30 40)
```

#### Pizza Noise
```
--nr-of-pizzas MIN MAX     Number of segments (default: 3 8)
--pizza-strength MIN MAX   Brightness change (default: 10 15)
```

#### Fourier Noise
```
--fourier-domain ampl|freq        Domain type (default: ampl)
--path-fourier-noise-ampl PATH    Amplitude noise path
--path-fourier-noise-freq PATH    Frequency noise path
--pass-value INT                  Filter pass value (default: 4)
--noise-proportion FLOAT          Noise proportion (default: 0.5)
```

## Example Workflows

### Generate Small Test Dataset

```bash
# 100 images (10 epsilon values × 10 copies)
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/test \
    --epsilon-range 0.0 0.1 \
    --epsilon-step 0.01 \
    --n-copies 10 \
    --seed 42
```

### Generate Full Steel Dataset with Perlin Noise

```bash
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/steel_perlin \
    --epsilon-range 0.0 1.0 \
    --epsilon-step 0.001 \
    --n-copies 50 \
    --alpha 0.45 \
    --seed 42
```

### Generate Combined Noise Dataset

```bash
# Perlin + Pizza + Bubble noise
python scripts/generate_dataset.py \
    --noise perlin pizza bubble \
    --output data/generated/combined \
    --n-copies 50 \
    --alpha 0.45 \
    --nr-of-pizzas 3 5 \
    --range-of-blobs 30 40 \
    --seed 42
```

## Parallel Generation

For large datasets, you can run multiple processes in parallel by splitting the epsilon range:

```bash
# Process 1: epsilon 0.0 to 0.25
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/parallel \
    --epsilon-range 0.0 0.25 \
    --n-copies 50 \
    --name-prefix img_1 \
    --seed 42 &

# Process 2: epsilon 0.25 to 0.5
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/parallel \
    --epsilon-range 0.25 0.5 \
    --n-copies 50 \
    --name-prefix img_2 \
    --seed 43 &

# Process 3: epsilon 0.5 to 0.75
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/parallel \
    --epsilon-range 0.5 0.75 \
    --n-copies 50 \
    --name-prefix img_3 \
    --seed 44 &

# Process 4: epsilon 0.75 to 1.0
python scripts/generate_dataset.py \
    --noise perlin \
    --output data/generated/parallel \
    --epsilon-range 0.75 1.0 \
    --n-copies 50 \
    --name-prefix img_4 \
    --seed 45 &

wait
echo "Parallel generation completed!"
```

## Output Structure

The script generates:
- PNG images named with the specified prefix (e.g., `img_00000.png`)
- `parameters.csv` containing metadata for each image:
  - `filename`: Image filename
  - `width`, `height`: Image dimensions
  - `epsilon`: Epsilon value
  - `ring_center_width`, `ring_center_height`: Ring center coordinates
  - `min_brightness`, `max_brightness`: Brightness range
  - `used_noise`: Noise index (for average noise)

## Tips

1. **Use seeds for reproducibility** - Always specify `--seed` for reproducible datasets
2. **Start small** - Test with small epsilon ranges before generating full datasets
3. **Monitor disk space** - 50,000 images (640×480) ≈ 2-3 GB
4. **Use parallel processing** - Split large jobs across multiple processes
5. **Combine noise types** - Perlin + Pizza noise often produces realistic results
