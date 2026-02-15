#!/bin/bash
# Example script for generating Perlin noise dataset
# This replicates the functionality of the old gen1.py, gen2.py, etc.

# Configuration
OUTPUT_DIR="data/generated/perlin"
EPSILON_START=0.0
EPSILON_END=1.0
EPSILON_STEP=0.001
N_COPIES=50
ALPHA=0.45
SEED=42

echo "========================================="
echo "Perlin Noise Dataset Generation"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Epsilon range: $EPSILON_START to $EPSILON_END"
echo "Step: $EPSILON_STEP"
echo "Copies per epsilon: $N_COPIES"
echo "Alpha: $ALPHA"
echo "Seed: $SEED"
echo "========================================="

# Generate dataset
python scripts/generate_dataset.py \
    --noise perlin \
    --output "$OUTPUT_DIR" \
    --epsilon-range $EPSILON_START $EPSILON_END \
    --epsilon-step $EPSILON_STEP \
    --n-copies $N_COPIES \
    --alpha $ALPHA \
    --scale 100.0 \
    --octaves 6 \
    --persistence-range 0.7 0.9 \
    --lacunarity 2.0 \
    --clip-limit-range 1.0 2.0 \
    --seed $SEED

echo ""
echo "Generation completed!"
echo "Images saved to: $OUTPUT_DIR"
