#!/bin/bash
# Example script for generating dataset with combined noise types
# This combines Perlin + Triangular noise for more realistic results

OUTPUT_DIR="data/generated/perlin_triangular"
EPSILON_START=0.0
EPSILON_END=1.0
EPSILON_STEP=0.001
N_COPIES=50
SEED=42

echo "========================================="
echo "Combined Noise Dataset Generation"
echo "Perlin + Triangular Noise"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Epsilon range: $EPSILON_START to $EPSILON_END"
echo "========================================="

# Generate dataset with both Perlin and Triangular noise
python scripts/generate_dataset.py \
    --noise perlin triangular \
    --output "$OUTPUT_DIR" \
    --epsilon-range $EPSILON_START $EPSILON_END \
    --epsilon-step $EPSILON_STEP \
    --n-copies $N_COPIES \
    --alpha 0.45 \
    --scale 100.0 \
    --octaves 6 \
    --persistence-range 0.7 0.9 \
    --lacunarity 2.0 \
    --clip-limit-range 1.0 2.0 \
    --nr-of-triangulars 3 5 \
    --triangular-strength 20 30 \
    --seed $SEED

echo ""
echo "Generation completed!"
echo "Images saved to: $OUTPUT_DIR"
