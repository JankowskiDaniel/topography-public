#!/bin/bash
# Example script for parallel Perlin noise dataset generation
# Splits generation into 4 parallel processes

OUTPUT_DIR="data/generated/perlin_parallel"
N_COPIES=50
ALPHA=0.45

echo "========================================="
echo "Parallel Perlin Noise Generation"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Splitting into 4 parallel processes"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process 1: epsilon 0.0 to 0.25
echo "Starting process 1 (epsilon 0.0-0.25)..."
python scripts/generate_dataset.py \
    --noise perlin \
    --output "$OUTPUT_DIR" \
    --epsilon-range 0.0 0.25 \
    --epsilon-step 0.001 \
    --n-copies $N_COPIES \
    --name-prefix img_p1 \
    --alpha $ALPHA \
    --seed 42 &
PID1=$!

# Process 2: epsilon 0.25 to 0.5
echo "Starting process 2 (epsilon 0.25-0.5)..."
python scripts/generate_dataset.py \
    --noise perlin \
    --output "$OUTPUT_DIR" \
    --epsilon-range 0.25 0.5 \
    --epsilon-step 0.001 \
    --n-copies $N_COPIES \
    --name-prefix img_p2 \
    --alpha $ALPHA \
    --seed 43 &
PID2=$!

# Process 3: epsilon 0.5 to 0.75
echo "Starting process 3 (epsilon 0.5-0.75)..."
python scripts/generate_dataset.py \
    --noise perlin \
    --output "$OUTPUT_DIR" \
    --epsilon-range 0.5 0.75 \
    --epsilon-step 0.001 \
    --n-copies $N_COPIES \
    --name-prefix img_p3 \
    --alpha $ALPHA \
    --seed 44 &
PID3=$!

# Process 4: epsilon 0.75 to 1.0
echo "Starting process 4 (epsilon 0.75-1.0)..."
python scripts/generate_dataset.py \
    --noise perlin \
    --output "$OUTPUT_DIR" \
    --epsilon-range 0.75 1.0 \
    --epsilon-step 0.001 \
    --n-copies $N_COPIES \
    --name-prefix img_p4 \
    --alpha $ALPHA \
    --seed 45 &
PID4=$!

# Wait for all processes to complete
echo ""
echo "Waiting for all processes to complete..."
wait $PID1
echo "Process 1 completed"
wait $PID2
echo "Process 2 completed"
wait $PID3
echo "Process 3 completed"
wait $PID4
echo "Process 4 completed"

echo ""
echo "========================================="
echo "All processes completed!"
echo "Merging parameter files..."
echo "========================================="

# Merge CSV files (if you want a single parameters.csv)
# First, copy the header from the first file
head -1 "$OUTPUT_DIR/parameters.csv" > "$OUTPUT_DIR/parameters_merged.csv"

# Then append data from all files (skip headers)
for csv in "$OUTPUT_DIR"/parameters*.csv; do
    tail -n +2 "$csv" >> "$OUTPUT_DIR/parameters_merged.csv"
done

echo "Merged parameters saved to: $OUTPUT_DIR/parameters_merged.csv"
echo "Total images generated: $(ls "$OUTPUT_DIR"/*.png | wc -l)"
