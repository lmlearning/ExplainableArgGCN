#!/bin/bash
#
# Generates multiple neighborhood visualizations by automatically sampling
# nodes from the test dataset, first by top influence and then randomly.

# --- Configuration ---
TEST_DATA_DIR="test_data"
CHECKPOINT_FILE=${1:-"checkpoints/refined_best.pth"}
OUTPUT_DIR_BASE="visualizations"
ATTACKER_THRESHOLD=0.85
DEFENDER_THRESHOLD=0.75

# --- Pre-flight Checks ---
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file not found at '$CHECKPOINT_FILE'"
    exit 1
fi
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "Error: Test data directory not found at '$TEST_DATA_DIR'"
    exit 1
fi

# --- Step 1: Generate Visualizations for Top 3 Most Influential Nodes ---
TOP_SAMPLES_DIR="$OUTPUT_DIR_BASE/top_influence_samples"
echo "--- Generating visualizations for Top 3 influential nodes ---"
rm -rf "$TOP_SAMPLES_DIR"
mkdir -p "$TOP_SAMPLES_DIR"

python visualize_neighbourhoods.py \
    --data_dir "$TEST_DATA_DIR" \
    --ckpt "$CHECKPOINT_FILE" \
    --output_dir "$TOP_SAMPLES_DIR" \
    --num_samples 3 \
    --sampling_method "top" \
    --influence_threshold $ATTACKER_THRESHOLD $DEFENDER_THRESHOLD \
    --in_only

echo "Top influence DOT files saved to: $TOP_SAMPLES_DIR"
echo

# --- Step 2: Generate Visualizations for 10 Random Nodes ---
RANDOM_SAMPLES_DIR="$OUTPUT_DIR_BASE/random_samples"
echo "--- Generating visualizations for 10 random nodes ---"
rm -rf "$RANDOM_SAMPLES_DIR"
mkdir -p "$RANDOM_SAMPLES_DIR"

python visualize_neighbourhoods.py \
    --data_dir "$TEST_DATA_DIR" \
    --ckpt "$CHECKPOINT_FILE" \
    --output_dir "$RANDOM_SAMPLES_DIR" \
    --num_samples 10 \
    --sampling_method "random" \
    --influence_threshold $ATTACKER_THRESHOLD $DEFENDER_THRESHOLD

echo "Random sample DOT files saved to: $RANDOM_SAMPLES_DIR"
echo
echo "--- Visualization generation complete. ---"