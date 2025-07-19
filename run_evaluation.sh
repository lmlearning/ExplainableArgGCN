#!/bin/bash
#
# Runs the quantitative evaluation to generate intrinsic explainability metrics
# (Spearman, Gini, AUC, etc.) over the entire test set.

# --- Configuration ---
TEST_DATA_DIR="test_data"
CHECKPOINT_FILE=${1:-"checkpoints/refined_best.pth"}
OUTPUT_DIR="results"
METRICS_FILE="intrinsic_metrics.json"

# --- Pre-flight Checks ---
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file not found at '$CHECKPOINT_FILE'"
    exit 1
fi
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "Error: Test data directory not found at '$TEST_DATA_DIR'"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# --- Run Quantitative Evaluation Script ---
echo "--- Running quantitative evaluation for model: $CHECKPOINT_FILE ---"
python evaluate_intrinsic.py \
    --data_dir "$TEST_DATA_DIR" \
    --ckpt "$CHECKPOINT_FILE" \
    --export_json "$OUTPUT_DIR/$METRICS_FILE"

echo "--- Quantitative evaluation complete. ---"
echo "Metrics saved to: $OUTPUT_DIR/$METRICS_FILE"