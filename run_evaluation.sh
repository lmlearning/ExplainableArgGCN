#!/bin/bash
#
# Runs a two-stage evaluation:
# 1. Intrinsic Metrics: Detailed explainability metrics for the main Refined-AFGCN model.
# 2. Comparative Metrics: Core metrics for the Refined-AFGCN model against all trained baselines.

# --- Configuration ---
TEST_DATA_DIR="test_data"
CHECKPOINT_DIR="checkpoints"
OUTPUT_DIR="results"

# --- Pre-flight Checks ---
if [ ! -d "$TEST_DATA_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Required directories ('test_data' or 'checkpoints') not found."
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# --- Stage 1: Intrinsic Explainability Evaluation (Refined Model Only) ---
echo "--- Stage 1: Running Intrinsic Explainability Evaluation ---"
REFINED_CKPT="$CHECKPOINT_DIR/refined_best.pth"
INTRINSIC_METRICS_FILE="$OUTPUT_DIR/intrinsic_metrics.json"

if [ -f "$REFINED_CKPT" ]; then
    python3 evaluate_intrinsic.py --data_dir "$TEST_DATA_DIR" --ckpt "$REFINED_CKPT" --export_json "$INTRINSIC_METRICS_FILE"
    echo "Intrinsic metrics saved to: $INTRINSIC_METRICS_FILE"
else
    echo "Warning: Refined model checkpoint not found. Skipping intrinsic evaluation."
fi
echo

# --- Stage 2: Comparative Evaluation vs. Baselines ---
echo "--- Stage 2: Running Comparative Evaluation vs. Baselines ---"
COMPARATIVE_METRICS_FILE="$OUTPUT_DIR/comparative_metrics.csv"
MODELS_TO_EVAL="refined afgcn gcn gat gin"
EVAL_ARGS=""

for model_name in $MODELS_TO_EVAL; do
    ckpt_path="$CHECKPOINT_DIR/${model_name}_best.pth"
    if [ -f "$ckpt_path" ]; then
        # Correctly format as "key=value" for the positional argument
        EVAL_ARGS+="${model_name}=${ckpt_path} "
    else
        echo "Warning: Checkpoint for baseline '$model_name' not found. It will be excluded from comparison."
    fi
done

if [ -n "$EVAL_ARGS" ]; then
    python3 evaluate_pyg_baselines.py \
        --data_dir "$TEST_DATA_DIR" \
        $EVAL_ARGS > "$COMPARATIVE_METRICS_FILE"
    echo "Comparative metrics saved to: $COMPARATIVE_METRICS_FILE"
else
    echo "No valid checkpoints found for comparative evaluation. Skipping."
fi
echo

echo "--- Evaluation complete. ---"