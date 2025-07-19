#!/bin/bash
#
# Runs a four-stage evaluation:
# 1. Intrinsic Metrics: Detailed explainability metrics for the main Refined-AFGCN model.
# 2. Comparative Metrics: Core metrics for the Refined-AFGCN model against all trained baselines.
# 3. Ablation Analysis: Core metrics for each model variant from the ablation studies.
# 4. Rank Loss Sweep Analysis: Core metrics for each model variant from the rank loss sweep.

# --- Configuration ---
TEST_DATA_DIR="test_data"
CHECKPOINT_DIR="checkpoints"
OUTPUT_DIR="results"

# --- Pre-flight Checks ---
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "Error: Test data directory not found at '$TEST_DATA_DIR'"
    exit 1
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoints directory not found at '$CHECKPOINT_DIR'. Please run training first."
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# --- Stage 1: Intrinsic Explainability Evaluation ---
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
    if [ -f "$ckpt_path" ]; then EVAL_ARGS+="$model_name=$ckpt_path "; else echo "Warning: Checkpoint for '$model_name' not found."; fi
done
if [ -n "$EVAL_ARGS" ]; then
    python3 evaluate_pyg_baselines.py --data_dir "$TEST_DATA_DIR" $EVAL_ARGS > "$COMPARATIVE_METRICS_FILE"
    echo "Comparative metrics saved to: $COMPARATIVE_METRICS_FILE"
fi
echo

# --- Stage 3: Ablation Study Evaluation ---
echo "--- Stage 3: Running Ablation Study Evaluation ---"
ABLATION_CHECKPOINT_DIR="$CHECKPOINT_DIR/ablations"
ABLATION_METRICS_FILE="$OUTPUT_DIR/ablation_metrics.csv"
if [ -d "$ABLATION_CHECKPOINT_DIR" ] && [ "$(ls -A $ABLATION_CHECKPOINT_DIR)" ]; then
    echo "ablation_name,model,accuracy,mcc,spearman,kendall" > "$ABLATION_METRICS_FILE"
    for ckpt_path in "$ABLATION_CHECKPOINT_DIR"/*.pth; do
        if [ -f "$ckpt_path" ]; then
            filename=$(basename "$ckpt_path")
            ablation_name=$(echo "$filename" | sed -n 's/refined_ablation_\(.*\)\.pth/\1/p')
            echo "Evaluating ablation checkpoint for: $ablation_name"
            output=$(python3 evaluate_pyg_baselines.py --data_dir "$TEST_DATA_DIR" "refined=$ckpt_path" | tail -n 1)
            echo "$ablation_name,$output" >> "$ABLATION_METRICS_FILE"
        fi
    done
    echo "Ablation metrics saved to: $ABLATION_METRICS_FILE"
else
    echo "Warning: Ablation checkpoints not found. Skipping ablation evaluation."
fi
echo

# --- Stage 4: Rank Loss Sweep Evaluation ---
echo "--- Stage 4: Running Rank Loss Sweep Evaluation ---"
SWEEP_CHECKPOINT_DIR="$CHECKPOINT_DIR/rank_sweep"
SWEEP_METRICS_FILE="$OUTPUT_DIR/rank_sweep_metrics.csv"
if [ -d "$SWEEP_CHECKPOINT_DIR" ] && [ "$(ls -A $SWEEP_CHECKPOINT_DIR)" ]; then
    echo "lambda,model,accuracy,mcc,spearman,kendall" > "$SWEEP_METRICS_FILE"
    for ckpt_path in "$SWEEP_CHECKPOINT_DIR"/*.pth; do
        if [ -f "$ckpt_path" ]; then
            filename=$(basename "$ckpt_path")
            lambda_str=$(echo "$filename" | sed -n 's/refined_lambda_\(.*\)\.pth/\1/p' | tr '_' '.')
            echo "Evaluating sweep checkpoint for lambda = $lambda_str"
            output=$(python3 evaluate_pyg_baselines.py --data_dir "$TEST_DATA_DIR" "refined=$ckpt_path" | tail -n 1)
            echo "$lambda_str,$output" >> "$SWEEP_METRICS_FILE"
        fi
    done
    echo "Rank loss sweep metrics saved to: $SWEEP_METRICS_FILE"
else
    echo "Warning: Rank loss sweep checkpoints not found. Skipping sweep evaluation."
fi
echo

echo "--- All evaluations complete. ---"