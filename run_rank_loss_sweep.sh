#!/bin/bash
#
# Trains 10 variants of the Refined-AFGCN model across a range of
# rank loss weights to analyze the trade-off between accuracy and alignment.

# --- Configuration ---
TRAIN_DIR="training_data"
VAL_DIR="validation_data"
CHECKPOINT_DIR="checkpoints/rank_sweep"
EPOCHS=100
COMMON_ARGS="--model refined --training_dir $TRAIN_DIR --validation_dir $VAL_DIR --epochs $EPOCHS --use_class_weights"


# Logarithmic scale for lambda (0.1, ~0.21, ~0.46, ..., 100)
LAMBDAS=($(seq 0 9 | xargs -I {} echo "10^({}-1)" | bc -l))

# Ensure checkpoint directory exists
mkdir -p $CHECKPOINT_DIR

# --- Run Rank Loss Sweep ---
echo "--- Starting rank loss sweep for Refined-AFGCN ---"
for lambda_val in "${LAMBDAS[@]}"; do
    # Format for filename, e.g., 0.1 becomes 0_1
    filename_lambda=$(echo "$lambda_val" | tr '.' '_')
    echo "Training with rank_loss_weight = $lambda_val"

    python3 train_consolidated.py \
        $COMMON_ARGS \
        --rank_loss_weight "$lambda_val" \
        --checkpoint "$CHECKPOINT_DIR/refined_lambda_${filename_lambda}.pth"
done

echo "--- Rank loss sweep complete. ---"