#!/bin/bash
#
# Trains the main Refined-AFGCN model and specified baselines.
# Checkpoints are saved to the 'checkpoints' directory.

# --- Configuration ---
TRAIN_DIR="training_data"
VAL_DIR="validation_data"
CHECKPOINT_DIR="checkpoints"
EPOCHS=100
COMMON_ARGS="--training_dir $TRAIN_DIR --validation_dir $VAL_DIR --epochs $EPOCHS --use_class_weights"

# Ensure checkpoint directory exists
mkdir -p $CHECKPOINT_DIR

# --- Train Main Model ---
echo "--- Training Main Refined-AFGCN Model ---"
python train_consolidated.py \
    --model refined \
    $COMMON_ARGS \
    --rank_loss_weight 0.1 \
    --checkpoint "$CHECKPOINT_DIR/refined_best.pth"

# --- Train Baselines ---
BASELINES="afgcn gcn gat gin graphsage"

for model_name in $BASELINES; do
    echo "--- Training Baseline: $model_name ---"
    python train_consolidated.py \
        --model "$model_name" \
        $COMMON_ARGS \
        --no_rank_loss \
        --checkpoint "$CHECKPOINT_DIR/${model_name}_best.pth"
done

echo "--- All main and baseline training complete. ---"