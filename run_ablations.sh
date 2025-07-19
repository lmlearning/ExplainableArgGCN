#!/bin/bash
#
# Runs all internal ablation studies for the Refined-AFGCN model.
# Checkpoints are saved to 'checkpoints/ablations/'.

# --- Configuration ---
TRAIN_DIR="training_data"
VAL_DIR="validation_data"
CHECKPOINT_DIR="checkpoints/ablations"
EPOCHS=100
COMMON_ARGS="--model refined --training_dir $TRAIN_DIR --validation_dir $VAL_DIR --epochs $EPOCHS --use_class_weights"

# Maps the CLI flag to the checkpoint filename suffix
declare -A ABLATIONS
ABLATIONS=(
    ["--no_pairing"]="no_pairing"
    ["--no_struct"]="no_struct"
    ["--no_residual"]="no_residual"
    ["--no_ln"]="no_ln"
    ["--no_rank_loss"]="no_rank_loss"
)

# Ensure checkpoint directory exists
mkdir -p $CHECKPOINT_DIR

# --- Run Ablation Trainings ---
for flag in "${!ABLATIONS[@]}"; do
    suffix="${ABLATIONS[$flag]}"
    echo "--- Running Ablation: $suffix ($flag) ---"

    if [[ "$flag" == "--no_rank_loss" ]]; then
        RANK_ARG="--no_rank_loss"
    else
        RANK_ARG="--rank_loss_weight 0.1"
    fi

    python train_consolidated.py \
        $COMMON_ARGS \
        $flag \
        $RANK_ARG \
        --checkpoint "$CHECKPOINT_DIR/refined_ablation_${suffix}.pth"
done

echo "--- All ablation studies complete. ---"