#!/bin/bash
#SBATCH --account=gpu_gres
#SBATCH --job-name=bg_reweight_20_trainings
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8000
#SBATCH --time=6:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# ---- Script starts here ----

# Fix: Proper Bash arithmetic for array task ID
# INIT_SEED=$((5 * SLURM_ARRAY_TASK_ID))
# INIT_SEED=7
# INIT_SEED=$((2 * SLURM_ARRAY_TASK_ID + 14))

# Set config and output directories
CONFIG_FILE=../configs/hh4b_bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_inclusive_b_region_bratioAll_postEE_DeltaProb.yml
OUT_DIR=../out/bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_inclusive_b_region_bratioAll_postEE_DeltaProb

# Start 5 jobs in parallel
#for i in {0..0}; do
for INIT_SEED in 0 6 10 18; do
    SEED=$((i + INIT_SEED))
    #SEED=$((INIT_SEED))
    RUN_DIR="${OUT_DIR}/run$(printf "%02d" $SEED)"
    #mkdir -p "$RUN_DIR"

    echo "[$SLURM_ARRAY_TASK_ID] Launching training with seed $SEED -> $RUN_DIR"

    # Run each training in background (&) to parallelize
    ml_train -o "$RUN_DIR" \
             --eval --onnx --roc --histos --history \
             --gpus 0 -n 1 -c "$CONFIG_FILE" -s "$SEED"&
done

# Wait for all background jobs to finish
wait
echo "All trainings finished for this batch."

# Post-processing: collect best models
#for i in {0..0}; do
for INIT_SEED in 0 6 10 18; do
    SEED=$((i + INIT_SEED))
    #SEED=$((INIT_SEED))
    RUN_DIR="${OUT_DIR}/run$(printf "%02d" $SEED)"
    MODEL_DIR="$RUN_DIR/state_dict"
    BEST_MODEL=$(ls "$MODEL_DIR"/*best_epoch*.onnx 2>/dev/null | head -n 1)

    if [[ -n "$BEST_MODEL" ]]; then
        mkdir -p "$OUT_DIR/best_models"
        cp "$BEST_MODEL" "$OUT_DIR/best_models/best_model_run$(printf "%02d" $SEED).onnx"
        echo "Copied best model to best_models/ for seed $SEED"
    else
        echo "No best model found in $MODEL_DIR for seed $SEED"
    fi
done
