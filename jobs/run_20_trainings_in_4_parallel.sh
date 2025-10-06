#!/bin/bash
#SBATCH --account=gpu_gres
#SBATCH --job-name=bg_reweight_20_trainings
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2  # 4 array jobs total, 2 runs at a time

# ---- Script starts here ----

# Check arguments
if [[ -z "$1" ]]; then
    echo "Usage: $0 <CONFIG> [OUT_DIR]"
    exit 1
fi

CONFIG="$1"
OUT_DIR="${2:-../out/bkg_reweighting/${CONFIG}}"
CONFIG_FILE="../configs/bkg_reweighting/${CONFIG}.yml"

LOAD_LAST=false
INIT_SEED=$((5 * SLURM_ARRAY_TASK_ID))

echo "Using CONFIG: $CONFIG"
echo "Using CONFIG_FILE: $CONFIG_FILE"
echo "Output directory: $OUT_DIR"

# Start 5 jobs in parallel
for i in {0..4}; do
    SEED=$((i + INIT_SEED))
    RUN_DIR="${OUT_DIR}/run$(printf "%02d" $SEED)"
    MODEL_DIR="$RUN_DIR/state_dict"
    mkdir -p "$RUN_DIR"

    if $LOAD_LAST; then
        for ((epoch=50; epoch>=1; epoch--)); do
            candidate="$MODEL_DIR/model_${epoch}_state_dict.pt"
            echo "Searching for $candidate"
            if [[ -f "$candidate" ]]; then
                MODEL_PATH="$candidate"
                echo "Resuming from: $MODEL_PATH"
                break
            fi
        done
    fi

    echo "[$SLURM_ARRAY_TASK_ID] Launching training with seed $SEED -> $RUN_DIR"

	echo $MODEL_DIR
    # Run each training in background (&) to parallelize
    shopt -s nullglob
    BEST_MODEL=$(ls "$MODEL_DIR"/*best_epoch*.onnx 2>/dev/null | head -n 1)
echo $BEST_MODEL
	# Capture matching files into an arrayV
    matches=("$MODEL_DIR"/*best_epoch*.onnx)

	# Set BEST_MODEL only if we found matches
    if [[ "$LOAD_LAST" == true && ${#matches[@]} -gt 0 ]]; then
        echo "Skipping this run, because it is already finished"
    elif [[ -s "./comet_token.key" ]]; then
        {
            read -r API_UNAME 
            read -r API_KEY
        } <./comet_token.key
        API_TAGS=("DNN_training" "bkg_reweighting" "slurm")
        echo "Found Comet username: $API_UNAME"
        echo "Found Comet API key."

        if $LOAD_LAST; then
            ml_train -o "$RUN_DIR" \
                     --eval --onnx --roc --histos --history \
                     --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" \
                     --load-model "$MODEL_PATH" \
                     --comet-token "$API_KEY" --comet-name "$API_UNAME" \
                     --comet-tags "${API_TAGS[@]}" &
        else
            ml_train -o "$RUN_DIR" \
                     --eval --onnx --roc --histos --history \
                     --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" --overwrite \
                     --comet-token "$API_KEY" --comet-name "$API_UNAME" \
                     --comet-tags "${API_TAGS[@]}" &
        fi
    else
        if $LOAD_LAST; then
            ml_train -o "$RUN_DIR" \
                     --eval --onnx --roc --histos --history \
                     --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" \
                     --load-model "$MODEL_PATH" &
        else
            ml_train -o "$RUN_DIR" \
                     --eval --onnx --roc --histos --history \
                     --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" --overwrite &
        fi
    fi

done

# Wait for all background jobs to finish
wait
echo "All trainings finished for this batch."

# Post-processing: collect best models
for i in {0..4}; do
    SEED=$((i + INIT_SEED))
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
