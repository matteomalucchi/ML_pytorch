#!/bin/bash
#SBATCH --account=gpu_gres
#SBATCH --job-name=ml_train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=8:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Generic ML training script.
# Self-submits to Slurm when invoked directly.
#
# Usage:
#   ./run_training.sh --config /full/path/config.yml --outdir /full/path/outdir [OPTIONS] [-- EXTRA_ML_TRAIN_ARGS]
#
# Required:
#   -c, --config FILE       Full path to YAML config file
#   -o, --outdir DIR        Full path to output directory
#
# Optional:
#   -n, --n-trainings INT   Total number of trainings (default: 1)
#   -p, --nodes INT         Number of parallel Slurm nodes/array jobs (default: 1)
#   -s, --init-seed INT     Starting random seed (default: 0)
#   --ratio                 Use average-ratio ONNX aggregation (ml_onnx -ar)
#   -k, --k-folds INT       K-fold training: train the K folds (one job each,
#                           same seed) and merge them with ml_kfold_merge
#   --load-last             Resume from latest checkpoint instead of restarting
#   --time HH:MM:SS         Slurm wall-clock time limit (default: 8:30:00)
#   --mem-per-cpu MB        Slurm memory per CPU in MB (default: 4000)
#   --cpus INT              Slurm CPUs per task (default: 4)
#   -g, --gpus-per-node INT GPUs per Slurm node/array job (default: 1). The
#                           trainings of a node are distributed round-robin
#                           over its GPUs (one process per training)
#   --no-slurm              Run directly without Slurm (for local testing)
#   -- EXTRA                Additional arguments forwarded to ml_train

# ─── Defaults ─────────────────────────────────────────────────────────────────
N_TRAININGS=1
NODES=1
INIT_SEED=0
RATIO=false
K_FOLDS=0
LOAD_LAST=false
MODE="train"
NO_SLURM=false
EXTRA_ARGS=()
CONFIG_FILE=""
OUT_DIR=""
SLURM_TIME="12:00:00"
SLURM_MEM=4000
SLURM_CPUS=4
GPUS_PER_NODE=1

# ─── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --config /full/path/config.yml --outdir /full/path/outdir [OPTIONS] [-- EXTRA_ML_TRAIN_ARGS]

Required:
  -c, --config FILE       Full path to YAML config file
  -o, --outdir DIR        Full path to output directory

Optional:
  -n, --n-trainings INT   Total number of trainings (default: 1)
  -p, --nodes INT         Number of parallel Slurm nodes/array jobs (default: 1)
  -s, --init-seed INT     Starting random seed (default: 0)
  --ratio                 Average-ratio ONNX aggregation (ml_onnx -ar)
  -k, --k-folds INT       K-fold training: train the K folds in parallel (same
                          seed, outputs in OUTDIR/fold_<k>) and merge them with
                          ml_kfold_merge (not combinable with -n > 1)
  --load-last             Resume from latest checkpoint
  --time HH:MM:SS         Slurm wall-clock time limit (default: 8:30:00)
  --mem-per-cpu MB        Slurm memory per CPU in MB (default: 4000)
  --cpus INT              Slurm CPUs per task (default: 4)
  -g, --gpus-per-node INT GPUs per node/array job (default: 1); the trainings
                          of a node are spread round-robin over its GPUs
  --no-slurm              Run directly without Slurm (for testing; forces NODES=1)
  -- EXTRA                Extra arguments forwarded to ml_train

Examples:
  # Single training
  $0 -c /full/path/DNN_config.yml -o /full/path/out

  # 20 trainings across 4 nodes, with ratio ONNX aggregation
  $0 -c /full/path/DNN_config.yml -o /full/path/out -n 20 -p 4 --ratio

  # 20 trainings on 1 node with 4 GPUs (5 trainings per GPU)
  $0 -c /full/path/DNN_config.yml -o /full/path/out -n 20 -g 4 --cpus 8 --ratio

  # 5 trainings on 1 node, just convert (no ratio)
  $0 -c /full/path/DNN_config.yml -o /full/path/out -n 5

  # 5-fold training, the 5 folds in parallel on 1 node, with ratio ONNX models
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 5 --ratio

  # Local test
  $0 -c /full/path/DNN_config.yml -o /full/path/out --no-slurm
EOF
    exit 1
}

# ─── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)       CONFIG_FILE="$2"; shift 2 ;;
        -o|--outdir)       OUT_DIR="$2"; shift 2 ;;
        -n|--n-trainings)  N_TRAININGS="$2"; shift 2 ;;
        -p|--nodes)        NODES="$2"; shift 2 ;;
        -s|--init-seed)    INIT_SEED="$2"; shift 2 ;;
        --ratio)           RATIO=true; shift ;;
        -k|--k-folds)      K_FOLDS="$2"; shift 2 ;;
        --load-last)       LOAD_LAST=true; shift ;;
        --time)            SLURM_TIME="$2"; shift 2 ;;
        --mem-per-cpu)     SLURM_MEM="$2"; shift 2 ;;
        --cpus)            SLURM_CPUS="$2"; shift 2 ;;
        -g|--gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --no-slurm)        NO_SLURM=true; shift ;;
        --mode)            MODE="$2"; shift 2 ;;
        -h|--help)         usage ;;
        --)                shift; EXTRA_ARGS=("$@"); break ;;
        *)                 echo "Unknown argument: $1"; usage ;;
    esac
done

# ─── Validation ───────────────────────────────────────────────────────────────
if [[ -z "$CONFIG_FILE" || -z "$OUT_DIR" ]]; then
    echo "Error: --config and --outdir are required."
    usage
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ "$GPUS_PER_NODE" -lt 1 ]]; then
    echo "Error: --gpus-per-node must be >= 1."
    exit 1
fi

# K-fold training: one training per fold, all with the same seed
if [[ "$K_FOLDS" -ge 2 ]]; then
    if [[ "$N_TRAININGS" -gt 1 && "$N_TRAININGS" -ne "$K_FOLDS" ]]; then
        echo "Error: --k-folds cannot be combined with --n-trainings > 1."
        exit 1
    fi
    N_TRAININGS="$K_FOLDS"
fi

# With --no-slurm, collapse to a single-node run
[[ "$NO_SLURM" == true ]] && NODES=1

TRAININGS_PER_NODE=$(( (N_TRAININGS + NODES - 1) / NODES ))


# ─── Helper: canonical run directory for a given seed ────────────────────────
# (with --k-folds the index is the fold: OUT_DIR/fold_<seed - INIT_SEED>)
run_path() {
    if [[ "$K_FOLDS" -ge 2 ]]; then
        printf "%s/fold_%d" "$OUT_DIR" "$(( $1 - INIT_SEED ))"
    else
        printf "%s/run%02d" "$OUT_DIR" "$1"
    fi
}

# ─── Build NUL-delimited arg list for self-submission ─────────────────────────
# Uses NUL delimiter so paths with spaces are handled correctly.
build_submit_args() {
    local target_mode="$1"
    local args=(
        --config "$CONFIG_FILE"
        --outdir "$OUT_DIR"
        --n-trainings "$N_TRAININGS"
        --nodes "$NODES"
        --init-seed "$INIT_SEED"
        --k-folds "$K_FOLDS"
        --time "$SLURM_TIME"
        --mem-per-cpu "$SLURM_MEM"
        --cpus "$SLURM_CPUS"
        --gpus-per-node "$GPUS_PER_NODE"
        --mode "$target_mode"
    )
    [[ "$RATIO"     == true ]] && args+=(--ratio)
    [[ "$LOAD_LAST" == true ]] && args+=(--load-last)
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && args+=(-- "${EXTRA_ARGS[@]}")
    printf '%s\0' "${args[@]}"
}

# ─── Self-submission ──────────────────────────────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" && "$NO_SLURM" != true ]]; then
    SCRIPT_PATH="$(realpath "$0")"
    # Inside a Slurm job $0 points to the spooled copy, so pass the real
    # script directory through the environment (sbatch exports it by default).
    export ML_TRAIN_SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

    SBATCH_RESOURCES=(
        --time="$SLURM_TIME"
        --mem-per-cpu="$SLURM_MEM"
        --cpus-per-task="$SLURM_CPUS"
    )

    mkdir -p "$OUT_DIR"
    mkdir -p "$(run_path "$INIT_SEED")"

    # Slurm logs always go to OUT_DIR: ml_train --overwrite does rm -rf run_dir,
    # which would delete a log written inside the run directory.
    SLURM_LOG_DIR="$OUT_DIR"

    if [[ "$NODES" -eq 1 ]]; then
        # Single job: all trainings on one node, postproc runs inline
        mapfile -d '' TRAIN_ARGS < <(build_submit_args train)
        echo "Submitting single job (${N_TRAININGS} training(s) on ${GPUS_PER_NODE} GPU(s))..."
        sbatch --job-name="ml_train" \
            --output="${SLURM_LOG_DIR}/slurm-%j.out" \
            --gres="gpu:${GPUS_PER_NODE}" \
        "${SBATCH_RESOURCES[@]}" "$SCRIPT_PATH" "${TRAIN_ARGS[@]}"
    else
        # Array job per node + dependent postproc job
        ARRAY_MAX=$(( NODES - 1 ))
        mapfile -d '' TRAIN_ARGS < <(build_submit_args train)
        echo "Submitting array job: ${NODES} nodes × ${TRAININGS_PER_NODE} trainings on ${GPUS_PER_NODE} GPU(s) each (total: ${N_TRAININGS})..."
        TRAIN_JOB_ID=$(sbatch --parsable \
            --job-name="ml_train_array" \
            --array="0-${ARRAY_MAX}" \
            --output="${SLURM_LOG_DIR}/slurm-%A_%a.out" \
            --gres="gpu:${GPUS_PER_NODE}" \
            "${SBATCH_RESOURCES[@]}" \
            "$SCRIPT_PATH" "${TRAIN_ARGS[@]}")
        echo "  Training array job submitted: ${TRAIN_JOB_ID}"

        mapfile -d '' POSTPROC_ARGS < <(build_submit_args postproc)
        echo "Submitting post-processing job (depends on ${TRAIN_JOB_ID})..."
        POSTPROC_JOB_ID=$(sbatch --parsable \
            --job-name="ml_postproc" \
            --dependency="afterok:${TRAIN_JOB_ID}" \
            --output="${SLURM_LOG_DIR}/slurm-%j.out" \
            --gres="gpu:1" \
            "${SBATCH_RESOURCES[@]}" \
            "$SCRIPT_PATH" "${POSTPROC_ARGS[@]}")
        echo "  Post-processing job submitted: ${POSTPROC_JOB_ID}"
    fi
    exit 0
fi

# ─── Comet ML credentials (looked up relative to script) ──────────────────────
SCRIPT_DIR="${ML_TRAIN_SCRIPT_DIR:-$(dirname "$(realpath "$0")")}"
COMET_KEY_FILE="${SCRIPT_DIR}/comet_token.key"
USE_COMET=false
API_UNAME=""
API_KEY=""

if [[ -s "$COMET_KEY_FILE" ]]; then
    { read -r API_UNAME; read -r API_KEY; } < "$COMET_KEY_FILE"
    echo "Found Comet credentials for: $API_UNAME"
    USE_COMET=true
fi

# ─── Run a single training ─────────────────────────────────────────────────────
run_one_training() {
    local seed="$1"
    local run_dir="$2"
    local gpu="$3"
    local num_workers="$4"
    local model_dir="${run_dir}/state_dict"
    mkdir -p "$run_dir"

    # Skip if already finished (only relevant when --load-last)
    shopt -s nullglob
    local finished_models=("$model_dir"/*best_epoch*.onnx)
    shopt -u nullglob
    if [[ "$LOAD_LAST" == true && ${#finished_models[@]} -gt 0 ]]; then
        echo "Skipping seed ${seed}: already finished."
        return 0
    fi

    # Find latest checkpoint for resuming
    local load_model_args=()
    if [[ "$LOAD_LAST" == true ]]; then
        for ((epoch=200; epoch>=1; epoch--)); do
            local ckpt="${model_dir}/model_${epoch}_state_dict.pt"
            if [[ -f "$ckpt" ]]; then
                load_model_args=(--load-model "$ckpt")
                echo "Seed ${seed}: resuming from epoch ${epoch} (${ckpt})"
                break
            fi
        done
    fi

    local config_tag
    config_tag=$(basename "$CONFIG_FILE" .yml)

    local train_args=(
        -o "$run_dir"
        --eval --onnx --roc --histos --history
        --gpus "$gpu" -n "$num_workers"
        -c "$CONFIG_FILE"
        "${load_model_args[@]}"
    )
    [[ "$LOAD_LAST" != true ]] && train_args+=(--overwrite)
    if [[ "$K_FOLDS" -ge 2 ]]; then
        train_args+=(-s "$INIT_SEED" --k-folds "$K_FOLDS" --fold "$(( seed - INIT_SEED ))")
    elif [[ "$N_TRAININGS" -gt 1 ]]; then
        train_args+=(-s "$seed")
    fi
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && train_args+=("${EXTRA_ARGS[@]}")

    local comet_args=()
    if [[ "$USE_COMET" == true ]]; then
        comet_args=(
            --comet-token "$API_KEY"
            --comet-name "$API_UNAME"
            --comet-tags "DNN_training" "$config_tag" "slurm"
        )
    fi

    echo "Starting training: seed=${seed} on GPU ${gpu} -> ${run_dir}"
    ml_train "${train_args[@]}" "${comet_args[@]}"
}

# ─── Post-processing: collect best models + ONNX aggregation ──────────────────
run_postproc() {
    if [[ "$K_FOLDS" -ge 2 ]]; then
        # out-of-fold scores, summary of the folds and combined ONNX models
        # the config of the folds is read from OUT_DIR/fold_0, with the ml_train
        # options used for the trainings
        local merge_args=(-i "$OUT_DIR" -k "$K_FOLDS")
        [[ "$RATIO" == true ]] && merge_args+=(--ratio)
        echo "Merging the ${K_FOLDS} folds in ${OUT_DIR}..."
        ml_kfold_merge "${merge_args[@]}"
        return
    fi

    local best_models_dir="${OUT_DIR}/best_models"
    mkdir -p "$best_models_dir"

    echo "Collecting best models -> ${best_models_dir}"
    for ((i=0; i<N_TRAININGS; i++)); do
        local seed=$(( INIT_SEED + i ))
        local tag
        tag=$(printf "%02d" "$seed")
        local model_dir
        model_dir="$(run_path "$seed")/state_dict"
        local best_model
        best_model=$(ls "$model_dir"/*best_epoch*.onnx 2>/dev/null | head -n 1)
        if [[ -n "$best_model" ]]; then
            cp "$best_model" "${best_models_dir}/best_model_run${tag}.onnx"
            echo "  run${tag}: copied $(basename "$best_model")"
        else
            echo "  Warning: no best model found in ${model_dir}"
        fi
    done

    echo "Running ONNX aggregation in ${best_models_dir}..."
    cd "$OUT_DIR" || { echo "Error: cannot cd to ${OUT_DIR}"; exit 1; }

    local onnx_args=(-i best_models -o best_models)
    [[ "$RATIO" == true ]] && onnx_args+=(-ar)

    ml_onnx "${onnx_args[@]}" --config "$CONFIG_FILE"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
echo "=== ML Training Script ==="
echo "  Config:     ${CONFIG_FILE}"
echo "  Output:     ${OUT_DIR}"
echo "  Trainings:  ${N_TRAININGS} (init seed: ${INIT_SEED})"
echo "  Nodes:      ${NODES}"
echo "  GPUs/node:  ${GPUS_PER_NODE}"
echo "  Mode:       ${MODE}"
echo "  Ratio:      ${RATIO}"
[[ "$K_FOLDS" -ge 2 ]] && echo "  K-folds:    ${K_FOLDS}"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && echo "  Array task: ${SLURM_ARRAY_TASK_ID}"

mkdir -p "$OUT_DIR"

case "$MODE" in
    train)
        TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
        NODE_START=$(( INIT_SEED + TASK_ID * TRAININGS_PER_NODE ))
        NODE_END=$(( NODE_START + TRAININGS_PER_NODE - 1 ))
        MAX_SEED=$(( INIT_SEED + N_TRAININGS - 1 ))
        [[ "$NODE_END" -gt "$MAX_SEED" ]] && NODE_END="$MAX_SEED"

        if [[ "$NODE_START" -gt "$MAX_SEED" ]]; then
            echo "Node ${TASK_ID}: no trainings assigned (N_TRAININGS=${N_TRAININGS}, NODES=${NODES}). Exiting."
            exit 0
        fi

        N_NODE_TRAININGS=$(( NODE_END - NODE_START + 1 ))

        # GPUs actually visible to this job (Slurm sets CUDA_VISIBLE_DEVICES
        # to the allocated ones, which are then numbered 0..N-1)
        N_GPUS="$GPUS_PER_NODE"
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
            N_GPUS=${#VISIBLE_GPUS[@]}
        fi
        [[ "$N_GPUS" -gt "$N_NODE_TRAININGS" ]] && N_GPUS="$N_NODE_TRAININGS"

        # Split the CPUs among the parallel trainings, so that the torch/numpy
        # threads of the different processes do not oversubscribe the cores
        N_CPUS="${SLURM_CPUS_PER_TASK:-$SLURM_CPUS}"
        CPUS_PER_TRAINING=$(( N_CPUS / N_NODE_TRAININGS ))
        [[ "$CPUS_PER_TRAINING" -lt 1 ]] && CPUS_PER_TRAINING=1
        export OMP_NUM_THREADS="$CPUS_PER_TRAINING"
        export MKL_NUM_THREADS="$CPUS_PER_TRAINING"
        # DataLoader workers, only used if the datasets do not fit on the GPU
        # (see data_on_gpu in the config)
        NUM_WORKERS=$(( CPUS_PER_TRAINING - 1 ))

        echo "Node ${TASK_ID}: launching seeds ${NODE_START}–${NODE_END} in parallel on ${N_GPUS} GPU(s), ${CPUS_PER_TRAINING} CPU(s) per training"

        for ((seed=NODE_START; seed<=NODE_END; seed++)); do
            gpu=$(( (seed - NODE_START) % N_GPUS ))
            run_one_training "$seed" "$(run_path "$seed")" "$gpu" "$NUM_WORKERS" &
        done
        wait
        echo "Node ${TASK_ID}: all trainings complete."

        # For single-node multi-training: run postproc inline (no separate job)
        if [[ "$N_TRAININGS" -gt 1 && "$NODES" -eq 1 ]]; then
            run_postproc
        fi
        ;;

    postproc)
        run_postproc
        ;;

    *)
        echo "Error: unknown mode '${MODE}'."
        exit 1
        ;;
esac

echo "Done."
