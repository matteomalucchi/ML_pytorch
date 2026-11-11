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
#   --no-kfold-cache        K-fold: every fold loads the events itself, instead
#                           of loading them once in a temporary cache
#   --save-roc              Save also the ROC points (tpr_fpr*.npz) of the
#                           trainings and of the k-fold merge
#   --alpha                 Boosted multiclass morphing: compute the ttbar
#                           fraction alpha with ml_alpha and use it in the ONNX
#                           conversion (ttbar subtraction)
#   --alpha-den-computed    Use also the computed alpha of the denominator
#                           region (default: alpha_den 0.0)
#   --load-last             Resume from latest checkpoint instead of restarting
#   --time HH:MM:SS         Slurm wall-clock time limit (default: 8:30:00)
#   --mem-per-cpu MB        Slurm memory per CPU in MB (default: 4000)
#   --cpus INT              Slurm CPUs per task (default: 4)
#   -g, --gpus-per-node INT GPUs per Slurm node/array job (default: 1). The
#                           trainings of a node are distributed round-robin
#                           over its GPUs (one process per training)
#   --no-slurm              Run directly without Slurm (for local testing)
#   -- EXTRA                Additional arguments forwarded to ml_train
#
# K-fold in several jobs (see the README, "K-fold in several jobs"): the events
# are loaded once by a CPU job into a shared cache, then the folds are trained
# by small GPU jobs that copy it to their node, then a CPU job merges them.
#   --prepare-job           Submit the CPU prepare job, an array of GPU jobs
#                           with --folds-per-job folds each, and the CPU merge
#                           job, chained with dependencies
#   --folds-per-job INT     Folds per GPU job with --prepare-job (default: 1)
#   --prepare-only          Submit only the CPU prepare job
#   --folds LIST            Submit one GPU job training only these folds (e.g.
#                           0-4 or 5,7,9) from the shared cache
#   --merge-only            Submit only the CPU merge job
#   --cache-dir DIR         Shared cache (default: OUTDIR/kfold_cache_shared),
#                           removed after the merge
#   --keep-cache            Do not remove the shared cache after the merge
#   --cpu-partition STR     Partition of the CPU jobs (default: standard)
#   --cpu-account STR       Account of the CPU jobs (default: t3)
#   --cpu-cpus INT          CPUs of the CPU jobs (default: 8)
#   --cpu-mem-per-cpu MB    Memory per CPU of the CPU jobs (default: --mem-per-cpu)
#   --cpu-time HH:MM:SS     Time limit of the CPU jobs (default: --time)
#   --prepare-gpu           Run the prepare job on the GPU partition (1 GPU,
#                           -p gpu --account=gpu_gres) instead of the CPU queue
#   --merge-gpu             Run the merge job on the GPU partition
#   --after JOBIDS          With --folds / --merge-only / --fold-jobs /
#                           --missing-folds: start only after these jobs
#                           (colon-separated) succeed, e.g. the prepare job
#   --fold-jobs             Submit the fold jobs (array, --folds-per-job folds
#                           each) and the merge after them, from an existing
#                           shared cache or after the prepare job (--after)
#   --missing-folds         Resubmit only the folds without a finished model
#                           (resuming from their checkpoints) and the merge
#   --slots-per-gpu INT     Run at most INT trainings per GPU at a time; the
#                           others start when a GPU frees up (default: 0 = all
#                           at once)
#   --separate-merge        Single job / -p array: merge in a separate CPU job
#                           after the training job, so the GPUs are released
#                           when the last training ends
#
# Resource monitor (jobs/resource_monitor.py): every job samples its CPU,
# memory and GPU usage (per job, per GPU and per training) and the time spent
# in every phase, and writes OUTDIR/resource_usage/<mode>_<jobid>[_<task>]_*
# (CSV time series + summary.txt/json with time occupancy, usage vs the
# requested resources and suggested --cpus/--mem-per-cpu/--time/-g). The
# summary is also printed at the end of the Slurm log.
#   --monitor-interval SEC  Seconds between samples (default: 10)
#   --no-monitor            Do not run the resource monitor

# ─── Defaults ─────────────────────────────────────────────────────────────────
N_TRAININGS=1
NODES=1
INIT_SEED=0
RATIO=false
K_FOLDS=0
KFOLD_CACHE=true
ALPHA=false
ALPHA_DEN_COMPUTED=false
SAVE_ROC=false
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
SPLIT_JOBS=false
PREPARE_ONLY=false
MERGE_ONLY=false
FOLDS=""
FOLDS_PER_JOB=1
SHARED_CACHE_DIR=""
KEEP_CACHE=false
CPU_PARTITION="standard"
CPU_ACCOUNT="t3"
CPU_CPUS=8
CPU_MEM=""
CPU_TIME=""
PREPARE_GPU=false
MERGE_GPU=false
AFTER=""
FOLD_JOBS=false
MISSING_FOLDS=false
SLOTS_PER_GPU=0
SEPARATE_MERGE=false
MONITOR=true
MONITOR_INTERVAL=10
# partition and account of the GPU jobs (as in the #SBATCH lines above)
GPU_PARTITION="gpu"
GPU_ACCOUNT="gpu_gres"

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
                          ml_kfold_merge (not combinable with -n > 1). The
                          events are loaded once in OUTDIR/kfold_cache and
                          memory-mapped by all the folds
  --no-kfold-cache        K-fold: every fold loads the events itself
  --save-roc              Save also the ROC points (tpr_fpr*.npz) of the
                          trainings and of the k-fold merge (off by default)
  --alpha                 Boosted 4-node multiclass morphing: compute the ttbar
                          fraction alpha = N_ttbar / N_data with ml_alpha (from
                          the data_dirs and the regions of the config, saved
                          in OUTDIR/alpha.json) and pass it to ml_onnx -mc
                          (or ml_kfold_merge) for the ttbar subtraction
  --alpha-den-computed    With --alpha: use also the computed alpha of the
                          denominator region (default alpha_den 0.0, i.e. no
                          ttbar subtraction in the denominator)
  --load-last             Resume from latest checkpoint
  --time HH:MM:SS         Slurm wall-clock time limit (default: 8:30:00)
  --mem-per-cpu MB        Slurm memory per CPU in MB (default: 4000)
  --cpus INT              Slurm CPUs per task (default: 4)
  -g, --gpus-per-node INT GPUs per node/array job (default: 1); the trainings
                          of a node are spread round-robin over its GPUs
  --no-slurm              Run directly without Slurm (for testing; forces NODES=1)
  -- EXTRA                Extra arguments forwarded to ml_train

K-fold in several jobs (the events are loaded once by a CPU job into a shared
cache, the folds are trained by small GPU jobs that copy it to their node, and
a CPU job merges them):
  --prepare-job           Submit the CPU prepare job, an array of GPU jobs with
                          --folds-per-job folds each (-g GPUs per job) and the
                          CPU merge job, chained with dependencies. The array
                          tasks start one by one when their GPUs are free
  --folds-per-job INT     Folds per GPU job with --prepare-job (default: 1)
  --prepare-only          Submit only the CPU prepare job
  --folds LIST            Submit one GPU job training only these folds (e.g.
                          0-4 or 5,7,9) from the shared cache; repeat it for
                          the other folds when resources are free
  --merge-only            Submit only the CPU merge job, after all the folds
  --cache-dir DIR         Shared cache, visible from all the nodes (default:
                          OUTDIR/kfold_cache_shared); removed after the merge
  --keep-cache            Do not remove the shared cache after the merge
  --cpu-partition STR     Partition of the CPU jobs (default: standard)
  --cpu-account STR       Account of the CPU jobs (default: t3)
  --cpu-cpus INT          CPUs of the CPU jobs (default: 8)
  --cpu-mem-per-cpu MB    Memory per CPU of the CPU jobs (default: --mem-per-cpu)
  --cpu-time HH:MM:SS     Time limit of the CPU jobs (default: --time)
  --prepare-gpu           Run the prepare job on the GPU partition (1 GPU,
                          -p gpu --account=gpu_gres) instead of the CPU queue,
                          with the --cpu-* resources; the GPU is not used
  --merge-gpu             Run the merge job on the GPU partition too
  --after JOBIDS          With --folds, --merge-only, --fold-jobs or
                          --missing-folds: start only after these jobs
                          (colon-separated ids) succeed, so they can be
                          submitted right after the prepare job (or, for the
                          merge, after the fold jobs)
  --fold-jobs             Submit all the fold jobs (an array with
                          --folds-per-job folds each, -g GPUs) and the merge
                          after them, using an existing shared cache or the
                          prepare job given with --after
  --missing-folds         Recovery: resubmit only the folds that have no
                          finished model yet (with --load-last, so interrupted
                          folds resume from their last checkpoint) and the
                          merge after them

Occupancy of a single job with several GPUs:
  --slots-per-gpu INT     Run at most INT trainings per GPU at a time; when one
                          ends, the next one starts on the GPU that freed up
                          (default: 0, all the trainings at once)
  --separate-merge        Merge (or post-process) in a separate CPU job after
                          the training job, instead of inside it, so the GPUs
                          are released as soon as the last training ends

Resource monitor (CPU, memory, GPU and time occupancy of every job, saved in
OUTDIR/resource_usage/, with a summary and suggested resources):
  --monitor-interval SEC  Seconds between samples (default: 10)
  --no-monitor            Do not run the resource monitor

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

  # 20-fold training in 20 jobs of 1 GPU each, started when GPUs are free
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --prepare-job -g 1 --cpus 4

  # The same by hand: prepare, then train the folds in pieces, then merge
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --prepare-only
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --folds 0-7 -g 8 --cpus 32
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --folds 8-19 -g 4 --cpus 24
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --merge-only

  # Prepare on a GPU node, then all the fold jobs and the merge, waiting for it
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --prepare-only --prepare-gpu   # -> job id P
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --fold-jobs --after P -g 1

  # Some folds failed: resubmit only them (resuming) and the merge
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 --missing-folds -g 1

  # One big job: at most 2 folds per GPU at a time, merge in a separate CPU job
  $0 -c /full/path/DNN_config.yml -o /full/path/out -k 20 -g 8 --cpus 40 --slots-per-gpu 2 --separate-merge

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
        --no-kfold-cache)  KFOLD_CACHE=false; shift ;;
        --alpha)           ALPHA=true; shift ;;
        --save-roc)        SAVE_ROC=true; shift ;;
        --alpha-den-computed) ALPHA_DEN_COMPUTED=true; shift ;;
        --load-last)       LOAD_LAST=true; shift ;;
        --time)            SLURM_TIME="$2"; shift 2 ;;
        --mem-per-cpu)     SLURM_MEM="$2"; shift 2 ;;
        --cpus)            SLURM_CPUS="$2"; shift 2 ;;
        -g|--gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --no-slurm)        NO_SLURM=true; shift ;;
        --prepare-job)     SPLIT_JOBS=true; shift ;;
        --folds-per-job)   FOLDS_PER_JOB="$2"; shift 2 ;;
        --prepare-only)    PREPARE_ONLY=true; shift ;;
        --folds)           FOLDS="$2"; shift 2 ;;
        --merge-only)      MERGE_ONLY=true; shift ;;
        --cache-dir)       SHARED_CACHE_DIR="$2"; shift 2 ;;
        --keep-cache)      KEEP_CACHE=true; shift ;;
        --cpu-partition)   CPU_PARTITION="$2"; shift 2 ;;
        --cpu-account)     CPU_ACCOUNT="$2"; shift 2 ;;
        --cpu-cpus)        CPU_CPUS="$2"; shift 2 ;;
        --cpu-mem-per-cpu) CPU_MEM="$2"; shift 2 ;;
        --cpu-time)        CPU_TIME="$2"; shift 2 ;;
        --prepare-gpu)     PREPARE_GPU=true; shift ;;
        --merge-gpu)       MERGE_GPU=true; shift ;;
        --after)           AFTER="$2"; shift 2 ;;
        --fold-jobs)       FOLD_JOBS=true; shift ;;
        --missing-folds)   MISSING_FOLDS=true; shift ;;
        --slots-per-gpu)   SLOTS_PER_GPU="$2"; shift 2 ;;
        --separate-merge)  SEPARATE_MERGE=true; shift ;;
        --monitor-interval) MONITOR_INTERVAL="$2"; shift 2 ;;
        --no-monitor)      MONITOR=false; shift ;;
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

# K-fold cache: the events are loaded once and memory-mapped by all the folds
USE_KFOLD_CACHE=false
[[ "$K_FOLDS" -ge 2 && "$KFOLD_CACHE" == true ]] && USE_KFOLD_CACHE=true
# The cache holds a copy of all the events: it is written to the node-local
# temporary directory of each job (one per array task), not to OUT_DIR, and
# removed when the job ends. Only the record of the split (k.npy and the
# metadata) is kept, in OUT_DIR/kfold_split.
KFOLD_CACHE_DIR="${TMPDIR:-/tmp}/ml_kfold_cache_${SLURM_JOB_ID:-$$}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
KFOLD_SPLIT_DIR="${OUT_DIR}/kfold_split"

# ─── K-fold in several jobs ──────────────────────────────────────────────────
[[ -z "$SHARED_CACHE_DIR" ]] && SHARED_CACHE_DIR="${OUT_DIR}/kfold_cache_shared"
[[ -z "$CPU_MEM" ]] && CPU_MEM="$SLURM_MEM"
[[ -z "$CPU_TIME" ]] && CPU_TIME="$SLURM_TIME"
N_SPLIT_OPTIONS=0
for opt in "$SPLIT_JOBS" "$PREPARE_ONLY" "$MERGE_ONLY" "$FOLD_JOBS" "$MISSING_FOLDS"; do
    [[ "$opt" == true ]] && N_SPLIT_OPTIONS=$(( N_SPLIT_OPTIONS + 1 ))
done
[[ -n "$FOLDS" ]] && N_SPLIT_OPTIONS=$(( N_SPLIT_OPTIONS + 1 ))
if [[ "$N_SPLIT_OPTIONS" -gt 1 ]]; then
    echo "Error: use only one of --prepare-job, --prepare-only, --folds, --merge-only, --fold-jobs and --missing-folds."
    exit 1
fi
if [[ "$N_SPLIT_OPTIONS" -eq 1 ]]; then
    if [[ "$K_FOLDS" -lt 2 || "$KFOLD_CACHE" != true ]]; then
        echo "Error: --prepare-job, --prepare-only, --folds, --merge-only, --fold-jobs and --missing-folds need --k-folds >= 2 and the k-fold cache."
        exit 1
    fi
    if [[ "$NODES" -ne 1 ]]; then
        echo "Error: -p/--nodes cannot be combined with the k-fold split options (use --folds-per-job)."
        exit 1
    fi
fi
if [[ -n "$AFTER" && -z "$FOLDS" && "$MERGE_ONLY" != true && "$FOLD_JOBS" != true && "$MISSING_FOLDS" != true ]]; then
    echo "Error: --after can be used only with --folds, --merge-only, --fold-jobs or --missing-folds."
    exit 1
fi
if [[ -n "$AFTER" && ! "$AFTER" =~ ^[0-9_]+(:[0-9_]+)*$ ]]; then
    echo "Error: --after needs job ids separated by colons, e.g. 1234 or 1234:1235."
    exit 1
fi
if [[ ! "$SLOTS_PER_GPU" =~ ^[0-9]+$ ]]; then
    echo "Error: --slots-per-gpu must be an integer >= 0."
    exit 1
fi
if [[ ! "$MONITOR_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk "BEGIN{exit !($MONITOR_INTERVAL > 0)}"; then
    echo "Error: --monitor-interval must be a number of seconds > 0."
    exit 1
fi
if [[ "$FOLDS_PER_JOB" -lt 1 ]]; then
    echo "Error: --folds-per-job must be >= 1."
    exit 1
fi

# Folds of a list like "0-4,7,9-10" (sorted, unique), checked against K_FOLDS
parse_folds() {
    local item start end f out=()
    IFS=',' read -r -a items <<< "$1"
    for item in "${items[@]}"; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start="${BASH_REMATCH[1]}"; end="${BASH_REMATCH[2]}"
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            start="$item"; end="$item"
        else
            echo "Error: invalid fold list '$1'." >&2; return 1
        fi
        for ((f=start; f<=end; f++)); do
            if [[ "$f" -ge "$K_FOLDS" ]]; then
                echo "Error: fold $f is not in 0 ... $(( K_FOLDS - 1 ))." >&2; return 1
            fi
            out+=("$f")
        done
    done
    [[ ${#out[@]} -eq 0 ]] && { echo "Error: empty fold list '$1'." >&2; return 1; }
    printf '%s\n' "${out[@]}" | sort -n -u | tr '\n' ' '
}
if [[ -n "$FOLDS" ]]; then
    FOLD_LIST=$(parse_folds "$FOLDS") || exit 1
fi
N_FOLD_JOBS=$(( (K_FOLDS + FOLDS_PER_JOB - 1) / FOLDS_PER_JOB ))


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
    [[ "$KFOLD_CACHE" != true ]] && args+=(--no-kfold-cache)
    [[ "$ALPHA" == true ]] && args+=(--alpha)
    [[ "$SAVE_ROC" == true ]] && args+=(--save-roc)
    [[ "$ALPHA_DEN_COMPUTED" == true ]] && args+=(--alpha-den-computed)
    args+=(--folds-per-job "$FOLDS_PER_JOB" --cache-dir "$SHARED_CACHE_DIR")
    [[ "$KEEP_CACHE" == true ]] && args+=(--keep-cache)
    [[ -n "$FOLDS" ]] && args+=(--folds "$FOLDS")
    args+=(--slots-per-gpu "$SLOTS_PER_GPU")
    [[ "$SEPARATE_MERGE" == true ]] && args+=(--separate-merge)
    args+=(--monitor-interval "$MONITOR_INTERVAL")
    [[ "$MONITOR" != true ]] && args+=(--no-monitor)
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

    # CPU jobs (prepare and merge of the k-fold in several jobs): submitted with
    # --wrap, so the GPU #SBATCH lines of this script are not applied
    # (with on_gpu=true on the GPU partition with 1 GPU, e.g. when the CPU queue
    # is busy or its nodes have too little memory)
    submit_cpu_job() {
        local job_name="$1" mode="$2" dependency="$3" on_gpu="${4:-false}"
        local mode_args wrapped
        mapfile -d '' mode_args < <(build_submit_args "$mode")
        wrapped="bash $(printf '%q ' "$SCRIPT_PATH" "${mode_args[@]}")"
        local dep_args=()
        [[ -n "$dependency" ]] && dep_args=(--dependency="afterok:${dependency}")
        local queue_args=(--partition="$CPU_PARTITION" --account="$CPU_ACCOUNT")
        [[ "$on_gpu" == true ]] && queue_args=(
            --partition="$GPU_PARTITION" --account="$GPU_ACCOUNT" --gres="gpu:1"
        )
        sbatch --parsable \
            --job-name="$job_name" \
            "${queue_args[@]}" \
            --output="${SLURM_LOG_DIR}/slurm-%j.out" \
            --time="$CPU_TIME" \
            --mem-per-cpu="$CPU_MEM" \
            --cpus-per-task="$CPU_CPUS" \
            "${dep_args[@]}" \
            --wrap="$wrapped"
    }
    # GPU jobs training folds from the shared cache
    # (n_array_tasks: an array of that many jobs, each with --folds-per-job of
    # the folds of --folds; by default an array over all the folds without
    # --folds, a single job with --folds)
    submit_fold_jobs() {
        local dependency="$1" n_array_tasks="${2:-}" array_args=()
        local fold_args
        mapfile -d '' fold_args < <(build_submit_args train_folds)
        local dep_args=()
        [[ -n "$dependency" ]] && dep_args=(--dependency="afterok:${dependency}")
        local output="${SLURM_LOG_DIR}/slurm-%j.out"
        [[ -z "$n_array_tasks" && -z "$FOLDS" ]] && n_array_tasks="$N_FOLD_JOBS"
        if [[ -n "$n_array_tasks" ]]; then
            array_args=(--array="0-$(( n_array_tasks - 1 ))")
            output="${SLURM_LOG_DIR}/slurm-%A_%a.out"
        fi
        sbatch --parsable \
            --job-name="ml_train_folds" \
            "${array_args[@]}" \
            "${dep_args[@]}" \
            --output="$output" \
            --gres="gpu:${GPUS_PER_NODE}" \
            "${SBATCH_RESOURCES[@]}" \
            "$SCRIPT_PATH" "${fold_args[@]}"
    }

    queue_name() { [[ "$1" == true ]] && echo "${GPU_PARTITION}, 1 GPU" || echo "${CPU_PARTITION}, no GPU"; }

    if [[ "$SPLIT_JOBS" == true ]]; then
        PREPARE_JOB_ID=$(submit_cpu_job ml_kfold_prepare prepare_shared "" "$PREPARE_GPU") || exit 1
        echo "K-fold prepare job ($(queue_name "$PREPARE_GPU")): ${PREPARE_JOB_ID} -> ${SHARED_CACHE_DIR}"
        TRAIN_JOB_ID=$(submit_fold_jobs "$PREPARE_JOB_ID") || exit 1
        echo "Fold jobs: array ${TRAIN_JOB_ID}, ${N_FOLD_JOBS} job(s) of ${FOLDS_PER_JOB} fold(s) on ${GPUS_PER_NODE} GPU(s) each, after ${PREPARE_JOB_ID}"
        MERGE_JOB_ID=$(submit_cpu_job ml_kfold_merge merge_shared "$TRAIN_JOB_ID" "$MERGE_GPU") || exit 1
        echo "K-fold merge job ($(queue_name "$MERGE_GPU")): ${MERGE_JOB_ID}, after ${TRAIN_JOB_ID}"
        echo "If a fold job fails, the merge does not start (cancel it with: scancel ${MERGE_JOB_ID}) and"
        echo "  $0 -c $CONFIG_FILE -o $OUT_DIR -k $K_FOLDS --missing-folds -g ${GPUS_PER_NODE}"
        echo "resubmits only the unfinished folds and the merge."
        exit 0
    elif [[ "$PREPARE_ONLY" == true ]]; then
        PREPARE_JOB_ID=$(submit_cpu_job ml_kfold_prepare prepare_shared "" "$PREPARE_GPU") || exit 1
        echo "K-fold prepare job ($(queue_name "$PREPARE_GPU")): ${PREPARE_JOB_ID} -> ${SHARED_CACHE_DIR}"
        echo "Submit now the fold jobs and the merge (they start when the prepare job has"
        echo "succeeded and their GPUs are free), e.g. one job per fold on 1 GPU:"
        echo "  $0 -c $CONFIG_FILE -o $OUT_DIR -k $K_FOLDS --fold-jobs --after ${PREPARE_JOB_ID} -g 1 --cpus <CPUs> --mem-per-cpu <MB>"
        exit 0
    elif [[ "$FOLD_JOBS" == true ]]; then
        if [[ -z "$AFTER" && ! -f "${SHARED_CACHE_DIR}/kfold_cache_meta.json" ]]; then
            echo "Error: no shared k-fold cache in ${SHARED_CACHE_DIR}: give the prepare job with --after <id>, or run --prepare-only first."
            exit 1
        fi
        TRAIN_JOB_ID=$(submit_fold_jobs "$AFTER") || exit 1
        echo "Fold jobs: array ${TRAIN_JOB_ID}, ${N_FOLD_JOBS} job(s) of ${FOLDS_PER_JOB} fold(s) on ${GPUS_PER_NODE} GPU(s) each${AFTER:+, after ${AFTER}}"
        MERGE_JOB_ID=$(submit_cpu_job ml_kfold_merge merge_shared "$TRAIN_JOB_ID" "$MERGE_GPU") || exit 1
        echo "K-fold merge job ($(queue_name "$MERGE_GPU")): ${MERGE_JOB_ID}, after ${TRAIN_JOB_ID}"
        echo "If a fold job fails, the merge does not start (cancel it with: scancel ${MERGE_JOB_ID}) and"
        echo "  $0 -c $CONFIG_FILE -o $OUT_DIR -k $K_FOLDS --missing-folds -g ${GPUS_PER_NODE}"
        echo "resubmits only the unfinished folds and the merge."
        exit 0
    elif [[ "$MISSING_FOLDS" == true ]]; then
        MISSING=()
        for ((f=0; f<K_FOLDS; f++)); do
            shopt -s nullglob
            finished=("${OUT_DIR}/fold_${f}/state_dict/"model_best_epoch_*.onnx)
            shopt -u nullglob
            [[ ${#finished[@]} -eq 0 ]] && MISSING+=("$f")
        done
        if [[ ${#MISSING[@]} -eq 0 ]]; then
            echo "All the ${K_FOLDS} folds are finished: submitting only the merge."
            MERGE_JOB_ID=$(submit_cpu_job ml_kfold_merge merge_shared "$AFTER" "$MERGE_GPU") || exit 1
            echo "K-fold merge job ($(queue_name "$MERGE_GPU")): ${MERGE_JOB_ID}${AFTER:+, after ${AFTER}}"
            exit 0
        fi
        if [[ -z "$AFTER" && ! -f "${SHARED_CACHE_DIR}/kfold_cache_meta.json" ]]; then
            echo "Error: the folds ${MISSING[*]} are not finished but there is no shared k-fold cache in ${SHARED_CACHE_DIR}."
            echo "Prepare it again (same config and seed, the split is identical) and resubmit after it:"
            echo "  $0 -c $CONFIG_FILE -o $OUT_DIR -k $K_FOLDS --prepare-only   # -> job id P"
            echo "  $0 -c $CONFIG_FILE -o $OUT_DIR -k $K_FOLDS --missing-folds --after P -g ${GPUS_PER_NODE}"
            exit 1
        fi
        echo "Unfinished folds: ${MISSING[*]} (they resume from their last checkpoint)"
        FOLDS=$(IFS=,; echo "${MISSING[*]}")
        LOAD_LAST=true
        N_MISSING_JOBS=$(( (${#MISSING[@]} + FOLDS_PER_JOB - 1) / FOLDS_PER_JOB ))
        TRAIN_JOB_ID=$(submit_fold_jobs "$AFTER" "$N_MISSING_JOBS") || exit 1
        echo "Fold jobs: array ${TRAIN_JOB_ID}, ${N_MISSING_JOBS} job(s) of ${FOLDS_PER_JOB} fold(s) on ${GPUS_PER_NODE} GPU(s) each${AFTER:+, after ${AFTER}}"
        MERGE_JOB_ID=$(submit_cpu_job ml_kfold_merge merge_shared "$TRAIN_JOB_ID" "$MERGE_GPU") || exit 1
        echo "K-fold merge job ($(queue_name "$MERGE_GPU")): ${MERGE_JOB_ID}, after ${TRAIN_JOB_ID}"
        exit 0
    elif [[ -n "$FOLDS" ]]; then
        if [[ -z "$AFTER" && ! -f "${SHARED_CACHE_DIR}/kfold_cache_meta.json" ]]; then
            echo "Warning: the shared cache ${SHARED_CACHE_DIR} is not ready yet: the job will fail if it is not when it starts (run --prepare-only first, or use --after <prepare job id>)."
        fi
        TRAIN_JOB_ID=$(submit_fold_jobs "$AFTER") || exit 1
        echo "Fold job ${TRAIN_JOB_ID}: folds ${FOLD_LIST}on ${GPUS_PER_NODE} GPU(s)${AFTER:+, after ${AFTER}}"
        exit 0
    elif [[ "$MERGE_ONLY" == true ]]; then
        MERGE_JOB_ID=$(submit_cpu_job ml_kfold_merge merge_shared "$AFTER" "$MERGE_GPU") || exit 1
        echo "K-fold merge job ($(queue_name "$MERGE_GPU")): ${MERGE_JOB_ID}${AFTER:+, after ${AFTER}}"
        exit 0
    fi

    if [[ "$NODES" -eq 1 ]]; then
        # Single job: all trainings on one node, postproc runs inline
        mapfile -d '' TRAIN_ARGS < <(build_submit_args train)
        echo "Submitting single job (${N_TRAININGS} training(s) on ${GPUS_PER_NODE} GPU(s))..."
        TRAIN_JOB_ID=$(sbatch --parsable --job-name="ml_train" \
            --output="${SLURM_LOG_DIR}/slurm-%j.out" \
            --gres="gpu:${GPUS_PER_NODE}" \
            "${SBATCH_RESOURCES[@]}" "$SCRIPT_PATH" "${TRAIN_ARGS[@]}") || exit 1
        echo "  Training job submitted: ${TRAIN_JOB_ID}"
        if [[ "$SEPARATE_MERGE" == true && "$N_TRAININGS" -gt 1 ]]; then
            MERGE_MODE=postproc
            [[ "$K_FOLDS" -ge 2 ]] && MERGE_MODE=merge_shared
            MERGE_JOB_ID=$(submit_cpu_job ml_postproc "$MERGE_MODE" "$TRAIN_JOB_ID" "$MERGE_GPU") || exit 1
            echo "  Merge job ($(queue_name "$MERGE_GPU")): ${MERGE_JOB_ID}, after ${TRAIN_JOB_ID}"
        fi
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

        if [[ "$SEPARATE_MERGE" == true ]]; then
            MERGE_MODE=postproc
            [[ "$K_FOLDS" -ge 2 ]] && MERGE_MODE=merge_shared
            POSTPROC_JOB_ID=$(submit_cpu_job ml_postproc "$MERGE_MODE" "$TRAIN_JOB_ID" "$MERGE_GPU") || exit 1
            echo "  Post-processing job ($(queue_name "$MERGE_GPU")): ${POSTPROC_JOB_ID}, after ${TRAIN_JOB_ID}"
            exit 0
        fi
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
    # restart from scratch if there is no checkpoint to resume from (e.g. a
    # fold that never started), without the interactive overwrite question
    [[ "$LOAD_LAST" != true || ${#load_model_args[@]} -eq 0 ]] && train_args+=(--overwrite)
    [[ "$SAVE_ROC" == true ]] && train_args+=(--save-roc)
    if [[ "$K_FOLDS" -ge 2 ]]; then
        train_args+=(-s "$INIT_SEED" --k-folds "$K_FOLDS" --fold "$(( seed - INIT_SEED ))")
        [[ "$USE_KFOLD_CACHE" == true ]] && train_args+=(--kfold-cache "$KFOLD_CACHE_DIR")
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

# ─── K-fold cache: load the events once for all the folds ────────────────────
# (in the temporary directory of the job, removed when the job ends, unless
# "keep" is given: the shared cache of the k-fold in several jobs)
run_prepare() {
    local cache_dir="${1:-$KFOLD_CACHE_DIR}" keep="${2:-}"
    # all the CPUs of the job for loading the events
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$SLURM_CPUS}"
    export MKL_NUM_THREADS="$OMP_NUM_THREADS"
    local prepare_args=(
        -c "$CONFIG_FILE" -s "$INIT_SEED" --k-folds "$K_FOLDS"
        --kfold-cache "$cache_dir" --kfold-split-dir "$KFOLD_SPLIT_DIR"
        --kfold-prepare
    )
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && prepare_args+=("${EXTRA_ARGS[@]}")
    echo "Preparing the k-fold cache in ${cache_dir}..."
    mark_phase kfold_prepare
    # remove the cache when the job ends, also if it fails (see on_exit)
    [[ "$keep" != keep ]] && REMOVE_ON_EXIT="$KFOLD_CACHE_DIR"
    ml_train "${prepare_args[@]}" || { echo "Error: k-fold cache preparation failed"; exit 1; }
}

# Copy the shared cache to the temporary directory of this job (read locally,
# not from the network filesystem), removed when the job ends
copy_shared_cache() {
    if [[ ! -f "${SHARED_CACHE_DIR}/kfold_cache_meta.json" ]]; then
        echo "Error: no k-fold cache in ${SHARED_CACHE_DIR}: run run_training.sh --prepare-only first."
        exit 1
    fi
    REMOVE_ON_EXIT="$KFOLD_CACHE_DIR"
    mark_phase copy_cache
    echo "Copying the k-fold cache ${SHARED_CACHE_DIR} -> ${KFOLD_CACHE_DIR}..."
    mkdir -p "$KFOLD_CACHE_DIR"
    cp -r "${SHARED_CACHE_DIR}/." "$KFOLD_CACHE_DIR/" || { echo "Error: copying the k-fold cache failed"; exit 1; }
}

# Train the trainings (seeds, or folds INIT_SEED + fold) given as arguments
# in parallel, spread over the GPUs and CPUs of this job
launch_trainings() {
    local seeds=("$@")
    local n_trainings=${#seeds[@]}
    mark_phase train

    # GPUs actually visible to this job (Slurm sets CUDA_VISIBLE_DEVICES
    # to the allocated ones, which are then numbered 0..N-1)
    local n_gpus="$GPUS_PER_NODE"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
        n_gpus=${#VISIBLE_GPUS[@]}
    fi
    [[ "$n_gpus" -gt "$n_trainings" ]] && n_gpus="$n_trainings"

    # with --slots-per-gpu, at most that many trainings per GPU at a time
    local max_parallel="$n_trainings"
    if [[ "$SLOTS_PER_GPU" -gt 0 && $(( n_gpus * SLOTS_PER_GPU )) -lt "$n_trainings" ]]; then
        max_parallel=$(( n_gpus * SLOTS_PER_GPU ))
    fi

    # Split the CPUs among the parallel trainings, so that the torch/numpy
    # threads of the different processes do not oversubscribe the cores
    local n_cpus="${SLURM_CPUS_PER_TASK:-$SLURM_CPUS}"
    local cpus_per_training=$(( n_cpus / max_parallel ))
    [[ "$cpus_per_training" -lt 1 ]] && cpus_per_training=1
    export OMP_NUM_THREADS="$cpus_per_training"
    export MKL_NUM_THREADS="$cpus_per_training"
    # DataLoader workers, only used if the datasets do not fit on the GPU
    # (see data_on_gpu in the config)
    local num_workers=$(( cpus_per_training - 1 ))

    local i
    if [[ "$max_parallel" -ge "$n_trainings" ]]; then
        echo "Launching ${n_trainings} training(s) (seeds/folds ${seeds[*]}) in parallel on ${n_gpus} GPU(s), ${cpus_per_training} CPU(s) per training"
        for ((i=0; i<n_trainings; i++)); do
            run_one_training "${seeds[$i]}" "$(run_path "${seeds[$i]}")" "$(( i % n_gpus ))" "$num_workers" &
        done
        wait
        return
    fi

    # Dynamic scheduling: at most SLOTS_PER_GPU trainings per GPU; when one
    # ends, the next training starts on the GPU with the fewest running ones,
    # so no GPU stays idle while trainings are waiting
    echo "Launching ${n_trainings} training(s) (seeds/folds ${seeds[*]}) on ${n_gpus} GPU(s), at most ${SLOTS_PER_GPU} per GPU at a time (${max_parallel} in parallel), ${cpus_per_training} CPU(s) per training"
    local -a gpu_running=() pids=() pid_gpu=()
    local g best pid k
    for ((g=0; g<n_gpus; g++)); do gpu_running[$g]=0; done
    for ((i=0; i<n_trainings; i++)); do
        while [[ ${#pids[@]} -ge "$max_parallel" ]]; do
            wait -n 2>/dev/null
            # free the slots of the trainings that have ended
            for k in "${!pids[@]}"; do
                if ! kill -0 "${pids[$k]}" 2>/dev/null; then
                    wait "${pids[$k]}" 2>/dev/null
                    gpu_running[${pid_gpu[$k]}]=$(( gpu_running[${pid_gpu[$k]}] - 1 ))
                    unset 'pids[k]' 'pid_gpu[k]'
                fi
            done
        done
        best=0
        for ((g=1; g<n_gpus; g++)); do
            [[ ${gpu_running[$g]} -lt ${gpu_running[$best]} ]] && best=$g
        done
        echo "Starting seed/fold ${seeds[$i]} on GPU ${best} ($(( gpu_running[best] + 1 )) running on it)"
        run_one_training "${seeds[$i]}" "$(run_path "${seeds[$i]}")" "$best" "$num_workers" &
        pid=$!
        pids[$i]=$pid
        pid_gpu[$i]=$best
        gpu_running[$best]=$(( gpu_running[best] + 1 ))
    done
    wait
}

# ─── Post-processing: collect best models + ONNX aggregation ──────────────────
# ─── Boosted morphing: ttbar fraction alpha from the PocketCoffea outputs ─────
compute_alpha() {
    mark_phase alpha
    local values
    values=$(ml_alpha -c "$CONFIG_FILE" --print --json "${OUT_DIR}/alpha.json") \
        || { echo "Error: ml_alpha failed"; exit 1; }
    read -r ALPHA_NUM ALPHA_DEN_VALUE <<< "$values"
    [[ "$ALPHA_DEN_COMPUTED" != true ]] && ALPHA_DEN_VALUE=0.0
    echo "ttbar fraction: alpha ${ALPHA_NUM}, alpha_den ${ALPHA_DEN_VALUE} (table in ${OUT_DIR}/alpha.json)"
}

run_postproc() {
    [[ "$ALPHA" == true ]] && compute_alpha
    mark_phase postproc
    if [[ "$K_FOLDS" -ge 2 ]]; then
        # out-of-fold scores, summary of the folds and combined ONNX models
        # the config of the folds is read from OUT_DIR/fold_0, with the ml_train
        # options used for the trainings
        local merge_args=(-i "$OUT_DIR" -k "$K_FOLDS")
        [[ "$RATIO" == true ]] && merge_args+=(--ratio)
        [[ "$SAVE_ROC" == true ]] && merge_args+=(--save-roc)
        [[ "$ALPHA" == true ]] && merge_args+=(--alpha "$ALPHA_NUM" --alpha-den "$ALPHA_DEN_VALUE")
        # the cache of this job (or the shared one), if there is one (otherwise
        # the merge loads the events again into a temporary directory)
        local merge_cache="${MERGE_CACHE_DIR:-$KFOLD_CACHE_DIR}"
        [[ "$USE_KFOLD_CACHE" == true && -d "$merge_cache" ]] && merge_args+=(--kfold-cache "$merge_cache")
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
    # multiclass morphing with the ttbar subtraction (average of the ratios)
    [[ "$ALPHA" == true ]] && onnx_args+=(-ar -mc -al "$ALPHA_NUM" -ald "$ALPHA_DEN_VALUE")

    ml_onnx "${onnx_args[@]}" --config "$CONFIG_FILE"
}

# ─── Resource monitor ─────────────────────────────────────────────────────────
# resource_monitor.py samples the CPU/memory/GPU usage of this job in the
# background and writes MONITOR_DIR/<tag>_* (see the header); the phases of
# the job (prepare, train, postproc, ...) are marked in <tag>_phases.txt
MONITOR_DIR="${OUT_DIR}/resource_usage"
MONITOR_PID=""
MONITOR_PHASE_FILE=""
REMOVE_ON_EXIT=""

mark_phase() {
    [[ -n "$MONITOR_PHASE_FILE" ]] && echo "$(date +%s) $*" >> "$MONITOR_PHASE_FILE"
    return 0
}

start_monitor() {
    [[ "$MONITOR" != true ]] && return
    local job="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local$$}}"
    MONITOR_TAG="${MODE}_${job}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
    mkdir -p "$MONITOR_DIR"
    MONITOR_PHASE_FILE="${MONITOR_DIR}/${MONITOR_TAG}_phases.txt"
    : > "$MONITOR_PHASE_FILE"
    python "${SCRIPT_DIR}/resource_monitor.py" --pid $$ \
        --outdir "$MONITOR_DIR" --tag "$MONITOR_TAG" \
        --interval "$MONITOR_INTERVAL" --phase-file "$MONITOR_PHASE_FILE" \
        --cpus "${SLURM_CPUS_PER_TASK:-$SLURM_CPUS}" \
        --mem-per-cpu "${SLURM_MEM_PER_CPU:-$SLURM_MEM}" \
        > "${MONITOR_DIR}/${MONITOR_TAG}_monitor.log" 2>&1 &
    MONITOR_PID=$!
    # not a job of this shell: the "wait" of launch_trainings must not wait for it
    disown "$MONITOR_PID"
    echo "Resource monitor (every ${MONITOR_INTERVAL} s) -> ${MONITOR_DIR}/${MONITOR_TAG}_*"
}

stop_monitor() {
    [[ -z "$MONITOR_PID" ]] && return
    mark_phase end "$1"
    kill -TERM "$MONITOR_PID" 2>/dev/null
    local i
    for ((i=0; i<120; i++)); do
        kill -0 "$MONITOR_PID" 2>/dev/null || break
        sleep 0.5
    done
    MONITOR_PID=""
    local summary="${MONITOR_DIR}/${MONITOR_TAG}_summary.txt"
    if [[ -f "$summary" ]]; then
        echo
        echo "=== Resource usage (${summary}) ==="
        cat "$summary"
    fi
}

# at the end of the job, also if it fails or is killed: stop the monitor
# (final summary) and remove the node-local k-fold cache
on_exit() {
    local rc=$?
    stop_monitor "$rc"
    if [[ -n "$REMOVE_ON_EXIT" ]]; then
        echo "Removing the k-fold cache ${REMOVE_ON_EXIT}"
        rm -rf "$REMOVE_ON_EXIT"
    fi
    exit "$rc"
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

# --no-slurm with the k-fold split options: run the corresponding step here
if [[ "$NO_SLURM" == true && "$MODE" == train ]]; then
    if [[ "$SPLIT_JOBS" == true ]]; then MODE=split_local
    elif [[ "$PREPARE_ONLY" == true ]]; then MODE=prepare_shared
    elif [[ -n "$FOLDS" ]]; then MODE=train_folds
    elif [[ "$MERGE_ONLY" == true ]]; then MODE=merge_shared
    elif [[ "$FOLD_JOBS" == true ]]; then
        FOLD_LIST=$(seq 0 $(( K_FOLDS - 1 )) | tr '\n' ' '); MODE=train_folds_merge
    elif [[ "$MISSING_FOLDS" == true ]]; then
        FOLD_LIST=""
        for ((f=0; f<K_FOLDS; f++)); do
            shopt -s nullglob
            finished=("${OUT_DIR}/fold_${f}/state_dict/"model_best_epoch_*.onnx)
            shopt -u nullglob
            [[ ${#finished[@]} -eq 0 ]] && FOLD_LIST+="$f "
        done
        echo "Unfinished folds: ${FOLD_LIST:-none}"
        LOAD_LAST=true; MODE=train_folds_merge
    fi
fi

trap on_exit EXIT
start_monitor

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

        # prepare the k-fold cache in the temporary directory of this job before
        # starting its folds (every array task prepares its own, on its node)
        [[ "$USE_KFOLD_CACHE" == true ]] && run_prepare

        launch_trainings $(seq "$NODE_START" "$NODE_END")
        echo "Node ${TASK_ID}: all trainings complete."

        # For single-node multi-training: run postproc inline (no separate job)
        if [[ "$N_TRAININGS" -gt 1 && "$NODES" -eq 1 && ( "$SEPARATE_MERGE" != true || "$NO_SLURM" == true ) ]]; then
            run_postproc
        fi
        ;;

    postproc)
        run_postproc
        ;;

    prepare_shared)
        # k-fold in several jobs: load the events once into the shared cache
        mkdir -p "$(dirname "$SHARED_CACHE_DIR")"
        run_prepare "$SHARED_CACHE_DIR" keep
        echo "Shared k-fold cache ready in ${SHARED_CACHE_DIR}"
        ;;

    train_folds)
        # k-fold in several jobs: train the folds given with --folds, or the
        # folds of this array task, from a local copy of the shared cache
        if [[ -n "$FOLDS" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
            # array over the folds of --folds (e.g. --missing-folds)
            ALL_FOLDS=($FOLD_LIST)
            JOB_FOLDS=("${ALL_FOLDS[@]:$(( SLURM_ARRAY_TASK_ID * FOLDS_PER_JOB )):${FOLDS_PER_JOB}}")
        elif [[ -n "$FOLDS" ]]; then
            JOB_FOLDS=($FOLD_LIST)
        else
            TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
            FIRST=$(( TASK_ID * FOLDS_PER_JOB ))
            LAST=$(( FIRST + FOLDS_PER_JOB - 1 ))
            [[ "$LAST" -ge "$K_FOLDS" ]] && LAST=$(( K_FOLDS - 1 ))
            JOB_FOLDS=($(seq "$FIRST" "$LAST"))
        fi
        copy_shared_cache
        launch_trainings $(for f in "${JOB_FOLDS[@]}"; do echo $(( INIT_SEED + f )); done)
        echo "Folds ${JOB_FOLDS[*]} complete."
        ;;

    train_folds_merge)
        # --fold-jobs / --missing-folds with --no-slurm: the folds, then the merge
        if [[ -n "$FOLD_LIST" ]]; then
            copy_shared_cache
            launch_trainings $(for f in $FOLD_LIST; do echo $(( INIT_SEED + f )); done)
        fi
        [[ -d "$SHARED_CACHE_DIR" ]] && MERGE_CACHE_DIR="$SHARED_CACHE_DIR"
        run_postproc || exit 1
        if [[ "$KEEP_CACHE" != true && -d "$SHARED_CACHE_DIR" ]]; then
            echo "Removing the shared k-fold cache ${SHARED_CACHE_DIR}"
            rm -rf "$SHARED_CACHE_DIR"
        fi
        ;;

    merge_shared)
        # k-fold in several jobs: merge the folds, with the shared cache (if
        # it is not there, the merge loads the events again)
        [[ -d "$SHARED_CACHE_DIR" ]] && MERGE_CACHE_DIR="$SHARED_CACHE_DIR"
        run_postproc || exit 1
        if [[ "$KEEP_CACHE" != true && -d "$SHARED_CACHE_DIR" ]]; then
            echo "Removing the shared k-fold cache ${SHARED_CACHE_DIR}"
            rm -rf "$SHARED_CACHE_DIR"
        fi
        ;;

    split_local)
        # --prepare-job with --no-slurm: the same steps one after the other
        mkdir -p "$(dirname "$SHARED_CACHE_DIR")"
        run_prepare "$SHARED_CACHE_DIR" keep
        copy_shared_cache
        for ((first=0; first<K_FOLDS; first+=FOLDS_PER_JOB)); do
            last=$(( first + FOLDS_PER_JOB - 1 ))
            [[ "$last" -ge "$K_FOLDS" ]] && last=$(( K_FOLDS - 1 ))
            launch_trainings $(seq $(( INIT_SEED + first )) $(( INIT_SEED + last )))
        done
        MERGE_CACHE_DIR="$SHARED_CACHE_DIR"
        run_postproc || exit 1
        if [[ "$KEEP_CACHE" != true ]]; then
            echo "Removing the shared k-fold cache ${SHARED_CACHE_DIR}"
            rm -rf "$SHARED_CACHE_DIR"
        fi
        ;;

    *)
        echo "Error: unknown mode '${MODE}'."
        exit 1
        ;;
esac

echo "Done."
