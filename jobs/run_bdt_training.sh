#!/bin/bash
#SBATCH --account=gpu_gres
#SBATCH --job-name=bdt_sig_bkg
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Train the XGBoost BDT of the boosted signal-vs-background classifier.
# Same as run_sig_bkg_classifier.sh, for configs with ML_model: XGB_multiclass_model.
#
# Usage, from the jobs/ directory (config path relative to ../configs/hh4b_sig_bkg_classifier/, .yml optional):
#   sbatch run_bdt_training.sh <CONFIG> [OUT_DIR] [OPTIONS] [-- EXTRA_ML_TRAIN_ARGS]
#   bash   run_bdt_training.sh <CONFIG> [OUT_DIR] [OPTIONS] [-- ...]      # run locally, no Slurm
#
# Options:
#   --gpus G       GPU number for XGBoost, or "cpu" (default 0; use cpu without a GPU)
#   --optuna N     run N Optuna trials of the hyperparameter search before the final training
#   --no-onnx      do not export the ONNX model
#   -h, --help     this help
# The Slurm time limit can be changed with `sbatch --time=HH:MM:SS run_bdt_training.sh ...`
# (the Optuna search needs more time: set `optuna.timeout` in the config below it).
# Default OUT_DIR: ../out/<config name> (i.e. /work/<user>/ML_pytorch/out/<config name>, next to the other trainings).
# ml_train is taken from the PATH, otherwise from $ML_PYTORCH_ENV/bin (default /work/$USER/micromamba/envs/ML_pytorch),
# so the job also works when it is submitted from a shell without the environment.
#
# Examples:
#   sbatch run_bdt_training.sh boosted/xgb_boosted_signal_background_classifier_50GeV_noH2WP_txbbwps.yml
#   sbatch run_bdt_training.sh boosted/xgb_..._optuna.yml /full/path/out --optuna 30
#   bash run_bdt_training.sh boosted/xgb_....yml /tmp/out --gpus cpu -- -e 50

usage() {
    sed -n '/^# Train the XGBoost/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}

GPUS=0
OPTUNA_TRIALS=""
ONNX=true
POSITIONAL=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)     GPUS="$2"; shift 2 ;;
        --optuna)   OPTUNA_TRIALS="$2"; shift 2 ;;
        --no-onnx)  ONNX=false; shift ;;
        -h|--help)  usage 0 ;;
        --)         shift; EXTRA_ARGS=("$@"); break ;;
        *)          POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
    echo "Error: the config is required."
    usage 1
fi

# The directory of this script. Under Slurm $0 is a spooled copy, so use the
# directory the job was submitted from (the jobs/ folder).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    JOBS_DIR="$SLURM_SUBMIT_DIR"
else
    JOBS_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
fi

CONFIG=${POSITIONAL[0]%.yml}
CONFIG_FILE="${JOBS_DIR}/../configs/hh4b_sig_bkg_classifier/${CONFIG}.yml"
OUT_DIR="${POSITIONAL[1]:-${JOBS_DIR}/../out/$(basename "$CONFIG")}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

# ml_train: from the PATH, or from the ML_pytorch environment
if ! command -v ml_train >/dev/null 2>&1; then
    ENV_BIN="${ML_PYTORCH_ENV:-/work/${USER}/micromamba/envs/ML_pytorch}/bin"
    [[ -x "$ENV_BIN/ml_train" ]] && export PATH="$ENV_BIN:$PATH"
fi
if ! command -v ml_train >/dev/null 2>&1; then
    echo "Error: ml_train not found: activate the ML_pytorch environment or set ML_PYTORCH_ENV"
    exit 1
fi
echo "ml_train:          $(command -v ml_train)"

echo "Using CONFIG_FILE: $CONFIG_FILE"
echo "Output directory:  $OUT_DIR"
echo "XGBoost device:    $GPUS"
[[ -n "$OPTUNA_TRIALS" ]] && echo "Optuna trials:     $OPTUNA_TRIALS"

# XGBoost (hist, CPU part) and the data loading use the CPUs of the job
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

train_args=(-o "$OUT_DIR" --eval --roc --histos --history --gpus "$GPUS" -n 2 -c "$CONFIG_FILE" --overwrite)
[[ "$ONNX" == true ]] && train_args+=(--onnx)
[[ -n "$OPTUNA_TRIALS" ]] && train_args+=(--optuna "$OPTUNA_TRIALS")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && train_args+=("${EXTRA_ARGS[@]}")

# Comet ML (optional): the same credentials file as the DNN scripts
COMET_KEY_FILE="${JOBS_DIR}/comet_token.key"
if [[ -s "$COMET_KEY_FILE" ]]; then
    { read -r API_UNAME; read -r API_KEY; } < "$COMET_KEY_FILE"
    echo "Found Comet username: $API_UNAME"
    train_args+=(--comet-token "$API_KEY" --comet-name "$API_UNAME" --comet-tags "BDT_training" "sig_bkg_classifier" "slurm")
fi

mkdir -p "$(dirname "$OUT_DIR")"
ml_train "${train_args[@]}"
