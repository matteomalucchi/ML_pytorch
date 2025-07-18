#!/bin/sh
#SBATCH --account=gpu_gres
#SBATCH --job-name=sig_bg_classifier
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=2:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# -- Editable entries --
LOAD_LAST=false
CONFIG=DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_inclusive_b_region_bratioAll_postEE_ArctanhDeltaProb
i=100

# ---- Script starts here ----

OUT_DIR="../out/sig_bkg_classifier/${CONFIG}"
CONFIG_FILE="../configs/ggF_bkg_classifier/${CONFIG}.yml"
SEED=$((i))
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

matches=("$MODEL_DIR"/*best_epoch*.onnx)

# Set BEST_MODEL only if we found matches
if [[ "$LOAD_LAST" == true && ${#matches[@]} -gt 0 ]]; then
	echo "Skipping this run, because it is already finished"
elif [[ -s "./comet_token.key" ]]; then
	{
		read -r API_UNAME
		read -r API_KEY
	} <./comet_token.key
	API_TAGS=("DNN_training" "sig_bkg_classifier" "slurm")
	echo "found Name $API_UNAME"
	echo "found Key $API_KEY"
	if [[ $LOAD_LAST == true ]]; then
		ml_train -o "$RUN_DIR" \
				 --eval --onnx --roc --histos --history\
				 --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" --load-model $MODEL_PATH --comet-token $API_KEY --comet-name $API_UNAME --comet-tags "${API_TAGS[@]}"
	else
		ml_train -o "$RUN_DIR" \
				 --eval --onnx --roc --histos --history\
			     --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" --overwrite --comet-token $API_KEY --comet-name $API_UNAME --comet-tags "${API_TAGS[@]}"
	fi
else
	if $LOAD_LAST; then
		ml_train -o "$RUN_DIR" \
				 --eval --onnx --roc --histos --history \
				 --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" --load-model $MODEL_PATH
	else
		ml_train -o "$RUN_DIR" \
				 --eval --onnx --roc --histos --history \
				 --gpus 0 -n 2 -c "$CONFIG_FILE" -s "$SEED" --overwrite
	fi

fi
