#!/bin/bash

config_file=configs/bkg_reweighting/DNN_AN_minDelta1em5.yml
#micromamba activate ML_pytorch

# Find the next available batch folder
batch_num=0
while [ -d "/work/mmalucch/out_ML_pytorch/DNN_AN_minDelta1em5/batch$(printf "%02d" $batch_num)" ]; do
    batch_num=$((batch_num + 1))
done

batch_dir="/work/mmalucch/out_ML_pytorch/DNN_AN_minDelta1em5/batch$(printf "%02d" $batch_num)"
mkdir -p "$batch_dir"
echo "Created batch directory: $batch_dir"

# Run the Python script 20 times with output to corresponding subfolders
for i in {0..19}; do
    run_dir="$batch_dir/run$(printf "%02d" $i)"
    echo "Running script with output to: $run_dir"
    ml_train  -o "$run_dir" -c $config_file -s $i
done

# Create best_models directory
best_models_dir="$batch_dir/best_models"
mkdir -p "$best_models_dir"
echo "Created best models directory: $best_models_dir"

# Find the last .pt file in each run folder and copy it
for i in {0..19}; do
    run_dir="$batch_dir/run$(printf "%02d" $i)/state_dict/"
    if [ -d "$run_dir" ]; then
        ls "$run_dir"/model_*state_dict.onnx
        best_model=$(ls "$run_dir"/model_*state_dict.onnx 2>/dev/null | grep -Eo 'model_[0-9]+\_state_dict.onnx' | sed 's/model_//; s/\_state_dict.onnx//' | sort -n | tail -n 1 | awk '{print "'$run_dir'/model_"$1"_state_dict.onnx"}')
        if [ -n "$best_model" ]; then
            cp "$best_model" "$best_models_dir/best_model_run$(printf "%02d" $i).onnx"
            echo "Copied $best_model to $best_models_dir/best_model_run$(printf "%02d" $i).onnx"
        else
            echo "No .onnx files found in $run_dir"
        fi
    else
        echo "No models directory found in $run_dir"
    fi
done
