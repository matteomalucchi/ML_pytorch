#!/bin/bash

config_file=$1
out_dir=$2
#micromamba activate ML_pytorch

# Find the next available batch folder
batch_num=0
while [ -d "$out_dir/batch$(printf "%02d" $batch_num)" ]; do
    batch_num=$((batch_num + 1))
done

batch_dir="$out_dir/batch$(printf "%02d" $batch_num)"
mkdir -p "$batch_dir"
echo "Created batch directory: $batch_dir"

# Run the Python script 20 times with output to corresponding subfolders
for i in {0..19}; do
    run_dir="$batch_dir/run$(printf "%02d" $i)"
    echo "Running script with output to: $run_dir"
    ml_train -o "$run_dir" -c $config_file -s $i
done

# Create best_models directory
best_models_dir="$batch_dir/best_models"
mkdir -p "$best_models_dir"
echo "Created best models directory: $best_models_dir"

# Find the last .pt file in each run folder and copy it
for i in {0..19}; do
    run_dir="$batch_dir/run$(printf "%02d" $i)/state_dict/"
    if [ -d "$run_dir" ]; then
        ls "$run_dir"/*best_epoch*.onnx
        # get the first model in the list
        best_model=$(ls "$run_dir"/*best_epoch*.onnx | head -n 1)
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
