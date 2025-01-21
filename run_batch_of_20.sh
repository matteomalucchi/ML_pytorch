#!/bin/bash

coffea_dir=/work/mmalucch/out_hh4b/out_JMENanoData_comparisonRun2_DNN_variables_SR_CR_PADm10_fixed_SPANETptVary0p3_1p7
config_file=configs/DNN_bkg_reweighting.yml

# Find the next available batch folder
batch_num=0
while [ -d "out/batch$(printf "%02d" $batch_num)" ]; do
    batch_num=$((batch_num + 1))
done

batch_dir="out/batch$(printf "%02d" $batch_num)"
mkdir -p "$batch_dir"
echo "Created batch directory: $batch_dir"

# Run the Python script 20 times with output to corresponding subfolders
for i in {0..19}; do
    run_dir="$batch_dir/run$(printf "%02d" $i)"
    echo "Running script with output to: $run_dir"
    python scripts/train.py -d $coffea_dir  -o "$run_dir" --eval --onnx --roc --histos --history --gpus 7 -n 4 -c $config_file
done

# Create best_models directory
best_models_dir="$batch_dir/best_models"
mkdir -p "$best_models_dir"
echo "Created best models directory: $best_models_dir"

# Find the last .pt file in each run folder and copy it
for i in {0..19}; do
    run_dir="$batch_dir/run$(printf "%02d" $i)/models"
    if [ -d "$run_dir" ]; then
        best_model=$(ls "$run_dir"/model_*.pt 2>/dev/null | grep -Eo 'model_[0-9]+\.pt' | sed 's/model_//; s/\.pt//' | sort -n | tail -n 1 | awk '{print "'$run_dir'/model_"$1".pt"}')
        if [ -n "$best_model" ]; then
            cp "$best_model" "$best_models_dir/best_model_run$(printf "%02d" $i).pt"
            echo "Copied $best_model to $best_models_dir/best_${best_model}_run$(printf "%02d" $i).pt"
        else
            echo "No .pt files found in $run_dir"
        fi
    else
        echo "No models directory found in $run_dir"
    fi
done
