import os
import shutil
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run a batch of 20 training jobs and copy the best model from each run.")
parser.add_argument("-o", "--base_dir", type=str, help="Base output directory for batch and run folders")
parser.add_argument("-c", "--config_file", type=str, help="Path to the config file")
parser.add_argument("-s", "--seeds", type=str, help="Number of seeds to run", default="0:20")
args = parser.parse_args()

out_dir=f"{args.base_dir}/{os.path.basename(args.config_file).replace('.yml','')}"
os.makedirs(out_dir, exist_ok=True)
print(f"Created output directory: {out_dir}")

seeds = args.seeds.split(":")

# Run the Python script N times with output to corresponding subfolders
for i in range(int(seeds[0]), int(seeds[1])):
    run_dir = f"{out_dir}/run{i:02d}"
    print(f"Running script with output to: {run_dir}")
    subprocess.run(["ml_train", "-o", run_dir, "-c", args.config_file, "-s", str(i)])

# Create best_models directory
best_models_dir = f"{out_dir}/best_models"
os.makedirs(best_models_dir, exist_ok=True)
print(f"Created best models directory: {best_models_dir}")

# Find the last .pt file in each run folder and copy it
for i in range(int(seeds[0]), int(seeds[1])):
    run_dir = f"{out_dir}/run{i:02d}/state_dict/"
    if os.path.isdir(run_dir):
        best_model = None
        try:
            best_model = sorted([f for f in os.listdir(run_dir) if "best_epoch" in f and f.endswith(".onnx")])[0]
        except IndexError:
            print(f"No .onnx files found in {run_dir}")
        
        if best_model:
            src = os.path.join(run_dir, best_model)
            dst = os.path.join(best_models_dir, f"best_model_run{i:02d}.onnx")
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
    else:
        print(f"No models directory found in {run_dir}")