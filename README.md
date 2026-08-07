# ML_pytorch

Repository with basic machine learning algorithms implemented in PyTorch.

The coffea files used as inputs are based on the output of [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea/tree/main). In particular, the framework was developed based on the output of the
[AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs) repository, which is a collection of analysis configurations for the PocketCoffea framework.

## Installation

To create the micromamba environment, you can use the following command:

```bash
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00  -p gpu --gres=gpu:1
micromamba env create -f ML_pytorch_env.yml
micromamba activate ML_pytorch
pip install -r requirements.txt
# install the package in editable mode
pip install -e .
```

## Connect to node with a gpu

To connect to a node with a gpu, you can use the following command:

```bash
# connect to a node with a gpu
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00  -p gpu --gres=gpu:1
# activate the environment
micromamba activate ML_pytorch
# check which gpu is available
echo $CUDA_VISIBLE_DEVICES # or echo $SLURM_JOB_GPUS
```

## Examples

To execute an example training, evaluate the model on the test set, plot the history and plot the signal/background histograms, you can use the following command:

```bash
ml_train  -c configs/example_DNN_config_ggF_VBF.yml
```

## Training on a cluster with Slurm

### Generic training script (recommended)

`jobs/run_training.sh` is a self-submitting script that handles any config, any number of trainings, and any number of parallel Slurm nodes. It self-submits to Slurm when called directly (no `sbatch` needed).

```bash
./jobs/run_training.sh --config /full/path/config.yml --outdir /full/path/outdir [OPTIONS]

Required:
  -c, --config FILE       Full path to YAML config file
  -o, --outdir DIR        Full path to output directory

Optional:
  -n, --n-trainings INT   Total number of trainings (default: 1)
  -p, --nodes INT         Number of parallel Slurm nodes/array jobs (default: 1)
  -s, --init-seed INT     Starting random seed (default: 0)
  --ratio                 Average-ratio ONNX aggregation (ml_onnx -ar), e.g. for bkg reweighting
  --onnx-var NAME         ONNX input variable name to probe (repeatable; auto-probed if omitted)
  --load-last             Resume from latest checkpoint
  --no-slurm              Run directly without Slurm (for local testing)
  -- EXTRA                Extra arguments forwarded to ml_train
```

**Examples:**

```bash
cd jobs/

# Single training (any config)
./run_training.sh -c /full/path/DNN_config_ggF_VBF.yml -o /full/path/out/ggF_VBF

# 20 trainings across 4 GPU nodes (5 per node in parallel), with ratio ONNX aggregation
./run_training.sh -c /full/path/DNN_config_bkg_reweighting.yml -o /full/path/out/bkg_rew \
  -n 20 -p 4 --ratio

# 5 trainings on 1 node, plain ONNX averaging (no ratio)
./run_training.sh -c /full/path/DNN_config_sig_bkg.yml -o /full/path/out/sig_bkg -n 5

# Local test (no Slurm submission)
./run_training.sh -c /full/path/DNN_config.yml -o /full/path/out --no-slurm
```

**How it works:**

- **`NODES=1`**: submits a single Slurm job; trainings run in parallel on one GPU node, then ONNX post-processing runs inline.
- **`NODES>1`**: submits a Slurm array job (one element per node) plus a dependent post-processing job that runs after all array tasks succeed.
- Each node runs its share of trainings (`N_TRAININGS / NODES`) in parallel using background processes.
- After all trainings finish, the best ONNX model from each run is copied to `best_models/` and `ml_onnx` is called to aggregate them. With `--ratio` the `-ar` flag is passed (average ratio, used for background reweighting); without it, a plain aggregation is performed.
- The ONNX input variable name is probed automatically from a list of known names; use `--onnx-var` to override.
- If `jobs/comet_token.key` exists, Comet ML logging is enabled automatically (see [COMET integration](#comet-integration)).

### Legacy per-use-case scripts

To execute either a 20x training for background reweighting or to run a `sig_bkg_classifier` model, there are two scripts that can be run with slurm:

```bash
# Outside of any node activate your environment (e.g. `micromamba activate ML_pytorch`)
cd jobs/
# If the output folder is not provided, it will have the same name as the config file without the extension
# For 20x training for bkg reweighting:
sbatch run_20_trainings_in_4_parallel.sh <config_file> <output_folder>
# when this has finished, you can merge the results with:
cd <output_folder>
ml_onnx -i best_models -o best_models -ar -v bkg_morphing_dnn_DeltaProb_input_variables

# For sig_bkg_reweighting
sbatch run_sig_bkg_classifier.sh <config_file> <output_folder>
```

To execute 5 runs in a node without the interactive access to the GPU node (the given config and folder names are just examples):

```bash
# Outside of any node activate your environment (e.g. `micromamba activate ML_pytorch`)

# Then run this command:
sbatch --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 5000 --time 12:00:00  -p gpu --gres=gpu:1 --wrap=". ./run_batch_of_5.sh /work/tharte/datasets/ML_pytorch/configs/bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE.yml out/bkg_reweighting/SPANET_ptFlat_20_runs_postEE 0"
```

## Additional scripts

The training will produce the ONNX model to be used in PocketCoffea for background morphing, as well as plots with the training history, the ROC curve and an overtraining check.

These plots can be produced using the following command:

```bash
# Plot the history of a training 
ml_history -i <training_log_file>

# Plot the ROC curve and overtraining check
ml_sb -i <training_directory>
```

## COMET integration

Additionally, there are now options to send the metrics of the training to [COMET](https://www.comet.com/site) (academics accounts are available for free):
To set it up together with the files mentioned above:

```bash
# Open the file with the editor of your choice
vim jobs/comet_token.key
# in the first line write your username, and in the second line, write your token (to be retrieved on the website):
# <uname>
# <token>
```

The scripts will read this file if it exists and automatically sends the information to `ml_pytorch`
