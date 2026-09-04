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

# install the package in editable mode
pip install -e .
```

`requirements.txt` also installs the `HEPPlotter` class from the
[AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs) repository,
which is used by all the plotting scripts (see
[Plotting with HEPPlotter](#plotting-with-hepplotter)).

### Update HEPPlotter

> [!IMPORTANT]
> To Install the `HEPPlotter` class you can use
>
> ```bash
> pip install --upgrade  --no-cache-dir git+https://github.com/matteomalucchi/AnalysisConfigs.git
> ```
>
> This command should be executed every time you want to pull from the AnalysisConfigs repository and update the `HEPPlotter`.
> If it doesn't update, you should first uninstall it with `pip uninstall configs` and then install it again with the command above.

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

## `ml_train` options

`ml_train` is the main entry point for training and evaluation. All options below override the corresponding config-file values when provided.

```
ml_train [OPTIONS]

  -c, --config FILE            Path to the YAML configuration file
  -o, --output-dir DIR         Output directory (overrides config)
  -d, --data-dirs DIR [DIR…]   Data directories (overrides config)
  -b, --batch-size INT         Batch size
  -e, --epochs INT             Number of epochs
  -n, --num-workers INT        Number of DataLoader workers
  -s, --seed STR               Seed for shuffling and weight initialisation
  -g, --gpus STR               GPU indices, comma-separated (e.g. "0,1")

Evaluation / output
  -ev, --eval                  Evaluate the model on the test set (no training)
  -em, --eval-model PATH       Path to an existing model to evaluate instead of training
  -l,  --load-model PATH       Load a checkpoint and continue training from it
  --onnx                       Export the best model to ONNX format after training
  -sm, --save-model            Save the full model object next to the state dict
  -s-n, --save-numpy           Save numpy arrays of the output scores
  --overwrite                  Overwrite an existing output directory

Plots
  --histos                     Plot signal/background output distributions
  --roc                        Plot the ROC curve
  --history                    Plot the training-loss history
  --input-plots                Plot input-variable distributions before training (default: on)
  --no-input-plots             Skip the input-variable plots
  --input-plots-dir DIR        Subdirectory for input-variable plots (default: input_variables)
  --input-plots-bins INT       Number of histogram bins for input-variable plots
  --input-plots-log            Also save input-variable plots with a log y axis

Comet ML logging
  -ct,  --comet-token STR      Comet API token
  -cn,  --comet-name STR       Comet username
  -cw,  --comet-workspace STR  Comet workspace (overrides auto-derived name)
  -ctg, --comet-tags TAG […]   Comet experiment tags
  --pin-memory                 Pin memory for faster GPU data transfer
```

### Run only evaluation (no training)

To evaluate an already-trained model without running a new training, use `-ev` together with `-em`:

```bash
# Evaluate an existing model on the test set and produce all plots
ml_train -c configs/example_DNN_config_ggF_VBF.yml \
  -ev -em <output_dir>/best_model_state_dict.pt \
  --histos --roc --history

# Evaluate and export to ONNX
ml_train -c configs/example_DNN_config_ggF_VBF.yml \
  -ev -em <output_dir>/best_model_state_dict.pt \
  --onnx -o <output_dir>
```

`-ev` skips all training epochs; `-em` points to the saved state-dict (`.pt` file). The config file is still required to reconstruct the model architecture and locate the test data.

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
- The input variable name is read automatically from the training config YAML (`input_variables` field).
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
ml_onnx -i best_models -o best_models -ar --config <config_file>
# or with explicit variable name (backward compatible):
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

The training will produce the ONNX model to be used in PocketCoffea for background morphing, as well as plots with the training history, the ROC curve and an overtraining check (one plot per class, see [Plotting with HEPPlotter](#plotting-with-hepplotter)).

These plots can be produced using the following command:

```bash
# Plot the history of a training 
ml_history -i <training_log_file>

# Plot the ROC curve and overtraining check (all background kl inclusive)
ml_sb -i <training_directory>

# Plot only for a specific background kl value (creates bkgkl_<value>/ subdirectory)
ml_sb -i <training_directory> -klb 1.0

# Plot for every available background kl value
ml_sb -i <training_directory> -klb full

# Plot the input variable distributions of signal and background
ml_input_vars -c <config_file> -o <output_directory>
```

## Input variable distributions

Before the training starts, the normalized distributions of the input variables
for signal and background are plotted in the CMS style, together with a
signal/background ratio panel. The plots are saved in the `input_variables`
subdirectory of the output directory of the training.

This step is enabled by default and can be steered either from the config file
or from the command line:

```yaml
# config file
input_plots: True             # produce the input variable plots
input_plots_dir: input_variables  # subdirectory of the output directory
input_plots_bins: 30          # number of bins of the histograms
input_plots_log: False        # save also the histograms with a log y axis
```

```bash
# disable the input variable plots for a training
ml_train -c configs/example_DNN_config_ggF_VBF.yml --no-input-plots

# change the binning and save also the log-scale version
ml_train -c configs/example_DNN_config_ggF_VBF.yml --input-plots-bins 50 --input-plots-log
```

The same plots can be produced standalone (without running a training) with:

```bash
ml_input_vars -c configs/example_DNN_config_ggF_VBF.yml -o <output_directory>
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

### Comet workspace

The **Comet workspace** is derived automatically from the name of the directory containing the config file, with underscores replaced by hyphens. For example, a config saved under `configs/hh4b_ggF_VBF/` maps to the workspace `hh4b-ggF-VBF`, accessible at:

```
https://www.comet.com/hh4b-ggf-vbf#projects
```

Within that workspace, each experiment is grouped into a **project** named after the config file (without extension), e.g. `DNN_config_ggF_VBF`.

**Existing workspaces:**

| Config directory | Comet workspace |
|---|---|
| `configs/hh4b_bkg_reweighting/` | [hh4b-bkg-reweighting](https://www.comet.com/hh4b-bkg-reweighting#projects) |
| `configs/hh4b_ggF_VBF/` | [hh4b-ggf-vbf](https://www.comet.com/hh4b-ggf-vbf#projects) |
| `configs/hh4b_sig_bkg_classifier/` | [hh4b-sig-bkg-classifier](https://www.comet.com/hh4b-sig-bkg-classifier#projects) |

To add a new workspace:
1. Go to [comet.com](https://www.comet.com) and create the workspace manually from the website.
2. Name it to match the parent directory of your config file (underscores → hyphens), e.g. `configs/hh4b_sig_bkg_classifier/` → workspace `hh4b-sig-bkg-classifier` → `https://www.comet.com/hh4b-sig-bkg-classifier#projects`.
3. Add it to the table above.

The workspace can also be overridden explicitly with `--comet-workspace <name>` when calling `ml_train`.
