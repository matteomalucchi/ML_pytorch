# ML_pytorch

DNN trainings in PyTorch for the HH→4b analysis: signal-vs-background classifiers, ggF-vs-VBF classifiers and background reweighting (morphing). Binary and multi-class classification are supported.

The inputs are the outputs of [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea/tree/main), produced with the configurations of [AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs). The trained models are exported to ONNX to be used back in PocketCoffea.

## Documentation

| Page | Content |
| --- | --- |
| [Configuration](docs/configuration.md) | config files, classes, input variables, models, `ml_train` options |
| [Training on Slurm](docs/slurm_training.md) | `run_training.sh`: many trainings / GPUs / nodes, copy to `/pnfs`, resource monitor |
| [K-fold training](docs/kfold_training.md) | k-fold trainings, splitting by event number, resuming failed folds, combined ONNX models |
| [Boosted HH4b](docs/boosted_hh4b.md) | multiclass background morphing with ttbar subtraction, sig/bkg classifier |

## Installation

On a GPU node:

```bash
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00 -p gpu --gres=gpu:1
micromamba env create -f ML_pytorch_env.yml
micromamba activate ML_pytorch
pip install -e .
```

The plotting uses the `HEPPlotter` class of [AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs), installed with the requirements. To update it:

```bash
pip install --upgrade --no-cache-dir git+https://github.com/matteomalucchi/AnalysisConfigs.git
```

If it does not update, run `pip uninstall configs` first.

## Quick start

Interactively on a GPU node:

```bash
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00 -p gpu --gres=gpu:1
micromamba activate ML_pytorch
ml_train -c configs/hh4b_ggF_VBF/example_DNN_config_ggF_VBF.yml -o out/example
```

This trains the model, evaluates it on the test set and makes the plots. On the cluster, submit it with Slurm instead (see [Training on Slurm](docs/slurm_training.md)):

```bash
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/outdir [-n 20 -g 4 --ratio]
```

## Repository layout

```
ml_pytorch/
├── defaults/   # default config values, input-variable lists, preprocessing functions
├── models/     # DNN architectures, selected by name with ML_model
├── scripts/    # command-line tools (ml_train, ml_sb, ml_onnx, ...)
└── utils/      # dataset loading, training loop, learning-rate schedules
configs/        # YAML configs, grouped by use case
jobs/           # Slurm scripts (run_training.sh)
docs/           # documentation
```

## Command-line tools

| Command | Use |
| --- | --- |
| `ml_train` | train and evaluate a model (see [Configuration](docs/configuration.md#ml_train-command-line-options)) |
| `ml_sb -i <training_dir>` | ROC curves and overtraining plots (`-klb 1.0` or `-klb full` for given background kl values) |
| `ml_history -i <log_file>` | plot the training history |
| `ml_input_vars -c <config> -o <dir>` | plot the input variable distributions |
| `ml_onnx` | combine the ONNX models of several trainings (`-ar` for reweighting) |
| `ml_kfold_merge`, `ml_kfold_weight_diag` | merge / check a k-fold training (see [K-fold training](docs/kfold_training.md)) |
| `ml_alpha -c <config>` | ttbar fraction for the boosted morphing (see [Boosted HH4b](docs/boosted_hh4b.md)) |

## Outputs of a training

Each training writes to the output directory (default `/work/$USER/out_ML_pytorch/<config_name>/`):

| File | Content |
| --- | --- |
| `config_parameters.yml` | the full config used (defaults + YAML + command line) |
| `ML_model.py`, `<input_variables>.py` | copies of the model and of the input variables used, to reload the training later |
| `logger_<name>.log` | training log, epoch by epoch |
| `state_dict/model_<epoch>_state_dict.pt` | checkpoints of the best epochs and of the last one |
| `state_dict/model_best_epoch_<n>.onnx` | **the ONNX model to use in PocketCoffea** |
| `input_variables/` | distributions of the input variables per class |
| plots | score distributions with train/test ratio (overtraining check), ROC curves, calibration curves, loss history |
| `score_lbl_array.npz` | the output scores, labels and weights (only with `--save-numpy`) |

For multi-class models, `ml_sb` produces one-vs-rest and one-vs-one ROC curves and the score distributions of every output node. With `--save-roc` the ROC points are saved too, to replot them later.

The input-variable plots can be configured or switched off in the config (`input_plots`, `input_plots_bins`, `input_plots_log`) or with `--no-input-plots`.

## Comet logging

The metrics of every epoch (loss, accuracy, learning rate) can be sent to [Comet](https://www.comet.com/site) (free academic accounts). Create `jobs/comet_token.key` with your user name on the first line and your API token on the second:

```
<username>
<token>
```

If the file exists, the logging is enabled automatically. The Comet workspace is the name of the directory of the config, with `_` replaced by `-`, and the project is the config file name:

| Config directory | Comet workspace |
| --- | --- |
| `configs/hh4b_bkg_reweighting/` | [hh4b-bkg-reweighting](https://www.comet.com/hh4b-bkg-reweighting#projects) |
| `configs/hh4b_ggF_VBF/` | [hh4b-ggf-vbf](https://www.comet.com/hh4b-ggf-vbf#projects) |
| `configs/hh4b_sig_bkg_classifier/` | [hh4b-sig-bkg-classifier](https://www.comet.com/hh4b-sig-bkg-classifier#projects) |

For a new config directory, create the workspace on the Comet website first and add it to this table. `--comet-workspace <name>` overrides the workspace.
