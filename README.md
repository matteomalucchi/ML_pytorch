# ML_pytorch

Repository with basic machine learning algorithms implemented in PyTorch.

The coffea files used as inputs are based on the output of [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea/tree/main). In particular, the framework was developed based on the output of the
[AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs) repository, which is a collection of analysis configurations for the PocketCoffea framework.

The framework supports both **binary classification** (signal vs background) and **multi-class classification** with an arbitrary number of classes — see [Configuration files](#configuration-files) below.

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

## Repository layout

```
ml_pytorch/
├── defaults/                # default config, input-variable lists, preprocessing fns
│   ├── default_configs.yml
│   └── sig_bkg_dnn_input_variables.py, ...
├── models/                  # DNN architectures (importable by name from configs)
│   ├── DNN_model.py                       # binary, 1-node sigmoid + BCELoss
│   ├── DNN_softmax_reweight_model.py      # binary, 2-node softmax + CrossEntropy
│   ├── DNN_sigmoid_reweight_model.py      # binary, sigmoid w/ batchnorm + dropout
│   └── DNN_multiclass_model.py            # multi-class, num_classes-node softmax
├── scripts/                 # console entry points (train, sig_bkg_eval, plot_history, ...)
└── utils/                   # dataset loader, training loop, learning-rate schedules
configs/                     # YAML configuration files grouped by use case
```

## Configuration files

Configurations are YAML files consumed by `ml_train`. Examples live under `configs/`. The framework supports **two layouts** for declaring training samples; the dataset loader auto-detects which is in use.

### Binary (legacy) layout

Use this when you have exactly two classes (signal vs background):

```yaml
signal_sample:    [GluGlutoHHto4B_spanet_skimmed]
signal_dataset:   [GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE]
signal_region:    [4b_signal_region]

background_sample:  [DATA_JetMET_JMENano_E_skimmed, DATA_JetMET_JMENano_F_skimmed]
background_dataset: [DATA_JetMET_JMENano_E_2022_postEE_EraE, DATA_JetMET_JMENano_F_2022_postEE_EraF]
background_region:  [2b_signal_region_postW]

input_variables: sig_bkg_dnn_input_variables   # module name in ml_pytorch/defaults/
ML_model: DNN_softmax_reweight_model           # module name in ml_pytorch/models/
data_format: coffea                            # one of: coffea, parquet, root
data_dirs: [/path/to/coffea/output/]

batch_size: 512
epochs: 50
learning_rate: 1e-3
learning_rate_schedule: e5_drop75              # see ml_pytorch/utils/learning_rate_schedules.py

train_fraction: 0.7
val_fraction: 0.2
test_fraction: 0.1

early_stopping: True
patience: 5
min_delta: 1e-5
eval_param: "loss"                             # or "acc"
```

Internally background is mapped to class index `0` and signal to class index `1`, matching the previous behaviour of all existing configs and saved models.

### Multi-class layout

For three or more classes, replace the `signal_*` / `background_*` blocks with a `classes` mapping. Each entry becomes one output node:

```yaml
input_variables: sig_bkg_dnn_input_variables
ML_model: DNN_multiclass_model
data_format: coffea
data_dirs: [/path/to/coffea/output/]

classes:
  background:
    sample:  [DATA_JetMET_JMENano_E_skimmed, DATA_JetMET_JMENano_F_skimmed, DATA_JetMET_JMENano_G_skimmed]
    region:  [2b_signal_region_postW]
    dataset: [DATA_JetMET_JMENano_E_2022_postEE_EraE, DATA_JetMET_JMENano_F_2022_postEE_EraF, DATA_JetMET_JMENano_G_2022_postEE_EraG]
    lbl: 0
  ggF_HH:
    sample:  [GluGlutoHHto4B_spanet_skimmed]
    region:  [4b_signal_region]
    dataset: [GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE]
    lbl: 1
  VBF_HH:
    sample:  [VBF_HHto4B]
    region:  [4b_signal_region]
    dataset: [VBFHHto4B_CV_1_C2V_1_C3_1_2022_postEE]
    lbl: 2

batch_size: 512
epochs: 50
learning_rate: 1e-3
learning_rate_schedule: e5_drop75
train_fraction: 0.7
val_fraction: 0.2
test_fraction: 0.1
```

Notes on the `classes` block:

* Each class needs `sample`, `region`, `dataset` (same semantics as the legacy keys) and a `lbl` integer.
* Classes are ordered by ascending `lbl`, and that order becomes the **internal class index** (`0..C-1`) used both as the `CrossEntropyLoss` target and as the value stored in the saved score arrays. The user-provided `lbl` is kept as a label/identifier and shown in plot legends (`"className (lbl=...)"`).
* `num_classes` is derived from the size of the block, so adding a class is purely a config change provided the model accepts a `num_classes` argument (see below).
* `undersample`, `oversample_split` and `split_oversample` are signal-vs-background concepts and are only supported when there are exactly two classes.

A ready-to-edit multi-class example is provided at `configs/hh4b_sig_bkg_classifier/example_DNN_config_multiclass.yml`.

### Other useful config keys

| key | meaning |
|---|---|
| `input_variables` | either a list of branch names or the name of a module in `ml_pytorch/defaults/` providing a `dnn_input_variables` `OrderedDict` |
| `preprocess_variables_functions` | mapping `{var: [func_name, [args...]]}` resolved against `ml_pytorch/defaults/preprocess_variables_functions.py` |
| `data_format` | `coffea`, `parquet`, or `root` |
| `data_dirs` | list of directories containing the input files (`.coffea` files / parquet config) |
| `oversample_split`, `split_oversample`, `undersample` | binary-only class balancing strategies |
| `eval`, `roc`, `histos`, `history`, `onnx` | toggle the post-training outputs |
| `gpus` | comma-separated GPU indices, or unset for CPU |
| `seed` | seed for both event shuffling and weight initialisation |

The default values used when a key is omitted are listed in `ml_pytorch/defaults/default_configs.yml`.

## DNN architecture files

Models live in `ml_pytorch/models/` and are referenced by the `ML_model` key in the YAML config (the filename without `.py`). Each file must expose:

* a `torch.nn.Module` subclass implementing `forward`;
* a top-level `get_model(...)` factory returning `(model, loss_fn, optimizer, scheduler)`;
* optionally an `export_model(model)` method on the module that wraps the network for ONNX export (e.g. to apply a softmax inside the exported graph).

### Binary models — legacy signature

Binary models (1-node sigmoid + `BCELoss`, or 2-node softmax + `CrossEntropyLoss`) keep the original `get_model` signature:

```python
def get_model(input_size, device, lr, lr_schedule, n_epochs):
    model = DNN(input_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")  # or BCELoss for 1 node
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_lr_scheduler(lr_schedule, optimizer, n_epochs)
    return model, loss_fn, optimizer, scheduler
```

You can use any of these models with the binary (legacy) config layout — see `DNN_model.py`, `DNN_softmax_reweight_model.py`, `DNN_sigmoid_reweight_model.py` and variants.

### Multi-class models — `num_classes` signature

A multi-class model must accept an extra `num_classes` parameter so the size of the output layer can be driven by the YAML. `ml_pytorch/scripts/train.py` introspects the function signature and dispatches automatically: a model with `num_classes` in its signature gets `num_classes` passed in, a legacy binary model does not.

A minimal example (`ml_pytorch/models/DNN_multiclass_model.py`) is provided:

```python
from torch import nn
import torch

from ml_pytorch.utils.learning_rate_schedules import get_lr_scheduler


class DNN(nn.Module):
    def __init__(self, dim_in: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64),     nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64),     nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),       # multi-class logits
        )

    def forward(self, x):
        return self.net(x)

    def export_model(self, model):
        # ONNX wrapper that applies softmax inside the graph
        class ONNXWrappedModel(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return torch.nn.functional.softmax(self.m(x), dim=1)
        return ONNXWrappedModel(model)


def get_model(input_size, num_classes, device, lr, lr_schedule, n_epochs):
    model = DNN(input_size, num_classes).to(device)
    # raw logits + CrossEntropyLoss for single-label multi-class
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(lr_schedule, optimizer, n_epochs)
    return model, loss_fn, optimizer, scheduler
```

To create your own architecture:

1. Add a new file in `ml_pytorch/models/`, e.g. `DNN_my_model.py`.
2. Define a `torch.nn.Module` whose `__init__` takes `dim_in` (and `num_classes` if multi-class).
3. Define `get_model(input_size, [num_classes,] device, lr, lr_schedule, n_epochs)` returning `(model, loss_fn, optimizer, scheduler)`. Use `reduction="none"` on the loss — the training loop applies per-event weights manually.
4. Reference the file (without `.py`) from your config: `ML_model: DNN_my_model`.

When using a legacy binary model with a multi-class config (`num_classes > 2`), `ml_train` raises an explicit error and points you at `DNN_multiclass_model`.

## Input variable files

Lists of input branches live in `ml_pytorch/defaults/`, each exposing a `dnn_input_variables` `OrderedDict` mapping output column name → `[collection, leaf]`. Reference them from a config either as a literal list of branch names:

```yaml
input_variables: [JetGood_pt, JetGood_eta, JetGood_phi, JetGood_mass, JetGood_btagPNetB]
```

or as the name of a module under `ml_pytorch/defaults/`:

```yaml
input_variables: sig_bkg_dnn_input_variables
```

In the second form the columns are resolved via `create_DNN_columns_list(run2, dnn_input_variables)` in `ml_pytorch/utils/tools.py`, which also handles the `Run2` branch-name suffix when the config sets `run2: True`.

## Training on a cluster with slurm

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

## Outputs of a training

Each training run writes to `${output_dir}` (default `/work/$USER/out_ML_pytorch/<config_name>/`):

* `config_parameters.yml` — full resolved config (defaults + CLI + YAML) snapshot.
* `ML_model.py` — copy of the model file used, so a saved run can be re-loaded with `-l`/`--load-model` even after the file in `ml_pytorch/models/` changes.
* `logger_<name>.log` — training log with epoch-by-epoch loss / accuracy / learning rate.
* `state_dict/model_<epoch>_state_dict.pt` — best-epoch checkpoint(s).
* `state_dict/model_best_epoch_<n>.onnx` — ONNX export of the best model (and one for the last epoch).
* `score_lbl_array.npz` — saved with `--save-numpy`. Contains:
  * `score_lbl_array_train`, `score_lbl_array_test`: `[scores..., label, weight, kl]` per event. For binary the score block is a single column, for multi-class it has `num_classes` columns.
  * `train_test_fractions`, `num_classes`, `class_info` (list of `{name, lbl, class_idx}` dicts).
* Plots: `sig_bkg_distributions_*`, `roc_curve_*`, training history, etc.

## Additional scripts

The training will produce the ONNX model to be used in PocketCoffea for background morphing, as well as plots with the training history, the ROC curve and an overtraining check.

These plots can be produced using the following command:

```bash
# Plot the history of a training
ml_history -i <training_log_file>

# Plot the ROC curve and overtraining check
ml_sb -i <training_directory>
```

### Multi-class evaluation outputs

When `ml_sb` (the `sig_bkg_eval` script) detects more than one score column in `score_lbl_array.npz`, it automatically switches to multi-class mode and produces:

* `roc_curve_one_vs_rest.{png,pdf}` — for each class *i*, the ROC of "is class *i*" using the score of output node *i*.
* `roc_curve_one_vs_one.{png,pdf}` — for each ordered pair `(a, b)`, the ROC of "class *a* vs class *b*" using events with true label `∈ {a, b}` and the score of node *a*.
* `score_distribution_class_<i>.{png,pdf}` (+ `_log`) — distribution of output node *i*'s score, split by true class, for train and test.
* `tpr_fpr_multiclass.npz` — the saved (`tpr`, `fpr`, `auc`) arrays so that arbitrary class combinations can be replotted later.

The binary path (single score column) is unchanged and still produces `sig_bkg_distributions_kl_*` and `roc_curve_kl_*`.

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
