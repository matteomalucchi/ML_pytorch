# Configuration files and options

Back to the [main README](../README.md).

A training is described by a YAML config passed to `ml_train -c <config>`. Ready-made configs are under `configs/`, grouped by use case. Any key you leave out takes its default from [`ml_pytorch/defaults/default_configs.yml`](../ml_pytorch/defaults/default_configs.yml), and command-line options override the config.

## Declaring the classes

### Binary (signal vs background)

```yaml
signal_sample:    [GluGlutoHHto4B_spanet_skimmed]
signal_dataset:   [GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE]
signal_region:    [4b_signal_region]

background_sample:  [DATA_JetMET_JMENano_E_skimmed, DATA_JetMET_JMENano_F_skimmed]
background_dataset: [DATA_JetMET_JMENano_E_2022_postEE_EraE, DATA_JetMET_JMENano_F_2022_postEE_EraF]
background_region:  [2b_signal_region_postW]
```

Background is class `0` and signal is class `1`.

### Multi-class

For three or more classes, use a `classes` block instead. Each entry is one output node:

```yaml
ML_model: DNN_multiclass_model

classes:
  background:
    sample:  [DATA_JetMET_JMENano_E_skimmed, DATA_JetMET_JMENano_F_skimmed]
    region:  [2b_signal_region_postW]
    dataset: [DATA_JetMET_JMENano_E_2022_postEE_EraE, DATA_JetMET_JMENano_F_2022_postEE_EraF]
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
```

- Every class needs `sample`, `region`, `dataset` and an integer `lbl`. The classes are ordered by `lbl`, and that order is the output node order.
- The number of classes is taken from the block, so adding a class only needs a config change.
- The class-balancing options `undersample`, `oversample_split` and `split_oversample` only work with two classes.
- The model must be a multi-class one (e.g. `DNN_multiclass_model`). `ml_train` gives an error if you use a binary model with more than two classes.

A ready-to-edit example is [`configs/hh4b_sig_bkg_classifier/example_DNN_config_multiclass.yml`](../configs/hh4b_sig_bkg_classifier/example_DNN_config_multiclass.yml).

## Main config keys

```yaml
input_variables: sig_bkg_dnn_input_variables   # see "Input variables" below
ML_model: DNN_softmax_reweight_model           # file name in ml_pytorch/models/, without .py
data_format: coffea                            # coffea, parquet or root
data_dirs: [/path/to/pocketcoffea/output/]

batch_size: 512
epochs: 50
learning_rate: 1e-3
learning_rate_schedule: e5_drop75              # constant, linear, eX_dropYY, wX_cosineYY
optimizer: adam                                # adam or adamw
weight_decay: null                             # null = optimizer default

train_fraction: 0.7
val_fraction: 0.2
test_fraction: 0.1

early_stopping: True
patience: 5
min_delta: 1e-5
eval_param: "loss"                             # or "acc"
seed: 0
```

Other useful keys:

| Key | Meaning |
| --- | --- |
| `oversample_split`, `split_oversample`, `undersample` | class balancing (binary only) |
| `eval`, `roc`, `histos`, `history`, `onnx` | switch on/off the outputs produced after the training |
| `gpus` | comma-separated GPU indices; unset to run on CPU |
| `seed` | seed for all the randomness (shuffling, splits, weight initialisation). Resumed trainings continue exactly as if they had not stopped |
| `deterministic` | fully reproducible results on GPU too (slower) |
| `shuffle_train` | reshuffle the training events every epoch (default `False`) |
| `data_on_gpu` | keep the whole dataset on the GPU for speed (default `True`); it falls back to RAM automatically if it does not fit |
| `n_best_checkpoints` | how many best-epoch checkpoints to keep (default `5`, `0` keeps all) |
| `parquet_dir` | parquet only: use this directory of parquet files instead of the one in `data_dirs` |
| `pad_value` | value used for missing objects; must match the PocketCoffea fill value |
| `pad_fill_value`, `pad_indicators` | options of the masked batch-norm models (`*_maskedBN1D*`) for the padded entries |
| `input_plots`, `input_plots_bins`, `input_plots_log` | input-variable plots made before the training |
| `k_folds` and related keys | see [K-fold training](kfold_training.md) |

## Input variables

`input_variables` is either a list of branch names:

```yaml
input_variables: [JetGood_pt, JetGood_eta, JetGood_phi, JetGood_mass, JetGood_btagPNetB]
```

or the name of a module in `ml_pytorch/defaults/` that defines a `dnn_input_variables` dictionary:

```yaml
input_variables: sig_bkg_dnn_input_variables
```

Variables can be transformed before the training with `preprocess_variables_functions`, using the functions in `ml_pytorch/defaults/preprocess_variables_functions.py`.

## Models

Models are in `ml_pytorch/models/` and are selected with `ML_model` (the file name without `.py`). The main ones:

| Model | Use |
| --- | --- |
| `DNN_model` | binary, single sigmoid output |
| `DNN_softmax_reweight_model` | binary, two softmax outputs (background reweighting) |
| `DNN_softmax_reweight_model_maskedBN1D*` | as above, ignoring padded objects in the batch norm |
| `DNN_multiclass_model` | multi-class |

To add your own model, create a new file in `ml_pytorch/models/` (copy `DNN_multiclass_model.py` as a starting point). It must contain:

1. a `torch.nn.Module` whose `__init__` takes `dim_in` (and `num_classes` for multi-class);
2. a function `get_model(input_size, [num_classes,] device, lr, lr_schedule, n_epochs, optimizer="adam", weight_decay=None)` returning `(model, loss_fn, optimizer, scheduler)`. Use `reduction="none"` for the loss (the event weights are applied by the training loop) and build the optimizer and scheduler with `get_optimizer` and `get_lr_scheduler`, so that the config settings are used.

Then set `ML_model: <your_file_name>` in the config.

## `ml_train` command-line options

The most useful options (`ml_train --help` lists all of them). They override the config.

```
  -c, --config FILE          YAML config file
  -o, --output-dir DIR       output directory
  -d, --data-dirs DIR ...    input data directories
  -b, --batch-size INT       batch size
  -e, --epochs INT           number of epochs
  -s, --seed STR             random seed
  -g, --gpus STR             GPU indices, e.g. "0,1"
  --overwrite                overwrite an existing output directory

  -ev, --eval                only evaluate, no training
  -em, --eval-model PATH     model to evaluate (with -ev)
  -l,  --load-model PATH     resume training from a checkpoint
  --onnx                     export the best model to ONNX
  -s-n, --save-numpy         save the output scores as numpy arrays

  --histos / --roc / --history   produce the score histograms / ROC / loss history plots
  --no-input-plots               skip the input-variable plots

  -k, --k-folds INT          k-fold training (see kfold_training.md)
  --no-data-on-gpu           use the standard DataLoaders instead of keeping the data on the GPU
  -ct, -cn, -cw, -ctg        Comet token / user name / workspace / tags
```

### Evaluate an existing model

```bash
ml_train -c configs/hh4b_ggF_VBF/example_DNN_config_ggF_VBF.yml \
  -ev -em <output_dir>/state_dict/model_<epoch>_state_dict.pt --histos --roc
```

The config is still needed to rebuild the model and find the test data.
