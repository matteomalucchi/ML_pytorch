# K-fold training

Back to the [main README](../README.md).

With k-fold training the events are split into K folds and K models are trained. The model of fold `k` is tested on fold `k` and trained on all the other events. Every event therefore gets an unbiased (out-of-fold) score, and the K models are combined into a single ONNX model.

## Config

```yaml
k_folds: 5
kfold_split: random          # random, or event (fold = event number % K)
kfold_event_column: event    # column with the event number, for kfold_split: event
kfold_onnx: average          # which combined model is the final one, see below
onnx_ratio: False            # True for background reweighting (builds the ratio model)
weight_log_clamp: null       # optional bound on log(weight), e.g. 3.912 = ln 50
train_fraction: 0.8          # the non-test events are split into
val_fraction: 0.2            # training and validation with this ratio
```

- `test_fraction` is not used: the test set of each fold is the fold itself.
- All the folds use the same `seed`, so the split is the same in every fold and in every job. Each fold still starts from different random weights.
- The test fold is never oversampled, so each event gets exactly one out-of-fold score.

An example config is [`configs/hh4b_sig_bkg_classifier/example_DNN_config_kfold.yml`](../configs/hh4b_sig_bkg_classifier/example_DNN_config_kfold.yml).

## Splitting by event number (`kfold_split: event`)

With `kfold_split: event` the fold of every event is `event % K` (with `K = 2`: even and odd events). The event number is needed in two places.

**1. In the training inputs.** The event number must be saved as a column in the PocketCoffea output used for the training, in every sample, dataset and region of the config. It is **not** an input variable of the DNN: it is only used to assign the folds, so do not add it to `input_variables`.

```yaml
k_folds: 2
kfold_split: event
kfold_event_column: event    # name of the event-number column in the input files
```

Set `kfold_event_column` to the name of the column exactly as it appears in the input files (coffea or parquet). If the column is missing, `ml_train` stops with an error that lists the available columns, so you can check the right name there.

**2. When applying the final model.** With `kfold_onnx: event` the final model is `kfold_onnx/kfold_<type>_event_fold.onnx`. It has two inputs:

| Input | Shape / type | Content |
|---|---|---|
| `args_0` | `(N, n_features)`, float32 | the input variables, as for the other models |
| `event` | `(N,)`, int64 | the event number of each event |

For every event it returns the output of fold `event % K`, the model that did not see that event during the training. Pass the event number as **int64**: event numbers are too large to be represented exactly as float32.

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("kfold_onnx/kfold_classifier_event_fold.onnx")
out = sess.run(None, {
    "args_0": inputs.astype(np.float32),     # (N, n_features)
    "event": events.event.astype(np.int64),  # (N,), e.g. the NanoAOD event number
})
```

The other combined models (`average`, `median`, ...) only take `args_0`. To use the per-fold models by hand with `K = 2`, apply `kfold_onnx/folds/model_fold0.onnx` to the even events and `model_fold1.onnx` to the odd events.

## Running

On Slurm (recommended), all the folds in one job and merged at the end:

```bash
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 -g 8 --cpus 40 --mem-per-cpu 6000 [--ratio]
```

Locally, or by hand:

```bash
# all the folds one after the other, then merged
ml_train -c config.yml -k 5 -o out/kfold

# a single fold, then merge the folds
ml_train -c config.yml -k 5 --fold 3 -o out/kfold/fold_3
ml_kfold_merge -i out/kfold -c config.yml -k 5 [--ratio]

# only rebuild the combined ONNX models (e.g. with another combination)
ml_kfold_merge -i out/kfold --ratio --kfold-onnx median --onnx-only
```

The events are loaded once and shared by all the folds (the *k-fold cache*), so many folds can run on the same node without running out of RAM. This is automatic; `--no-kfold-cache` turns it off.

## Splitting the folds over several jobs

A single job with all the folds has to wait for a whole node to be free. To start the folds as soon as single GPUs are free, use `--prepare-job`: one CPU job loads the events, one GPU job per fold (or per `--folds-per-job` folds) trains, and a CPU job merges at the end. The three steps are chained automatically.

```bash
# 20 jobs of 1 fold on 1 GPU each
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 --prepare-job -g 1 --cpus 4 --mem-per-cpu 6000

# 5 jobs of 4 folds on 2 GPUs each
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 --prepare-job --folds-per-job 4 -g 2 --cpus 8
```

The shared events are stored in `<outdir>/kfold_cache_shared`, which must be on a disk visible from all the nodes, and are removed after the merge.

The same steps can be run one at a time:

```bash
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 --prepare-only
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 --folds 0-7 -g 8 --cpus 32
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 --folds 8-19 -g 4 --cpus 16
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 --merge-only
```

Use the same config, `-s` and `-k` in all the steps.

Useful options:

| Option | Effect |
|---|---|
| `--cpu-partition`, `--cpu-account`, `--cpu-cpus`, `--cpu-mem-per-cpu`, `--cpu-time` | resources of the prepare and merge jobs (default `-p standard --account=t3`) |
| `--prepare-gpu`, `--merge-gpu` | run the prepare / merge job on the GPU partition instead |
| `--slots-per-gpu S` | in a single big job, run at most S folds per GPU and start the next one as soon as a GPU is free |
| `--separate-merge` | merge in a separate CPU job, so the GPUs are released as soon as the last fold ends |
| `--after JOBIDS` | wait for these jobs before starting |
| `--keep-cache` | keep the shared cache after the merge |

## When folds fail

If some folds crash or hit the time limit, the job prints the command to finish. The status of every fold is also written to `<outdir>/kfold_status.txt`. Then run one of:

```bash
# the same command as the first time, with --load-last
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 -g 8 --cpus 40 --load-last

# or: one GPU job per unfinished fold, then the merge
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out -k 20 --missing-folds -g 1 --cpus 4 --mem-per-cpu 6000
```

Finished folds are skipped, interrupted ones resume from their last checkpoint, and then the folds are merged. The folds are taken from the split saved in `<outdir>/kfold_split`, so the resumed folds are exactly the same as before. If the config or the input files changed in the meantime, the job stops with an error.

With `--prepare-job`, if a fold job fails the merge job stays pending forever (`DependencyNeverSatisfied` in `squeue`). Cancel it with `scancel <merge id>` and run the `--missing-folds` command above.

## Outputs

| Path | Content |
|---|---|
| `fold_<k>/` | a normal training output for every fold |
| `kfold_split/` | the record of the split, used to resume failed folds |
| `kfold/kfold_summary.yml` | best epoch and losses of every fold, and the name of the final ONNX model |
| `kfold/` | plots of the out-of-fold scores of all the events (and the loss history of all the folds) |
| `kfold_onnx/folds/model_fold<k>.onnx` | the best model of every fold |
| `kfold_onnx/kfold_<type>_<combination>.onnx` | the combined models |

`<type>` is `classifier` (the scores), `morphing_ratio` (the reweighting weight, with `onnx_ratio: True` or `--ratio`) or `morphing_ttbar_subtracted` (the boosted multiclass morphing, see [Boosted HH4b](boosted_hh4b.md)).

### Choosing the final model (`kfold_onnx`)

| `kfold_onnx` | Model | Use it for |
|---|---|---|
| `average` (default) | average of the folds | events not used in the training, e.g. reweighting applied in another region |
| `median` | median over the folds of the weight (needs `onnx_ratio` or the morphing) | as `average`, but robust against one fold with an outlying weight |
| `logaverage`, `logmedian` | geometric mean / median of the weights (needs a `*_logdiff` model) | as `average`, with even less sensitivity to single large weights |
| `event` | the fold that did **not** see the event, chosen from the event number (needs `kfold_split: event`); takes the inputs plus the `event` number, see [Splitting by event number](#splitting-by-event-number-kfold_split-event) | events that were also used in the training, e.g. sig/bkg or ggF/VBF classifiers |

All the combinations that apply to the model are built anyway; `kfold_onnx` only selects which one is the final model recorded in `kfold_summary.yml`.

To check how much the per-fold weights differ, run:

```bash
ml_kfold_weight_diag -i out/kfold
```
