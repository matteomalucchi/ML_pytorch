# Training on a cluster with Slurm

Back to the [main README](../README.md).

`jobs/run_training.sh` runs any config, any number of trainings, on any number of GPUs and nodes, and combines the ONNX models at the end. It submits itself to Slurm, so run it directly (no `sbatch`), from an environment where `ML_pytorch` is activated.

```bash
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/outdir [OPTIONS]
```

## Examples

```bash
# a single training
./jobs/run_training.sh -c /full/path/DNN_config_ggF_VBF.yml -o /full/path/out/ggF_VBF

# 20 trainings (seeds) on 1 node with 4 GPUs, combined for the background reweighting
./jobs/run_training.sh -c /full/path/DNN_config_bkg_reweighting.yml -o /full/path/out/bkg_rew -n 20 -g 4 --cpus 8 --ratio

# 20 trainings spread over 4 nodes
./jobs/run_training.sh -c /full/path/DNN_config_bkg_reweighting.yml -o /full/path/out/bkg_rew -n 20 -p 4 --ratio

# 20-fold training (see kfold_training.md)
./jobs/run_training.sh -c /full/path/config.yml -o /full/path/out/kfold -k 20 -g 8 --cpus 40 --mem-per-cpu 6000 --ratio

# test locally, without Slurm
./jobs/run_training.sh -c /full/path/DNN_config.yml -o /full/path/out --no-slurm
```

## Options

| Option | Meaning |
| --- | --- |
| `-c, --config FILE` | config file (full path), required |
| `-o, --outdir DIR` | output directory (full path), required |
| `-n, --n-trainings N` | number of trainings with different seeds (default 1) |
| `-s, --init-seed N` | first seed (default 0) |
| `-p, --nodes N` | number of Slurm jobs (nodes) to spread the trainings over (default 1) |
| `-g, --gpus-per-node N` | GPUs per job (default 1); the trainings are distributed over them |
| `--cpus N`, `--mem-per-cpu MB`, `--time HH:MM:SS` | Slurm resources per job (default 4 CPUs, 4000 MB) |
| `--ratio` | combine the models as weight ratios (background reweighting) |
| `--alpha` | boosted multiclass morphing with ttbar subtraction, see [Boosted HH4b](boosted_hh4b.md) |
| `-k, --k-folds K` | k-fold training, see [K-fold training](kfold_training.md) for the k-fold options |
| `--load-last` | resume the trainings from their last checkpoint |
| `--copy-to DIR` | copy the output to `DIR` (e.g. `/pnfs`) at the end, see below |
| `--no-slurm` | run in the current shell |
| `-- EXTRA` | extra options passed to `ml_train` |

Run `./jobs/run_training.sh --help` for the full list.

When all the trainings are done, the best ONNX model of every training is copied to `<outdir>/best_models/` and combined with `ml_onnx` into `best_models/average_model_from_onnx.onnx`.

If a training fails, the others still run to the end. The job then ends with an error and prints the command to resubmit the failed ones (usually the same command with `--load-last`).

If `jobs/comet_token.key` exists, the trainings are logged to Comet (see the [main README](../README.md#comet-logging)).

## Copy of the output to /pnfs

With `--copy-to DIR`, the output is written to `--outdir` as usual (e.g. on `/work`) and copied to `DIR` at the end, if the job succeeded. `/pnfs` is read-only on the batch nodes, so there the copy goes through xrootd and needs a grid proxy readable from the node. Create it before submitting:

```bash
voms-proxy-init -voms cms -out ~/.x509up_u$(id -u)
export X509_USER_PROXY=~/.x509up_u$(id -u)
```

```bash
./jobs/run_training.sh -c /full/path/config.yml -o /work/$USER/out -n 20 -g 4 --ratio \
  --copy-to /pnfs/psi.ch/cms/trivcat/store/user/$USER/out

# redo a failed copy, right away in the current shell
./jobs/run_training.sh -c /full/path/config.yml -o /work/$USER/out \
  --copy-to /pnfs/psi.ch/cms/trivcat/store/user/$USER/out --copy-only
```

## Resource usage monitor

Every job records its CPU, memory and GPU usage, and that of every training, in `<outdir>/resource_usage/`. The summary is printed at the end of the Slurm log and suggests values for `--cpus`, `--mem-per-cpu`, `--time` and the number of trainings per GPU. Use it to size the next submission. `--no-monitor` switches it off.

## Speed tips

The models are small, so the speed is limited by the data loading, not by the GPU:

- Keep `data_on_gpu: True` (default). The whole dataset is kept on the GPU when it fits, otherwise in RAM.
- A larger `batch_size` (e.g. 2048-8192) is the biggest speed-up, but changes the training: scale the learning rate accordingly.
- One GPU per training is enough; using several GPUs for one training (`--gpus 0,1`) is usually slower. Use the GPUs to run several trainings at the same time (`-g`), several trainings can also share a GPU.
- For large k-fold trainings, the limit is usually the RAM of the node; see [K-fold training](kfold_training.md).

## Legacy scripts

The older per-use-case scripts are still in `jobs/`:

```bash
cd jobs/
# 20 trainings for the background reweighting, then combine them
sbatch run_20_trainings_in_4_parallel.sh <config_file> <output_folder>
cd <output_folder> && ml_onnx -i best_models -o best_models -ar --config <config_file>

# signal-vs-background classifier
sbatch run_sig_bkg_classifier.sh <config_file> <output_folder>
```
