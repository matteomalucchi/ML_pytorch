# Training on a cluster with Slurm

Back to the [main README](../README.md).

## Generic training script (recommended)

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
  -k, --k-folds INT       K-fold training: one training per fold (same seed), in OUTDIR/fold_<k>,
                          merged with ml_kfold_merge (see "K-fold training" in the main README); the events are
                          loaded once in OUTDIR/kfold_cache and shared by the folds
  --no-kfold-cache        K-fold: every fold loads the events itself
  --prepare-job           K-fold in several jobs: CPU prepare job, array of GPU fold jobs
                          (--folds-per-job folds each), CPU merge job, chained
  --prepare-only / --folds LIST / --merge-only
                          the same steps submitted one at a time (see "K-fold in several jobs" in the main README)
  --cache-dir DIR         shared k-fold cache (default OUTDIR/kfold_cache_shared), --keep-cache
  --cpu-partition / --cpu-account / --cpu-cpus / --cpu-mem-per-cpu / --cpu-time
                          resources of the CPU jobs (default -p standard --account=t3)
  --prepare-gpu / --merge-gpu
                          run the prepare / merge job on the GPU partition (1 GPU)
  --after JOBIDS          with --folds / --merge-only / --fold-jobs / --missing-folds:
                          wait for these jobs (colon-separated)
  --fold-jobs             all the fold jobs (array) + the merge, after --after or from the shared cache
  --missing-folds         resubmit only the unfinished folds (resuming) + the merge
  --slots-per-gpu INT     at most INT trainings per GPU at a time, the others start when a GPU frees up
  --separate-merge        merge / post-process in a separate CPU job, releasing the GPUs earlier
  --monitor-interval SEC  seconds between the samples of the resource monitor (default: 10)
  --no-monitor            do not run the resource monitor (see "Resource usage monitor")
  --copy-to DIR           copy OUTDIR to DIR (e.g. on /pnfs) at the end of the last job, if it
                          succeeded (see "Copy of the output to /pnfs")
  --copy-redirector URL   xrootd redirector for the copy with xrdcp
                          (default: root://t3dcachedb03.psi.ch:1094)
  --copy-only             only copy OUTDIR to --copy-to now, in this shell (no Slurm)
  --load-last             Resume from latest checkpoint
  --time HH:MM:SS         Slurm wall-clock time limit
  --mem-per-cpu MB        Slurm memory per CPU in MB (default: 4000)
  --cpus INT              Slurm CPUs per node/array job (default: 4)
  -g, --gpus-per-node INT GPUs per node/array job (default: 1)
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

# 20 trainings on 1 node with 4 GPUs (5 per GPU in parallel)
./run_training.sh -c /full/path/DNN_config_bkg_reweighting.yml -o /full/path/out/bkg_rew \
  -n 20 -g 4 --cpus 8 --ratio

# 5 trainings on 1 node, plain ONNX averaging (no ratio)
./run_training.sh -c /full/path/DNN_config_sig_bkg.yml -o /full/path/out/sig_bkg -n 5

# Local test (no Slurm submission)
./run_training.sh -c /full/path/DNN_config.yml -o /full/path/out --no-slurm
```

**How it works:**

- **`NODES=1`**: submits a single Slurm job; trainings run in parallel on one GPU node, then ONNX post-processing runs inline.
- **`NODES>1`**: submits a Slurm array job (one element per node) plus a dependent post-processing job that runs after all array tasks succeed.
- Each node runs its share of trainings (`N_TRAININGS / NODES`) in parallel using background processes.
- With `--gpus-per-node N` each node/array job requests `--gres=gpu:N` and the trainings of the node are distributed round-robin over its GPUs (training `i` runs on GPU `i % N`). The CPUs of the job are split among the parallel trainings (`OMP_NUM_THREADS`, DataLoader workers), so the processes do not oversubscribe the cores.

## Copy of the output to /pnfs

With `--copy-to DIR` the output is written as usual to `--outdir` (e.g. on `/work`) and, at the end of the last job of the chain (the single training job, the inline or separate post-processing / k-fold merge job), if it succeeded, every file of `OUTDIR` is copied to the same relative path in `DIR`. `OUTDIR` is kept, and the shared k-fold cache is not copied.

- Files on dCache (`/pnfs`) cannot be overwritten: an existing copy is removed first (`rm`, or `xrdfs rm`).
- If `DIR` is writable through the mount (e.g. `/pnfs` on the T3 login nodes, or any normal disk) the copy is done with `rsync`. Otherwise (`/pnfs` is read-only on the batch nodes) it is done with `xrdcp --path --cksum adler32:source` through `--copy-redirector`. This needs a valid grid proxy readable from the batch node, e.g. `voms-proxy-init -voms cms -out ~/.x509up_u$(id -u)` and `export X509_USER_PROXY=~/.x509up_u$(id -u)` before submitting (`sbatch` exports the environment).
- The size of every copy is checked (the content is checked by rsync / the xrdcp checksum). A wrong copy is removed and the job ends with an error. The Slurm logs, still being written, are copied without the size check.
- To redo a failed copy (or to copy an existing output), run the same command with `--copy-only`. The copy then runs right away in the current shell, without Slurm.

```bash
./run_training.sh -c /full/path/config.yml -o /work/$USER/out -n 20 -g 4 --ratio \
  --copy-to /pnfs/psi.ch/cms/trivcat/store/user/$USER/out
# redo the copy by hand, e.g. from a login node
./run_training.sh -c /full/path/config.yml -o /work/$USER/out \
  --copy-to /pnfs/psi.ch/cms/trivcat/store/user/$USER/out --copy-only
```

## Resource usage monitor

Every job of `run_training.sh` (training, array task, prepare, merge, also with `--no-slurm`) runs `jobs/resource_monitor.py` in the background. Every `--monitor-interval` seconds (default 10) it samples:

- the CPU cores and memory of the whole job, from the Slurm cgroup of the job (exact, includes all the processes);
- every GPU of the job with `nvidia-smi`: utilisation, memory, power, and which trainings are on it;
- every training (`run<NN>` / `fold_<k>`, including its DataLoader workers) and step (`kfold_prepare`, `ml_kfold_merge`, `ml_onnx`, `ml_alpha`): CPU cores, RSS, GPU memory and SM utilisation of its processes.

The results go to `<outdir>/resource_usage/<mode>_<jobid>[_<array task>]_*`:

| File | Content |
|---|---|
| `*_summary.txt` / `*_summary.json` | time occupancy (elapsed vs time limit, time in every phase, wall time of every training), CPU and memory used vs allocated, per-GPU busy/unused time and utilisation, per-training table, and suggested `--cpus`, `--mem-per-cpu`, `--time`, trainings per GPU / `-g` |
| `*_job.csv`, `*_gpu.csv`, `*_trainings.csv` | the time series of the job, of every GPU and of every training |
| `*_phases.txt`, `*_monitor.log` | phase markers written by the script, monitor log |

The summary is printed at the end of the Slurm log, and is rewritten every 5 minutes during the job, so it is there also if the job hits its time limit. The suggestions are heuristics: the memory one uses the non-reclaimable peak (anon + shmem + mapped files) +20% +1 GB; "trainings per GPU" combines the peak GPU memory of a training with the GPU utilisation measured while the GPU was used. To run the monitor by hand on any process: `python jobs/resource_monitor.py --pid <PID> --outdir <dir> --tag <name> [--interval 10]`.

## Speed tips

The models are small DNNs, so the training is limited by the data loading and the Python overhead rather than by the GPU compute:

- `data_on_gpu: True` (default) copies the whole datasets to the GPU once and takes the batches as slices of it, removing the DataLoader workers and the per-batch host-to-device copies. If the datasets take more than half of the free GPU memory (e.g. tens of millions of events on an 11 GB GPU), they are kept in RAM: the batches are sliced there and copied to the GPU one by one. This is still much faster than the DataLoaders, which index and collate the events one at a time. `--no-data-on-gpu` uses the standard DataLoaders.
- The loss and accuracy are accumulated on the GPU (no synchronisation at every batch) and the validation/evaluation run without building the autograd graph.
- A larger `batch_size` (e.g. 2048-8192) is the biggest further speed-up, but it changes the training dynamics: scale the learning rate accordingly.
- `tf32: True` (default) speeds up the matrix multiplications on Ampere or newer GPUs.
- Multiple GPUs: one GPU is enough to saturate a single small training, and `--gpus 0,1` (DataParallel) is usually *slower*. Use the GPUs to run several trainings (seeds) at the same time with `run_training.sh --gpus-per-node`. Several trainings can also share one GPU.
- K-fold with large datasets: every fold is a separate process that loads all the events into RAM. The number of folds that can run at the same time on a node is usually limited by its RAM, not by its GPUs. With the k-fold cache (the default in `run_training.sh`) the events are loaded once and shared, so all the folds can run on one node. For example, `-k 20 -g 8 --cpus 40 --mem-per-cpu 6000` runs 20 folds on one 8-GPU node; see `configs/hh4b_bkg_reweighting/DNN_12_01_03_*_kfold20.yml`. Measure the memory with `sacct -j <job id> --format=MaxRSS`.
- After all trainings finish, the best ONNX model from each run is copied to `best_models/` and `ml_onnx` is called to aggregate them. With `--ratio` the `-ar` flag is passed (average ratio, used for background reweighting); without it, a plain aggregation is performed.
- The input variable name is read automatically from the training config YAML (`input_variables` field).
- If `jobs/comet_token.key` exists, Comet ML logging is enabled automatically (see [COMET integration](../README.md#comet-integration)).

## Legacy per-use-case scripts

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

