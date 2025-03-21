# ML_pytorch

Repository with basic machine learning algorithms implemented in PyTorch.

# Installation
To create the micromamba environment, you can use the following command:
```bash
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00  -p gpu --gres=gpu:1
micromamba env create -f ML_pytorch_env.yml
micromamba activate ML_pytorch
pip install -r requirements.txt
# install the package in editable mode
pip install -e .
```

# Connect to node with a gpu
To connect to a node with a gpu, you can use the following command:
```bash
# connect to a node with a gpu
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00  -p gpu --gres=gpu:1
# activate the environment
micromamba activate ML_pytorch
# check which gpu is available
echo $CUDA_VISIBLE_DEVICES # or echo $SLURM_JOB_GPUS
```

# Examples
To execute an example training, evaluate the model on the test set, plot the history and plot the signal/background histograms, you can use the following command:

```bash
python  ml_pytorch/scripts/train.py  -c configs/example_DNN_config_ggF_VBF.yml
```
