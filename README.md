# ML_pytorch

Repository with basic machine learning algorithms implemented in PyTorch.

# Installation
To create the micromamba environment, you can use the following command:
```bash
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00  -p gpu
micromamba env create -f ML_pytorch_env.yml
micromamba activate ML_pytorch
pip install -r requirements.txt
```

# Connect to node with a gpu
To connect to a node with a gpu, you can use the following command:
```bash
# connect to a node with a gpu
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00  -p gpu
# activate the environment
micromamba activate ML_pytorch
```

# Examples
To execute a training, evaluate the model on the test set, plot the history and plot the signal/background histograms, you can use the following command:

```bash
python  scripts/train.py -d /work/mmalucch/out_hh4b/out_vbf_ggf_dnn_full/ -o out/name_of_training --eval --onnx --roc --histos --history --gpus 7 -n 4 -e 10 -c configs/DNN_config_ggF_VBF.yml
```
