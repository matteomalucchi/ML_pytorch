# ML_pytorch

Repository with basic machine learning algorithms implemented in PyTorch. 

The coffea files used as inputs are based on the output of [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea/tree/main). In particular, the framework was developed based on the output of the 
[AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs) repository, which is a collection of analysis configurations for the PocketCoffea framework.


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
ml_train  -c configs/example_DNN_config_ggF_VBF.yml
```

To execute either a 20x training for background reweighting or to run a `sig_bkg_classifier` model, there are two scripts that can be run with slurm:
```bash
# Outside of any node activate your environment (e.g. `micromamba activate ML_pytorch`)
cd jobs/
# If the output folder is not provided, it will have the same name as the config file without the extension
# For 20x training:
sbatch run_20_trainings_in_4_parallel.sh <config_file> <output_folder>
# For sig_bkg_reweighting
sbatch run_sig_bkg_classifier.sh <config_file> <output_folder>
```


To execute 5 runs in a node without the interactive access to the GPU node (the given config and folder names are just examples):
```bash
# Outside of any node activate your environment (e.g. `micromamba activate ML_pytorch`)

# Then run this command:
sbatch --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 5000 --time 12:00:00  -p gpu --gres=gpu:1 --wrap=". ./run_batch_of_5.sh /work/tharte/datasets/ML_pytorch/configs/bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE.yml out/bkg_reweighting/SPANET_ptFlat_20_runs_postEE 0"
```

Additionally, there are now options to send the metrics of the training to [COMET](https://www.comet.com/site) (academics accounts are available for free):
To set it up together with the files mentioned above:
```bash
# Create file with token and username:
touch comet_token.key
# open the file with the editor of your choice
vim comet_token.key
# in the first line write your username, and in the second line, write your token (to be retrieved on the website):
# <uname>
# <token>
```
The scripts will read this file if it exists and automatically sends the information to `ml_pytorch`


