import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--config",
    required=False,
    default=None,
    help="Path to the configuration file",
    type=str,
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=None,
    help="Batch size",
    type=int,
)
parser.add_argument(
    "-e",
    "--epochs",
    default=None,
    help="Number of epochs",
    type=int,
)
parser.add_argument(
    "-n",
    "--num-workers",
    default=None,
    help="Number of workers for data loading",
    type=int,
)
parser.add_argument(
    "-d",
    "--data-dirs",
    nargs="+",
    default=None,
    help="Directory for data",
)
parser.add_argument(
    "-ev",
    "--eval",
    action="store_true",
    help="Evaluate the model on test dataset after training",
    default=None,
)
parser.add_argument(
    "-sm",
    "--save-model",
    action="store_true",
    help="Save also the model next to the state dict",
    default=False,
)
parser.add_argument(
    "-ct",
    "--comet-token",
    required=False,
    default=None,
    help="Token to register job on comet for job monitoring (`comet.com`)",
    type=str,
)
parser.add_argument(
    "-cn",
    "--comet-name",
    required=False,
    default=None,
    help="Username to register job on comet for job monitoring (`comet.com`)",
    type=str,
)
parser.add_argument(
    "-cw",
    "--comet-workspace",
    required=False,
    default=None,
    help="Workspace in which to register the model",
    type=str,
)
parser.add_argument(
    "-ctg",
    "--comet-tags",
    required=False,
    nargs="+",
    default=None,
    help="Username to register job on comet for job monitoring (`comet.com`)",
    type=str,
)
parser.add_argument(
    "-g", "--gpus", default=None, help="GPU numbers separated by a comma. Use 'cpu' for CPU", type=str
)
parser.add_argument(
    "--histos", default=None, help="Make histograms of sig and bkg output distribution", action="store_true"
)
parser.add_argument(
    "--roc", default=None, help="Make roc curve of discrimination", action="store_true"
)
parser.add_argument(
    "--input-plots",
    dest="input_plots",
    default=None,
    help="Plot the input variable distributions of signal and background before training (default: True)",
    action="store_true",
)
parser.add_argument(
    "--no-input-plots",
    dest="input_plots",
    default=None,
    help="Do not plot the input variable distributions of signal and background",
    action="store_false",
)
parser.add_argument(
    "--input-plots-dir",
    default=None,
    help="Subdirectory of the output directory where the input variable plots are saved",
    type=str,
)
parser.add_argument(
    "--input-plots-bins",
    default=None,
    help="Number of bins of the input variable histograms",
    type=int,
)
parser.add_argument(
    "--input-plots-log",
    default=None,
    help="Save also the input variable histograms with a logarithmic y axis",
    action="store_true",
)
parser.add_argument(
    "--history", default=None, help="Plot training history", action="store_true"
)
parser.add_argument(
    "-em",
    "--eval-model",
    default=None,
    help="Path to model to evaluate (should be the state at the best epoch after training)",
    type=str,
)
parser.add_argument(
    "-l",
    "--load-model",
    default=None,
    help="Path to model to load and continue training. The model should be the state at the best epoch",
    type=str,
)
parser.add_argument(
    "-mp",
    "--ml-model-path",
    default=None,
    help="Path to the ML_model.py file to use when loading/evaluating a model (default: <output_dir>/ML_model.py)",
    type=str,
)
parser.add_argument(
    "--onnx",
    default=None,
    help="Save model in ONNX format",
    action="store_true",
)
parser.add_argument(
    "--pin-memory",
    default=None,
    help="Pin memory for data loading",
    action="store_true",
)
parser.add_argument(
    "--data-on-gpu",
    dest="data_on_gpu",
    default=None,
    help="Keep the whole datasets on the GPU and take the batches from there (default: True)",
    action="store_true",
)
parser.add_argument(
    "--no-data-on-gpu",
    dest="data_on_gpu",
    default=None,
    help="Use the standard DataLoaders, copying every batch to the GPU",
    action="store_false",
)
parser.add_argument(
    "--overwrite",
    default=None,
    help="Overwrite output directory",
    action="store_true",
)
parser.add_argument(
    "-o",
    "--output-dir",
    default=None,
    help="Output directory",
    type=str,
)
parser.add_argument(
    "-s",
    "--seed",
    default=None,
    help="Seed for event shuffling and weights initialization",
    type=str,
)
parser.add_argument(
    "-s-n",
    "--save-numpy",
    default=None,
    help="Save numpy arrays of the output scores",
    action="store_true",
)
parser.add_argument(
    "-k",
    "--k-folds",
    default=None,
    help="Number of folds of the k-fold training (0 or 1 to disable)",
    type=int,
)
parser.add_argument(
    "-f",
    "--fold",
    default=None,
    help="Train only this fold of the k-fold training (default: all the folds)",
    type=int,
)
parser.add_argument(
    "--kfold-split",
    default=None,
    choices=["random", "event"],
    help="How to assign the events to the folds: random or by event number (event %% k_folds)",
    type=str,
)
parser.add_argument(
    "--onnx-ratio",
    default=None,
    help="K-fold: build also the ONNX model of the average of the ratios of the outputs of the folds (reweighting)",
    action="store_true",
)

# parser.print_help()
args = parser.parse_args()
