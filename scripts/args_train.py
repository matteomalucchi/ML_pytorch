import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-b",
    "--batch-size",
    default=0,
    help="Batch size",
    type=int,
)
parser.add_argument(
    "-e",
    "--epochs",
    default=10,
    help="Number of epochs",
    type=int,
)
parser.add_argument(
    "-n",
    "--num-workers",
    default=4,
    help="Number of workers for data loading",
    type=int,
)
parser.add_argument(
    "-d",
    "--data-dirs",
    nargs="+",
    default=[
    ],
    help="Directory for data",
)
parser.add_argument(
    "--eval",
    action="store_true",
    help="Evaluate the model",
    default=False,
)
parser.add_argument(
    "-g", "--gpus", default="", help="GPU numbers separated by a comma", type=str
)
parser.add_argument(
    "--histos", default=False, help="Make histograms", action="store_true"
)
parser.add_argument(
    "--roc", default=False, help="Make roc curve", action="store_true"
)
parser.add_argument(
    "--history", default=False, help="Plot history", action="store_true"
)
parser.add_argument(
    "--eval-model",
    default="",
    help="Path to model to evaluate",
    type=str,
)
parser.add_argument(
    "-l",
    "--load-model",
    default="",
    help="Path to model to load and continue training",
    type=str,
)
parser.add_argument(
    "--onnx",
    default=False,
    help="Save model in ONNX format",
    action="store_true",
)
parser.add_argument(
    "--pin-memory",
    default=False,
    help="Pin memory for data loading",
    action="store_true",
)
parser.add_argument(
    "--overwrite",
    default=False,
    help="Overwrite output directory",
    action="store_true",
)
parser.add_argument(
    "-o",
    "--output-dir",
    default="",
    help="Output directory",
    type=str,
)
parser.add_argument(
    "-c",
    "--config",
    default="../configs/DNN_input_lists_ggF_VBF.yml",
    help="Path to the configuration file",
    type=str,
)


# parser.print_help()
args = parser.parse_args()
