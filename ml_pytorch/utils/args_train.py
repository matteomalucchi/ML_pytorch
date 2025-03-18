import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--config",
    required=True,
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
    help="Evaluate the model",
    default=False,
)
parser.add_argument(
    "-os",
    "--oversample",
    action="store_true",
    help="If true, the signal samples are copied multiple times to match number of background samples",
    default=None,
)
parser.add_argument(
    "-g", "--gpus", default=None, help="GPU numbers separated by a comma", type=str
)
parser.add_argument(
    "--histos", default=None, help="Make histograms", action="store_true"
)
parser.add_argument(
    "--roc", default=None, help="Make roc curve", action="store_true"
)
parser.add_argument(
    "--history", default=None, help="Plot history", action="store_true"
)
parser.add_argument(
    "-em",
    "--eval-model",
    default=None,
    help="Path to model to evaluate",
    type=str,
)
parser.add_argument(
    "-l",
    "--load-model",
    default=None,
    help="Path to model to load and continue training",
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


# parser.print_help()
args = parser.parse_args()
