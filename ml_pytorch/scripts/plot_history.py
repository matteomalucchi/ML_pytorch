import argparse
from scipy.ndimage import uniform_filter1d
import os
import numpy as np
from glob import glob
import matplotlib

import mplhep as hep
from utils_configs.plot.HEPPlotter import HEPPlotter

from ml_pytorch.utils.lumitext import ENERGY_TEXT, get_lumitext_from_dir

LUMITEXT = ENERGY_TEXT

# CMS colour palette of mplhep, used consistently in all the plotting scripts
CMS_COLORS = [cycle["color"] for cycle in hep.style.CMS["axes.prop_cycle"]]


def read_from_txt(file):
    # get accuracy and loss and separate them between training and validation
    with open(file, "r") as f:
        lines = f.readlines()
        train_accuracy = []
        train_loss = []
        val_accuracy = []
        val_loss = []
        lr = []
        i = 0
        j = 0
        for line in lines:
            if "Training batch" in line and not "batch 0." in line:
                train_accuracy.append(float(line.split("accuracy: ")[1].split(" ")[0]))
                train_loss.append(float(line.split("loss: ")[1].split("\n")[0]))
                i += 1
            elif "Validation batch" in line and not "batch 0." in line:
                val_accuracy.append(float(line.split("accuracy: ")[1].split(" ")[0]))
                val_loss.append(float(line.split("loss: ")[1].split("\n")[0]))
                j += 1
                if j > 10000:
                    print(line)
            elif "learning rate" in line:
                lr.append(float(line.split("rate: ")[1].split("\n")[0]))

    print("len train accuracy: ", len(train_accuracy))
    print("len train loss: ", len(train_loss))
    print("len val accuracy: ", len(val_accuracy))
    print("len val loss: ", len(val_loss))
    print("\n")
    print("Training loss:")
    print(train_loss)
    print("Validation loss:")
    print(val_loss)
    return train_accuracy, train_loss, val_accuracy, val_loss, lr


def history_graph(values, n_train, color, linestyle, uniform_filter):
    """Graph of the smoothed values as a function of the epoch (10 entries per
    epoch in the training curve of length n_train; the validation curve is
    stretched to the same epoch range)."""
    return {
        "data": {
            "x": [np.linspace(0, n_train / 10, len(values)), None],
            "y": [uniform_filter1d(np.asarray(values, dtype=float), size=uniform_filter), None],
        },
        "style": {"color": color, "linestyle": linestyle, "markersize": 0},
    }


def run_history_plotter(
    series_dict, output, show, lumitext, ylabel="", figsize=None, vlines=(), **options
):
    """vlines: kwargs of ax.axvline of the vertical lines to draw (e.g. x, color)."""
    plotter = (
        HEPPlotter("CMS")
        .set_plot_config(lumitext=lumitext, figsize=figsize)
        .set_output(output)
        .set_labels(xlabel="Epoch", ylabel=ylabel)
        .set_data(series_dict, plot_type="graph")
        .set_options(grid=True, set_ylim=False, **options)
    )
    for line in vlines:
        plotter.add_line("v", **line)
    if show:
        plotter.show()
    plotter.run()


def plot_history(
    train_accuracy,
    train_loss,
    val_accuracy,
    val_loss,
    dir,
    show,
    uniform_filter=10,
    lenght=-1,
    comet_logger=None,
    lumitext=LUMITEXT,
):
    # save the full history (read e.g. by kfold_merge to plot all the folds)
    np.savez(
        f"{dir}/history.npz",
        train_accuracy=np.asarray(train_accuracy, dtype=float),
        train_loss=np.asarray(train_loss, dtype=float),
        val_accuracy=np.asarray(val_accuracy, dtype=float),
        val_loss=np.asarray(val_loss, dtype=float),
    )
    infos_dict = {
        "accuracy": {"train": train_accuracy[:lenght], "val": val_accuracy[:lenght]},
        "loss": {"train": train_loss[:lenght], "val": val_loss[:lenght]},
    }
    line_style = {
        "accuracy": "--",
        "loss": "-",
    }

    # one graph per curve: the smoothed values as a function of the epoch
    series_dict = {}
    for type, info in infos_dict.items():
        n_train = len(info["train"])
        if n_train == 0:
            continue
        series_dict[f"Training {type}"] = history_graph(
            info["train"], n_train, CMS_COLORS[0], line_style[type], uniform_filter
        )
        if len(info["val"]) == 0:
            continue
        series_dict[f"Validation {type}"] = history_graph(
            info["val"], n_train, CMS_COLORS[1], line_style[type], uniform_filter
        )

    if not series_dict:
        print("WARNING: no history to plot")
        return

    run_history_plotter(
        series_dict,
        f"{dir}/history",
        show,
        lumitext,
        legend_loc="center right",
        split_legend=False,
    )

    if comet_logger:
        comet_logger.log_image(f"{dir}/history.png", name="history")


def plot_history_folds(fold_dirs, out_dir, show=False, uniform_filter=10, lumitext=LUMITEXT):
    """Training / validation loss / accuracy of all the folds of a k-fold
    training, one plot per curve (<out_dir>/history_<train|val>_<loss|accuracy>),
    read from the fold_<k>/history.npz saved by plot_history."""
    histories = {}
    for k, d in enumerate(fold_dirs):
        if os.path.exists(f"{d}/history.npz"):
            histories[k] = np.load(f"{d}/history.npz")
        else:
            print(f"WARNING: no {d}/history.npz, fold {k} not in the history plots")
    # 20 distinct colours: the dark shades of tab20 first, then the light ones
    tab20 = matplotlib.colormaps["tab20"].colors
    colors = tab20[0::2] + tab20[1::2]
    for split, split_label in [("train", "Training"), ("val", "Validation")]:
        for type, legend_loc in [("loss", "upper right"), ("accuracy", "lower right")]:
            series_dict = {
                f"Fold {k}": history_graph(
                    h[f"{split}_{type}"],
                    len(h[f"train_{type}"]),
                    colors[k % len(colors)],
                    "-" if k < len(colors) else "--",
                    uniform_filter,
                )
                for k, h in histories.items()
                if len(h[f"{split}_{type}"]) and len(h[f"train_{type}"])
            }
            if not series_dict:
                continue
            # dashed vertical line at the last epoch of every fold, in its colour
            vlines = [
                {
                    "x": series["data"]["x"][0][-1],
                    "color": series["style"]["color"],
                    "linestyle": "--",
                    "linewidth": 1,
                }
                for series in series_dict.values()
            ]
            # big figure and legend in 2 columns to fit up to 20 folds
            run_history_plotter(
                series_dict,
                f"{out_dir}/history_{split}_{type}",
                show,
                lumitext,
                ylabel=f"{split_label} {type}",
                figsize=(16, 12),
                vlines=vlines,
                legend_loc=legend_loc,
                split_legend=True,
                legend_font_size=28,
            )


def plot_lr(lr, main_dir, show, comet_logger=None, lumitext=LUMITEXT):
    if len(lr) == 0:
        print("WARNING: no learning rate to plot")
        return

    series_dict = {
        "Learning rate": {
            "data": {
                "x": [np.arange(len(lr)), None],
                "y": [np.asarray(lr, dtype=float), None],
            },
            "style": {"color": CMS_COLORS[0], "linestyle": "-", "markersize": 0},
        }
    }

    run_history_plotter(
        series_dict, f"{main_dir}/lr", show, lumitext, ylabel="Learning rate",
        y_log=True, legend=False,
    )

    if comet_logger:
        comet_logger.log_image(f"{main_dir}/lr.png", name="learning_rate")


def main():
    # plot the history for the training and validation losses and accuracies
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, help="path to log file")
    parser.add_argument(
        "-u",
        "--uniform-filter",
        default=10,
        type=int,
        help="size of the uniform filter",
    )
    parser.add_argument(
        "-s",
        "--show",
        default=False,
        action="store_true",
        help="show plots",
    )
    parser.add_argument(
        "-l",
        "--lenght",
        default=-1,
        help="max lenght of the plot",
        type=int,
    )
    args = parser.parse_args()

    # find the file starting with logger in args.input_path using os.listdir
    log_file = args.input_path

    print(log_file)

    train_accuracy, train_loss, val_accuracy, val_loss, lr = read_from_txt(log_file)

    # get the years from the config saved in the training directory
    lumitext = get_lumitext_from_dir(os.path.dirname(args.input_path))

    plot_history(
        train_accuracy,
        train_loss,
        val_accuracy,
        val_loss,
        os.path.dirname(args.input_path),
        args.show,
        args.uniform_filter,
        args.lenght,
        lumitext=lumitext,
    )

    plot_lr(lr, os.path.dirname(args.input_path), args.show, lumitext=lumitext)


if __name__ == "__main__":
    main()
