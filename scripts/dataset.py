import uproot
import numpy as np
import torch
import math
import logging
import os
from coffea.util import load
import sys

sys.path.append("../")
from configs.DNN_input_lists import *

logger = logging.getLogger(__name__)


def get_variables(files, dimension, args, sample_list, sig_bkg, data_format="root"):
    for i, file in enumerate(files):
        logger.info(f"Loading file {file}")
        if data_format == "root":
            # open each file and get the Events tree using uproot
            file = uproot.open(f"{file}:Events")
            variables_array = np.array(
                [file[input].array(library="np") for input in DNN_input_variables]
            )
        elif data_format == "coffea":
            variables_dict = {}
            file = load(file)
            for sample in sample_list:
                for dataset in list(file["columns"][sample].keys())[0]:
                    for category in list(file["columns"][sample][dataset].keys())[0]:
                        vars = file["columns"][sample][dataset][category]
                        weights=file["columns"][sample][dataset][category]["weight"].value
                        # break
            for k in vars.keys():
                if k in DNN_input_variables:
                    variables_dict[k] = np.array(vars[k].value)
            variables_array = np.array(
                [variables_dict[input] for input in DNN_input_variables]
            )
            variables_array = np.append(variables_array, weights, axis=0)


        # concatenate all the variables into a single torch tensor
        if i == 0:
            variables = torch.tensor(variables_array, dtype=torch.float32)[
                :, : math.ceil(dimension)
            ]
        else:
            variables = torch.cat(
                (
                    variables,
                    torch.tensor(variables_array, dtype=torch.float32),
                ),
                dim=1,
            )[:, : math.ceil(dimension)]

    logger.info(f"number of {sig_bkg} events: {variables.shape[1]}")

    # sum of weights
    sumw = variables[-1].sum()
    logger.info(f"sum of weights {sig_bkg}: {sumw}")

    # multiply the weights by the weight factor
    variables[-1] = variables[-1] * args.weights[0]

    sumw = variables[-1].sum()
    logger.info(f"sum of weights {sig_bkg}: {sumw}")

    flag_tensor = (
        torch.ones_like(variables[0], dtype=torch.float32).unsqueeze(0)
        if sig_bkg == "signal"
        else torch.zeros_like(variables[0], dtype=torch.float32).unsqueeze(0)
    )

    X = (variables, flag_tensor)
    return X


def load_data(args, data_format="root"):
    batch_size = args.batch_size
    logger.info(f"Batch size: {batch_size}")

    dirs = args.data_dirs

    dimension = (args.train_size + args.val_size + args.test_size) / 2
    logger.info("Variables: %s", DNN_input_variables)

    # list of signal and background files
    sig_files = []
    bkg_files = []

    if data_format == "root":
        for x in dirs:
            files = os.listdir(x)
            for file in files:
                for signal in signal_list:
                    if signal in file and "SR" in file:
                        sig_files.append(x + file)
                for background in (
                    background_list_noVV if args.noVV else background_list
                ):
                    if background in file and "SR" in file:
                        bkg_files.append(x + file)
    elif data_format == "coffea":
        sig_files=[dir + "/output_all.coffea" for dir in dirs]
        bkg_files=[dir + "/output_all.coffea" for dir in dirs]


    X_sig = get_variables(sig_files, dimension, args, signal_list, "signal", data_format)
    X_bkg = get_variables(bkg_files, dimension, args, background_list, "background", data_format)

    X_fts = torch.cat((X_sig[0], X_bkg[0]), dim=1).transpose(1, 0)
    X_lbl = torch.cat((X_sig[1], X_bkg[1]), dim=1).transpose(1, 0)

    # split the dataset into training and val sets
    if args.train_size != -1 and args.val_size != -1 and args.test_size != -1:
        X_fts = X_fts[: args.train_size + args.val_size + args.test_size, :]
        X_lbl = X_lbl[: args.train_size + args.val_size + args.test_size, :]

    X = torch.utils.data.TensorDataset(X_fts, X_lbl)

    train_size = int(0.8 * len(X)) if args.train_size == -1 else args.train_size
    val_size = math.ceil((len(X) - train_size) / 2)
    test_size = math.floor((len(X) - train_size) / 2)

    logger.info(f"Total size: {len(X)}")
    logger.info(f"Training size: {train_size}")
    logger.info(f"Validation size: {val_size}")
    logger.info(f"Test size: {test_size}")

    gen = torch.Generator()
    gen.manual_seed(0)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        X, [train_size, val_size, test_size], generator=gen
    )

    # check size of the dataset
    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    training_loader = None
    val_loader = None
    test_loader = None

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    logger.info("Training loader size: %d", len(training_loader))

    if not args.eval_model:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=args.pin_memory,
        )
        logger.info("Validation loader size: %d", len(val_loader))

    if args.eval or args.eval_model:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=args.pin_memory,
        )
        logger.info("Test loader size: %d", len(test_loader))

    return (
        training_loader,
        val_loader,
        test_loader,
        train_size,
        val_size,
        test_size,
        X_fts,
        X_lbl,
        batch_size,
    )
