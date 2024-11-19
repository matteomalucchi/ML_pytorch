import uproot
import numpy as np
import torch
import math
import logging
import os
from coffea.util import load
import sys
import awkward as ak

sys.path.append("../")
# from configs.DNN_input_lists_ggF_VBF import *

logger = logging.getLogger(__name__)


def get_variables(files, dimension, args, input_variables,sample_list, sig_bkg, data_format="root"):
    for i, file_name in enumerate(files):
        logger.info(f"Loading file {file_name}")
        if data_format == "root":
            # open each file and get the Events tree using uproot
            file = uproot.open(f"{file_name}:Events")
            variables_array = np.array(
                [file[input].array(library="np") for input in input_variables]
            )
        elif data_format == "coffea":
            variables_dict = {}
            file = load(file_name)
            # print(file)
            for sample in sample_list:
                print("sample", sample)
                for dataset in list(file["columns"][sample].keys()):
                    print("dataset", dataset)
                    for category in list(file["columns"][sample][dataset].keys()):
                        print("category", category)
                        print(
                            "variables",
                            file["columns"][sample][dataset][category].keys(),
                        )
                        vars = file["columns"][sample][dataset][category]
                        weights = file["columns"][sample][dataset][category][
                            "weight"
                        ].value
                        # break
            for k in vars.keys():
                if k in input_variables:
                    # unflatten all the jet variables
                    collection = k.split("_")[0]
                    number_per_event = tuple(vars[f"{collection}_N"].value)

                    #TODO: fix the padding here
                    variables_dict[k] = ak.to_numpy(
                        ak.pad_none(
                            ak.unflatten(vars[k].value, number_per_event), 4, clip=True
                        )
                    )

            weights = np.expand_dims(vars["weight"].value, axis=0)
            variables_array = np.swapaxes(
                np.concatenate(
                    [variables_dict[input] for input in input_variables], axis=1
                ),
                0,
                1,
            )
            print("variables_array", variables_array.shape)
            print("weights", weights.shape)
            variables_array = np.append(variables_array, weights, axis=0)
            print("variables_array complete", variables_array.shape)

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

    flag_tensor = (
        torch.ones_like(variables[0], dtype=torch.float32).unsqueeze(0)
        if sig_bkg == "signal"
        else torch.zeros_like(variables[0], dtype=torch.float32).unsqueeze(0)
    )

    X = (variables, flag_tensor)
    return X


def load_data(args, data_format, input_variables, signal_list, background_list):
    batch_size = args.batch_size
    logger.info(f"Batch size: {batch_size}")

    dirs = args.data_dirs

    dimension = (args.train_size + args.val_size + args.test_size) / 2
    logger.info("Variables: %s", input_variables)

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
                for background in background_list:
                    if background in file and "SR" in file:
                        bkg_files.append(x + file)
    elif data_format == "coffea":
        sig_files = [dir + "/output_all.coffea" for dir in dirs]
        bkg_files = [dir + "/output_all.coffea" for dir in dirs]

    X_sig = get_variables(
        sig_files, dimension, args, input_variables, signal_list, "signal", data_format
    )
    X_bkg = get_variables(
        bkg_files, dimension, args,input_variables,  background_list, "background", data_format
    )

    # sum of weights
    sumw_sig = X_sig[0][-1].sum()
    sumw_bkg = X_bkg[0][-1].sum()
    print("sig weight", X_sig[0][-1][30:100], torch.all( (X_sig[0][-1]>=50.8362) & (X_sig[0][-1]<=50.8364), axis=0))
    logger.info(f"sum of weights before rescaling signal: {sumw_sig}")
    print("bkg weight", X_bkg[0][-1])
    logger.info(f"sum of weights before rescaling backgound: {sumw_bkg}")


    if args.weights:
        sig_weight = args.weights[0]
        bkg_weight = args.weights[1]
    else:
        #compute class weights such that sumw is the same for signal and background and each weight is order of 1
        # sig_weight = num_events_sig/sumw_sig
        # bkg_weight = num_events_bkg/sumw_bkg

        # Compute the effective class count
        # https://arxiv.org/pdf/1901.05555.pdf

        num_events_sig= X_sig[0].shape[1]
        sig_event_weights=X_sig[0][-1]
        beta_sig=1 - (1 /sig_event_weights.sum())
        sig_class_weights = (1-beta_sig) / (1 - beta_sig**num_events_sig)

        print(num_events_sig, sig_event_weights.sum(), beta_sig, sig_class_weights)

        num_events_bkg= X_bkg[0].shape[1]
        bkg_event_weights=X_bkg[0][-1]
        beta_bkg=1 - (1 /bkg_event_weights.sum())
        bkg_class_weights = (1-beta_bkg) / (1 - beta_bkg**num_events_bkg)

        print(num_events_bkg, bkg_event_weights.sum(), beta_bkg, bkg_class_weights)



    X_sig[0][-1]=X_sig[0][-1]*sig_class_weights
    X_bkg[0][-1]=X_bkg[0][-1]*bkg_class_weights

    print("\n rescale weights \n")
    # sum of weights
    sumw_sig = X_sig[0][-1].sum()
    sumw_bkg = X_bkg[0][-1].sum()
    print("sig weight", X_sig[0][-1])
    logger.info(f"sum of weights before rescaling signal: {sumw_sig}")
    print("bkg weight", X_bkg[0][-1])
    logger.info(f"sum of weights before rescaling backgound: {sumw_bkg}")


    raise ValueError("stop here")

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
