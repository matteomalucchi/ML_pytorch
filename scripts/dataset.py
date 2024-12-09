import uproot
import numpy as np
import torch
import math
import logging
import os
from coffea.util import load
import sys
import awkward as ak


logger = logging.getLogger(__name__)


def get_variables(
    files,
    dimension,
    args,
    input_variables,
    sample_list,
    dataset_sample,
    region,
    sig_bkg,
    data_format="root",
):
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
            vars = None
            weights = None
            for sample in sample_list:
                for dataset in list(file["columns"][sample].keys()):
                    print(f"dataset {dataset}")
                    if dataset == dataset_sample:
                        for category in list(file["columns"][sample][dataset].keys()):
                            print(f"category {category}")
                            if category == region:
                                vars = file["columns"][sample][dataset][category]
                                weights = file["columns"][sample][dataset][category][
                                    "weight"
                                ].value
            if not vars:
                logger.error(
                    f"region {region} not found in dataset {dataset_sample} for sample {sample_list}"
                )
                raise ValueError

            for k in input_variables:
                # unflatten all the jet variables
                collection = k.split("_")[0]

                # check if collection_N is present to unflatten the variables
                if f"{collection}_N" in vars.keys():
                    number_per_event = tuple(vars[f"{collection}_N"].value)
                    if ak.all(number_per_event == number_per_event[0]):
                        variables_dict[k] = ak.to_numpy(
                            ak.unflatten(vars[k].value, number_per_event)
                        )
                    else:
                        logger.warning(
                            f"number of {collection} jets per event is not the same for all events"
                        )
                        variables_dict[k] = ak.to_numpy(
                            ak.pad_none(
                                ak.unflatten(vars[k].value, number_per_event),
                                6,
                                clip=True,
                            )
                        )
                else:
                    variables_dict[k] = ak.to_numpy(ak.unflatten(vars[k].value, 1))

            weights = np.expand_dims(vars["weight"].value, axis=0)
            variables_array = np.swapaxes(
                np.concatenate(
                    [variables_dict[input] for input in input_variables], axis=1
                ),
                0,
                1,
            )
            logger.info(f"variables_array {variables_array.shape}")
            logger.info(f"weights {weights.shape}")
            variables_array = np.append(variables_array, weights, axis=0)
            logger.info(f"variables_array complete {variables_array.shape}")
            print(variables_array)
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
    print(X, X[0].shape)
    return X


def load_data(args, cfg):
    batch_size = args.batch_size if args.batch_size else cfg.batch_size
    logger.info(f"Batch size: {batch_size}")

    dirs = args.data_dirs

    dimension = (args.train_size + args.val_size + args.test_size) / 2
    logger.info("Variables: %s", cfg.input_variables)

    # list of signal and background files
    sig_files = []
    bkg_files = []

    if cfg.data_format == "root":
        for x in dirs:
            files = os.listdir(x)
            for file in files:
                for signal in cfg.signal_list:
                    if signal in file and "SR" in file:
                        sig_files.append(x + file)
                for background in cfg.background_list:
                    if background in file and "SR" in file:
                        bkg_files.append(x + file)
    elif cfg.data_format == "coffea":
        sig_files = [dir + "/output_all.coffea" for dir in dirs]
        bkg_files = [dir + "/output_all.coffea" for dir in dirs]

    X_sig = get_variables(
        sig_files,
        dimension,
        args,
        cfg.input_variables,
        cfg.signal_list,
        cfg.dataset_signal,
        cfg.region,
        "signal",
        cfg.data_format,
    )
    X_bkg = get_variables(
        bkg_files,
        dimension,
        args,
        cfg.input_variables,
        cfg.background_list,
        cfg.dataset_background,
        cfg.region,
        "background",
        cfg.data_format,
    )

    # sum of weights
    sumw_sig = X_sig[0][-1].sum()
    sumw_bkg = X_bkg[0][-1].sum()
    logger.info(f"sum of weights before rescaling signal: {sumw_sig}")
    logger.info(f"sum of weights before rescaling backgound: {sumw_bkg}")

    if args.weights:
        sig_class_weights = args.weights[0]
        bkg_class_weights = args.weights[1]
    else:
        # compute class weights such that sumw is the same for signal and background and each weight is order of 1
        num_events_sig = X_sig[0].shape[1]
        num_events_bkg = X_bkg[0].shape[1]
        if True:
            sig_class_weights = (num_events_sig + num_events_bkg) / sumw_sig
            bkg_class_weights = (num_events_sig + num_events_bkg) / sumw_bkg
        else:
            # Compute the effective class count
            # https://arxiv.org/pdf/1901.05555.pdf

            sig_event_weights = X_sig[0][-1]
            beta_sig = 1 - (1 / sig_event_weights.sum())
            sig_class_weights = (1 - beta_sig) / (1 - beta_sig**num_events_sig)

            logger.info(
                f"num event sig {num_events_sig}, sig_event_weights {sig_event_weights.sum()}, beta_sig {beta_sig}, sig_class_weights {sig_class_weights}"
            )

            bkg_event_weights = X_bkg[0][-1]
            beta_bkg = 1 - (1 / bkg_event_weights.sum())
            bkg_class_weights = (1 - beta_bkg) / (1 - beta_bkg**num_events_bkg)

            logger.info(
                f"num event bkg {num_events_bkg}, bkg_event_weights {bkg_event_weights.sum()}, beta_bkg {beta_bkg}, bkg_class_weights {bkg_class_weights}"
            )

    X_sig[0][-1] = X_sig[0][-1] * sig_class_weights
    X_bkg[0][-1] = X_bkg[0][-1] * bkg_class_weights

    # sum of weights
    sumw_sig = X_sig[0][-1].sum()
    sumw_bkg = X_bkg[0][-1].sum()
    logger.info(f"sum of weights after rescaling signal: {sumw_sig}")
    logger.info(f"sum of weights after rescaling backgound: {sumw_bkg}")

    X_fts = torch.cat((X_sig[0], X_bkg[0]), dim=1).transpose(1, 0)
    X_lbl = torch.cat((X_sig[1], X_bkg[1]), dim=1).transpose(1, 0)

    logger.info(f"X_fts shape: {X_fts.shape}")
    logger.info(f"X_lbl shape: {X_lbl.shape}")

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
