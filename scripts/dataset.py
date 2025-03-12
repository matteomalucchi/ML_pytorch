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
    total_fraction_of_events,
    input_variables,
    sample_list,
    dataset_sample,
    region,
    sig_bkg,
    data_format="root",
):
    tot_lenght = 0
    for i, file_name in enumerate(files):
        logger.info(f"Loading file {file_name}")
        if data_format == "root":
            # open each file and get the Events tree using uproot
            file = uproot.open(f"{file_name}:Events")
            variables_array = np.array(
                [file[input].array(library="np") for input in input_variables]
            )
        elif data_format == "coffea":
            vars = []
            weights = []
            logger.info(f"Currently working on file: {file_name}")
            variables_dict = {}
            file = load(file_name)
            print(f"sample_list: {sample_list}")
            for sample in sample_list:
                logger.info("sample %s", sample)
                print(list(file["columns"].keys()))
                if sample in list(file["columns"].keys()):
                    print(f"sample {sample} in file")
                    for dataset in list(file["columns"][sample].keys()):
                        print(f"searching dataset {dataset} in dataset_sample {dataset_sample}")
                        if dataset in dataset_sample:
                            logger.info("dataset %s", dataset)
                            for category in list(file["columns"][sample][dataset].keys()):
                                if category == region:
                                    logger.info("category %s", category)
                                    vars.append(file["columns"][sample][dataset][category])
                                    weights.append(
                                        file["columns"][sample][dataset][category][
                                            "weight"
                                        ].value
                                        / (file["sum_genweights"][dataset] if dataset in file["sum_genweights"] else 1)
                                    )
                                    if dataset in file["sum_genweights"]:
                                        logger.info(
                                            f"original weight: {file['columns'][sample][dataset][category]['weight'].value[0]}"
                                        )
                                        logger.info(f"sum_genweights: {file['sum_genweights'][dataset]}")
                                    logger.info(f"weight: {weights[0]}")
            if len(vars)<1:
                logger.error(
                    f"region {region} not found in dataset {dataset_sample} for sample {sample_list}"
                )
                raise ValueError
            
            logger.info(f"Found datasets: {len(vars)}")
            # Merge multiple lists:
            keys = set().union(*vars)
            print(keys)
            concat = {}
            for key in keys:
                concat[key] = np.concatenate([var[key].value for var in vars], axis=0)
            vars = concat
            # Concatenate multiple weights
            weights = np.concatenate(weights, axis=0)

            for k in input_variables:
                logger.info(k)
                # unflatten all the jet variables
                collection = k.split("_")[0]

                # check if collection_N is present to unflatten the variables
                if f"{collection}_N" in vars.keys() and k.split("_")[1] != "N":
                    #number_per_event = tuple(vars[f"{collection}_N"].value)
                    number_per_event = tuple(vars[f"{collection}_N"])
                    if ak.all(number_per_event == number_per_event[0]):
                        variables_dict[k] = ak.to_numpy(
                            ak.unflatten(vars[k], number_per_event)
                        )
                    else:
                        logger.warning(
                            f"number of {collection} jets per event is not the same for all events"
                        )
                        variables_dict[k] = ak.to_numpy(
                            ak.pad_none(
                                ak.unflatten(vars[k], number_per_event),
                                6,
                                clip=True,
                            )
                        )
                else:
                    variables_dict[k] = ak.to_numpy(ak.unflatten(vars[k], 1))

            weights = np.expand_dims(weights, axis=0)
            variables_array =np.concatenate(
                    [variables_dict[input] for input in input_variables], axis=1
                )
            print(variables_array)
            variables_array = np.swapaxes(variables_array, 0 , 1)
            logger.info(f"variables_array {variables_array.shape}")
            logger.info(f"weights {weights.shape}")
            variables_array = np.append(variables_array, weights, axis=0)
            logger.info(f"variables_array complete {variables_array.shape}")
            print(variables_array)
            print(variables_dict)

        tot_lenght += variables_array.shape[1]

        # concatenate all the variables into a single torch tensor
        if 'variables' not in locals():
            print(f"overwrite variables")
            variables = torch.tensor(variables_array, dtype=torch.float32)[
                :, : math.ceil(total_fraction_of_events * variables_array.shape[1])
            ]
        else:
            variables = torch.cat(
                (
                    variables,
                    torch.tensor(variables_array, dtype=torch.float32),
                ),
                dim=1,
            )[:, : math.ceil(total_fraction_of_events * variables_array.shape[1])]
        print(f"variable length {len(variables[0])}")

    logger.info(f"number of {sig_bkg} events: {variables.shape[1]}")

    flag_tensor = (
        torch.ones_like(variables[0], dtype=torch.float32).unsqueeze(0)
        if sig_bkg == "signal"
        else torch.zeros_like(variables[0], dtype=torch.float32).unsqueeze(0)
    )

    X = (variables, flag_tensor)
    return X, tot_lenght


def load_data(cfg, seed):
    batch_size = cfg.batch_size
    logger.debug(f"Batch size: {batch_size}")
    
    dirs = cfg.data_dirs

    total_fraction_of_events = cfg.train_fraction + cfg.val_fraction + cfg.test_fraction

    assert total_fraction_of_events <= 1.0, "Fractions must sum to less than 1.0"

    logger.debug("Variables: %s", cfg.input_variables)

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
        sig_files = [direct + file for direct in dirs for file in os.listdir(direct) if file.endswith(".coffea")]
        bkg_files = [direct + file for direct in dirs for file in os.listdir(direct) if file.endswith(".coffea")]
    else:
        logger.error(f"Data format {cfg.data_format} not supported")
        raise ValueError
    
    # Set signal region
    sig_region = cfg.signal_region if "signal_region" in cfg else cfg.region
    # Set background region
    bck_region = cfg.background_region if "background_region" in cfg else cfg.region

    X_sig, tot_lenght_sig = get_variables(
        sig_files,
        total_fraction_of_events,
        cfg.input_variables,
        cfg.signal_list,
        cfg.dataset_signal,
        sig_region,
        "signal",
        cfg.data_format,
    )
    X_bkg, tot_lenght_bkg = get_variables(
        bkg_files,
        total_fraction_of_events,
        cfg.input_variables,
        cfg.background_list,
        cfg.dataset_background,
        bck_region,
        "background",
        cfg.data_format,
    )

    # sum of weights
    sumw_sig = X_sig[0][-1].sum()
    sumw_bkg = X_bkg[0][-1].sum()
    logger.info(f"sum of weights before rescaling signal: {sumw_sig}")
    logger.info(f"sum of weights before rescaling backgound: {sumw_bkg}")

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

    rescaled_sig_weights = X_sig[0][-1] * sig_class_weights
    rescaled_bkg_weights = X_bkg[0][-1] * bkg_class_weights

    logger.info(f"sig_class_weights: {sig_class_weights}")
    logger.info(f"bkg_class_weights: {bkg_class_weights}")

    # sum of weights
    sumw_sig = rescaled_sig_weights.sum()
    sumw_bkg = rescaled_bkg_weights.sum()
    logger.info(f"sum of weights after rescaling signal: {sumw_sig}")
    logger.info(f"sum of weights after rescaling backgound: {sumw_bkg}")

    sig_class_weights_tensor = (
        torch.ones_like(X_sig[0][-1], dtype=torch.float32) * sig_class_weights
    ).unsqueeze(0)
    bkg_class_weights_tensor = (
        torch.ones_like(X_bkg[0][-1], dtype=torch.float32) * bkg_class_weights
    ).unsqueeze(0)

    X_fts = torch.cat((X_sig[0], X_bkg[0]), dim=1).transpose(1, 0)
    X_lbl = torch.cat((X_sig[1], X_bkg[1]), dim=1).transpose(1, 0)
    X_clsw = torch.cat(
        (sig_class_weights_tensor, bkg_class_weights_tensor), dim=1
    ).transpose(1, 0)

    logger.info(f"X_fts shape: {X_fts.shape}")
    logger.info(f"X_lbl shape: {X_lbl.shape}")
    logger.info(f"X_clsw shape: {X_clsw.shape}")

    train_size = math.floor((tot_lenght_sig + tot_lenght_bkg) * cfg.train_fraction)
    val_size = math.floor((tot_lenght_sig + tot_lenght_bkg) * cfg.val_fraction)
    test_size = math.floor((tot_lenght_sig + tot_lenght_bkg) * cfg.test_fraction)

    tot_events = train_size + val_size + test_size

    # keep only total_fraction_of_events
    X_fts = X_fts[:tot_events]
    X_lbl = X_lbl[:tot_events]
    X_clsw = X_clsw[:tot_events]

    X = torch.utils.data.TensorDataset(X_fts, X_lbl, X_clsw)

    logger.info(f"Total size: {len(X)}")
    logger.info(f"Training size: {train_size}")
    logger.info(f"Validation size: {val_size}")
    logger.info(f"Test size: {test_size}")

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        X, [train_size, val_size, test_size], generator=gen
    )

    training_loader = None
    val_loader = None
    test_loader = None

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=cfg.pin_memory,
    )
    logger.info("Training loader size: %d", len(training_loader))

    if not cfg.eval_model:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=cfg.pin_memory,
        )
        logger.info("Validation loader size: %d", len(val_loader))

    if cfg.eval or cfg.eval_model:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=cfg.pin_memory,
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
