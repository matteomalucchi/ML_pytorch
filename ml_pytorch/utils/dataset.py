import uproot
import numpy as np
import torch
import math
import logging
import os
from coffea.util import load
import awkward as ak

logger = logging.getLogger(__name__)


def oversample_dataset(X_dataset):

    X_fts, X_lbl, X_clsw = X_dataset[:][0], X_dataset[:][1], X_dataset[:][2]

    # get the signal and background events
    X_fts_sig = X_fts[X_lbl == 1]
    X_lbl_sig = X_lbl[X_lbl == 1]
    X_clsw_sig = X_clsw[X_lbl == 1]

    X_fts_bkg = X_fts[X_lbl == 0]
    X_lbl_bkg = X_lbl[X_lbl == 0]
    X_clsw_bkg = X_clsw[X_lbl == 0]

    num_events_bkg = int(torch.sum(X_lbl == 0))
    num_events_sig = int(torch.sum(X_lbl == 1))

    oversample_factor = num_events_bkg // num_events_sig + 1

    logger.info(f"Oversample factor: {oversample_factor}")

    X_fts_sig_oversampled = X_fts_sig.repeat((oversample_factor, 1))[:num_events_bkg]
    X_lbl_sig_oversampled = X_lbl_sig.repeat((oversample_factor))[:num_events_bkg]
    X_clsw_sig_oversampled = X_clsw_sig.repeat((oversample_factor, 1))[:num_events_bkg]
    logger.info(f"Number of background events {X_fts_bkg.shape[0]}")
    logger.info(f"Number of signal events before oversampling {X_fts_sig.shape[0]}")
    logger.info(
        f"Number of signal events after oversampling {X_fts_sig_oversampled.shape[0]}"
    )

    X_fts_oversampled = torch.cat((X_fts_sig_oversampled, X_fts_bkg), dim=0)
    X_lbl_oversampled = torch.cat((X_lbl_sig_oversampled, X_lbl_bkg), dim=0)
    X_clsw_oversampled = torch.cat((X_clsw_sig_oversampled, X_clsw_bkg), dim=0)

    # reshuffle the data
    idx = np.random.permutation(X_fts_oversampled.shape[0])
    X_fts_oversampled = X_fts_oversampled[idx]
    X_lbl_oversampled = X_lbl_oversampled[idx]
    X_clsw_oversampled = X_clsw_oversampled[idx]

    oversampled_dataset = torch.utils.data.TensorDataset(
        X_fts_oversampled, X_lbl_oversampled, X_clsw_oversampled
    )
    # breakpoint()
    return oversampled_dataset


def get_variables(
    files,
    total_fraction_of_events,
    input_variables,
    sample_list,
    dataset_list,
    region_list,
    sig_bkg,
    data_format="root",
):
    tot_lenght = 0
    if data_format == "root":
        for i, file_name in enumerate(files):
            logger.info(f"Loading file {file_name}")
            # open each file and get the Events tree using uproot
            file = uproot.open(f"{file_name}:Events")
            variables_array = np.array(
                [file[input].array(library="np") for input in input_variables]
            )
    elif data_format == "coffea":
        vars_array = []
        weights = []
        variables_dict = {}
        for i, file_name in enumerate(files):
            logger.info(f"Loading file {file_name}")
            file = load(file_name)
            logger.debug(f"sample_list: {sample_list}")
            if any([s not in list(file["columns"].keys()) for s in sample_list]):
                logger.error(
                    f"sample_list {sample_list} not in available samples {list(file['columns'].keys())}"
                )
                raise ValueError
            for sample in list(file["columns"].keys()):
                logger.info("sample %s", sample)
                logger.debug(list(file["columns"].keys()))
                if sample in sample_list:
                    logger.debug(f"sample {sample} in file")
                    if any(
                        [
                            d not in list(file["columns"][sample].keys())
                            for d in dataset_list
                        ]
                    ):
                        logger.warning(
                            f"dataset_list {dataset_list} not in available datasets {list(file['columns'][sample].keys())}"
                        )
                    for dataset in list(file["columns"][sample].keys()):
                        logger.debug(
                            f"searching dataset {dataset} in dataset_list {dataset_list}"
                        )
                        if dataset in dataset_list:
                            logger.info("dataset %s", dataset)
                            if any(
                                [
                                    region_file
                                    not in list(file["columns"][sample][dataset].keys())
                                    for region_file in region_list
                                ]
                            ):
                                logger.warning(
                                    f"region_list {region_list} not in available regions {list(file['columns'][sample][dataset].keys())}"
                                )
                            for region_file in list(
                                file["columns"][sample][dataset].keys()
                            ):
                                if region_file in region_list:
                                    logger.info("region_file %s", region_file)
                                    vars_array.append(
                                        file["columns"][sample][dataset][region_file]
                                    )
                                    weights.append(
                                        file["columns"][sample][dataset][region_file][
                                            "weight"
                                        ].value
                                        / (
                                            file["sum_genweights"][dataset]
                                            if dataset in file["sum_genweights"]
                                            else 1
                                        )
                                    )
                                    if dataset in file["sum_genweights"]:
                                        logger.info(
                                            f"original weight: {file['columns'][sample][dataset][region_file]['weight'].value[0]}"
                                        )
                                        logger.info(
                                            f"sum_genweights: {file['sum_genweights'][dataset]}"
                                        )
                                    logger.info(f"weight: {weights[0]}")
        if len(vars_array) < 1:
            logger.error(
                f"Could not find any datasets in the files {files} with the sample_list {sample_list} and dataset_list {dataset_list} and region {region_list}"
            )
            raise ValueError

        logger.info(f"Found datasets: {len(vars_array)}")
        # Merge multiple lists:
        keys = set().union(*vars_array)
        logger.info(keys)
        concat = {}
        for key in keys:
            concat[key] = np.concatenate([var[key].value for var in vars_array], axis=0)
        vars_array = concat
        # Concatenate multiple weights
        weights = np.concatenate(weights, axis=0)

        for k in input_variables:
            logger.info(k)
            # unflatten all the jet variables
            collection = k.split("_")[0]

            # check if collection_N is present to unflatten the variables
            if ":" in k:
                variable_name, pos = k.split(":")
                number_per_event = tuple(vars_array[f"{collection}_N"])
                if not ak.all(number_per_event == number_per_event[0]):
                    raise ValueError(
                        f"number of {collection} per event is not the same for all events"
                    )
                variables_dict[k] = ak.to_numpy(
                    ak.unflatten(
                        vars_array[variable_name][
                            np.arange(
                                int(pos),
                                len(vars_array[variable_name]),
                                number_per_event[0],
                            )
                        ],
                        1,
                    )
                )

            elif f"{collection}_N" in vars_array.keys() and k.split("_")[1] != "N":
                number_per_event = tuple(vars_array[f"{collection}_N"])
                if ak.all(number_per_event == number_per_event[0]):
                    variables_dict[k] = ak.to_numpy(
                        ak.unflatten(vars_array[k], number_per_event)
                    )
                else:
                    logger.warning(
                        f"number of {collection} per event is not the same for all events, \n padding collection to 5 ..."
                    )
                    variables_dict[k] = ak.to_numpy(
                        ak.pad_none(
                            ak.unflatten(vars_array[k], number_per_event),
                            5,
                            clip=True,
                        )
                    )
            else:
                variables_dict[k] = ak.to_numpy(ak.unflatten(vars_array[k], 1))

        weights = np.expand_dims(weights, axis=0)

        variables_array = np.concatenate(
            [variables_dict[input] for input in input_variables], axis=1
        )
        variables_array = np.swapaxes(variables_array, 0, 1)

        logger.info(f"variables_array {variables_array.shape}")
        logger.info(f"weights {weights.shape}")
        variables_array = np.append(variables_array, weights, axis=0)
        logger.info(f"variables_array complete {variables_array.shape}")

    tot_lenght += variables_array.shape[1]

    # concatenate all the variables into a single torch tensor
    if "variables" not in locals():
        logger.debug(f"overwrite variables")
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
    logger.info(f"variable length {len(variables[0])}")

    logger.info(f"number of {sig_bkg} events: {variables.shape[1]}")

    flag_tensor = (
        torch.ones_like(variables[0], dtype=torch.float32).unsqueeze(0)
        if sig_bkg == "signal"
        else torch.zeros_like(variables[0], dtype=torch.float32).unsqueeze(0)
    )
    
    #shuffle the variables
    idx = np.random.permutation(tot_lenght)
    print(idx)
    variables=variables[:,idx]

    X = (variables, flag_tensor)
    return X, tot_lenght


def load_data(cfg, seed):
    batch_size = cfg.batch_size
    logger.debug(f"Batch size: {batch_size}")
    
    #initialize numpy seed
    np.random.seed(int(seed))

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
                for signal in cfg.signal_sample:
                    if signal in file and "SR" in file:
                        sig_files.append(x + file)
                for background in cfg.background_sample:
                    if background in file and "SR" in file:
                        bkg_files.append(x + file)
    elif cfg.data_format == "coffea":
        sig_files = [
            direct + file
            for direct in dirs
            for file in os.listdir(direct)
            if file.endswith(".coffea")
        ]
        bkg_files = [
            direct + file
            for direct in dirs
            for file in os.listdir(direct)
            if file.endswith(".coffea")
        ]
    else:
        logger.error(f"Data format {cfg.data_format} not supported")
        raise ValueError

    X_sig, tot_lenght_sig = get_variables(
        sig_files,
        total_fraction_of_events,
        cfg.input_variables,
        cfg.signal_sample,
        cfg.signal_dataset,
        cfg.signal_region,
        "signal",
        cfg.data_format,
    )
    X_bkg, tot_lenght_bkg = get_variables(
        bkg_files,
        total_fraction_of_events,
        cfg.input_variables,
        cfg.background_sample,
        cfg.background_dataset,
        cfg.background_region,
        "background",
        cfg.data_format,
    )
        

    # compute class weights such that sumw is the same for signal and background and each weight is order of 1

    logger.info(f"Number of background events  {X_bkg[0].shape[1]}")
    logger.info(f"Number of signal events {X_sig[0].shape[1]}")
    
    if cfg.oversample and cfg.undersample:
        raise ValueError("Select only oversample or undersample")
    
    if cfg.undersample:
        logger.info("Performing undersampling of background")
        logger.info(f"Number of background events before undersampling {X_bkg[0].shape[1]}")
        num_events_sig = X_sig[0].shape[1]
        X_bkg_f = X_bkg[0][
            :, :num_events_sig
        ]
        X_bkg_l = X_bkg[1][
            :, :num_events_sig
        ]
        X_bkg = (X_bkg_f, X_bkg_l)
        logger.info(f"Number of background events after undersampling {X_bkg[0].shape[1]}")
        
    

    # if cfg.oversample:
    #     logger.info("Performing oversampling")
    #     num_events_sig = X_sig[0].shape[1]
    #     X_sig_f = X_sig[0].repeat((1, num_events_bkg // num_events_sig + 1))[
    #         :, :num_events_bkg
    #     ]
    #     X_sig_l = X_sig[1].repeat((1, num_events_bkg // num_events_sig + 1))[
    #         :, :num_events_bkg
    #     ]
    #     X_sig = (X_sig_f, X_sig_l)
    #     logger.info(f"Number of signal events after oversampling {X_sig[0].shape[1]}")

    num_events_bkg = X_bkg[0].shape[1]
    num_events_sig = X_sig[0].shape[1]

    # sum of weights
    sumw_sig = X_sig[0][-1].sum()
    sumw_bkg = X_bkg[0][-1].sum()
    logger.info(f"sum of weights before rescaling signal: {sumw_sig}")
    logger.info(f"sum of weights before rescaling backgound: {sumw_bkg}")

    if not cfg.oversample:
        if True:
            sig_class_weights = (num_events_sig + num_events_bkg) / (2 * sumw_sig)
            bkg_class_weights = (num_events_sig + num_events_bkg) / (2 * sumw_bkg)
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
    else:
        sig_class_weights = 1.0
        bkg_class_weights = 1.0

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
    X_lbl = torch.cat((X_sig[1], X_bkg[1]), dim=1).transpose(1, 0).flatten()
    X_clsw = torch.cat(
        (sig_class_weights_tensor, bkg_class_weights_tensor), dim=1
    ).transpose(1, 0)

    logger.info(f"X_fts shape: {X_fts.shape}")
    logger.info(f"X_lbl shape: {X_lbl.shape}")
    logger.info(f"X_clsw shape: {X_clsw.shape}")

    tot_num_events = num_events_sig + num_events_bkg
    if True:
        # shuffle the tensor with numpy random
        idx = np.random.permutation(tot_num_events)
        X_fts = X_fts[idx]
        X_lbl = X_lbl[idx]
        X_clsw = X_clsw[idx]

    train_size = math.floor(tot_num_events * cfg.train_fraction)
    val_size = math.floor(tot_num_events * cfg.val_fraction)
    test_size = math.floor(tot_num_events * cfg.test_fraction)

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

    #shuffle and split
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        X, [train_size, val_size, test_size], generator=gen
    )

    if cfg.oversample:
        #perform the oversampling of the signal separately for training, validation and testing datasets
        logger.info("Performing oversampling")
        train_dataset=oversample_dataset(train_dataset)
        val_dataset=oversample_dataset(val_dataset)
        test_dataset=oversample_dataset(test_dataset)

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
