import sys
import uproot
import numpy as np
import torch
import math
import logging
import os
from coffea.util import load
import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import hashlib
import json
import queue
import re
import threading
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

from ml_pytorch.defaults.preprocess_variables_functions import functions_dict


def parse_classes(cfg):
    """Normalize the two supported config layouts into a list of class
    definitions.

    Supported formats:
      * New multiclass layout: ``cfg.classes`` is a mapping
        ``{class_name: {sample, region, dataset, lbl}}``.
      * Legacy binary layout: ``cfg.signal_*`` / ``cfg.background_*``.

    Returns a list of dicts ordered by ``class_idx`` (0..C-1). Each dict has:
      ``name``, ``samples``, ``datasets``, ``regions``, ``lbl`` (user label),
      and ``class_idx`` (contiguous internal index used for training targets).
    """
    classes = []
    if "classes" in cfg and cfg.classes is not None:
        items = list(cfg.classes.items())
        # sort by user-provided ``lbl`` so that the internal class_idx is
        # deterministic regardless of dict iteration order
        items.sort(key=lambda kv: kv[1].lbl)
        for idx, (name, c) in enumerate(items):
            classes.append(
                {
                    "name": name,
                    "samples": list(c.sample),
                    "datasets": list(c.dataset),
                    "regions": list(c.region),
                    "lbl": int(c.lbl),
                    "class_idx": idx,
                }
            )
    else:
        # Legacy binary layout. Background first so it gets idx=0 and signal
        # idx=1, matching the previous behaviour (sig label == 1, bkg == 0).
        classes.append(
            {
                "name": "background",
                "samples": list(cfg.background_sample),
                "datasets": list(cfg.background_dataset),
                "regions": list(cfg.background_region),
                "lbl": 0,
                "class_idx": 0,
            }
        )
        classes.append(
            {
                "name": "signal",
                "samples": list(cfg.signal_sample),
                "datasets": list(cfg.signal_dataset),
                "regions": list(cfg.signal_region),
                "lbl": 1,
                "class_idx": 1,
            }
        )
    return classes


def oversample_dataset(X_dataset, is_binary):

    X_fts, X_lbl, X_clsw, X_k = X_dataset[:][0], X_dataset[:][1], X_dataset[:][2], X_dataset[:][3]

    if is_binary:
        X_fts_sig = X_fts[X_lbl == 1]
        X_lbl_sig = X_lbl[X_lbl == 1]
        X_clsw_sig = X_clsw[X_lbl == 1]
        X_k_sig = X_k[X_lbl == 1]

        X_fts_bkg = X_fts[X_lbl == 0]
        X_lbl_bkg = X_lbl[X_lbl == 0]
        X_clsw_bkg = X_clsw[X_lbl == 0]
        X_k_bkg = X_k[X_lbl == 0]

        num_events_bkg = int(torch.sum(X_lbl == 0))
        num_events_sig = int(torch.sum(X_lbl == 1))

        if num_events_sig > num_events_bkg:
            raise ValueError(
                "Number of signal events is greater than number of background events."
            )

        oversample_factor = num_events_bkg // num_events_sig + 1
        logger.info(f"Oversample factor: {oversample_factor}")

        X_fts_sig_oversampled  = X_fts_sig.repeat((oversample_factor, 1))[:num_events_bkg]
        X_lbl_sig_oversampled  = X_lbl_sig.repeat(oversample_factor)[:num_events_bkg]
        X_clsw_sig_oversampled = X_clsw_sig.repeat((oversample_factor, 1))[:num_events_bkg]
        X_k_sig_oversampled    = X_k_sig.repeat((oversample_factor, 1))[:num_events_bkg]

        logger.info(f"Number of background events: {num_events_bkg}")
        logger.info(f"Number of signal events before oversampling: {num_events_sig}")
        logger.info(f"Number of signal events after oversampling: {X_fts_sig_oversampled.shape[0]}")

        X_fts_out  = torch.cat((X_fts_sig_oversampled,  X_fts_bkg),  dim=0)
        X_lbl_out  = torch.cat((X_lbl_sig_oversampled,  X_lbl_bkg),  dim=0)
        X_clsw_out = torch.cat((X_clsw_sig_oversampled, X_clsw_bkg), dim=0)
        X_k_out    = torch.cat((X_k_sig_oversampled,    X_k_bkg),    dim=0)

    else:
        # Multiclass: oversample all classes to the size of the largest class
        classes = torch.unique(X_lbl)
        n_per_class = {int(c): int((X_lbl == c).sum()) for c in classes}
        n_max = max(n_per_class.values())
        logger.info(f"Multiclass oversampling — target size per class: {n_max}")

        fts_list, lbl_list, clsw_list, k_list = [], [], [], []
        for c in classes:
            c = int(c)
            mask = X_lbl == c
            n = n_per_class[c]

            X_fts_c  = X_fts[mask]
            X_lbl_c  = X_lbl[mask]
            X_clsw_c = X_clsw[mask]
            X_k_c    = X_k[mask]

            if n < n_max:
                oversample_factor = n_max // n + 1
                logger.info(f"Class {c}: oversampling {n} -> {n_max} (factor {oversample_factor})")
                X_fts_c  = X_fts_c.repeat((oversample_factor, 1))[:n_max]
                X_lbl_c  = X_lbl_c.repeat(oversample_factor)[:n_max]
                X_clsw_c = X_clsw_c.repeat((oversample_factor, 1))[:n_max]
                X_k_c    = X_k_c.repeat((oversample_factor, 1))[:n_max]
            else:
                logger.info(f"Class {c}: no oversampling needed ({n} events)")

            fts_list.append(X_fts_c)
            lbl_list.append(X_lbl_c)
            clsw_list.append(X_clsw_c)
            k_list.append(X_k_c)

        X_fts_out  = torch.cat(fts_list,  dim=0)
        X_lbl_out  = torch.cat(lbl_list,  dim=0)
        X_clsw_out = torch.cat(clsw_list, dim=0)
        X_k_out    = torch.cat(k_list,    dim=0)

    # Reshuffle
    idx = np.random.permutation(X_fts_out.shape[0])
    X_fts_out  = X_fts_out[idx]
    X_lbl_out  = X_lbl_out[idx]
    X_clsw_out = X_clsw_out[idx]
    X_k_out    = X_k_out[idx]

    return torch.utils.data.TensorDataset(X_fts_out, X_lbl_out, X_clsw_out, X_k_out)


def get_sumw_per_dataset(weights_list, dataset_names):
    """
    Compute the sum of weights of each dataset. Entries of weights_list
    belonging to the same dataset (e.g. multiple regions) are summed together.
    """
    sumw_per_dataset = {}
    for w, name in zip(weights_list, dataset_names):
        sumw_per_dataset[name] = sumw_per_dataset.get(name, 0.0) + np.sum(w)
    return sumw_per_dataset


def log_sumw_per_dataset(weights_list, dataset_names, step):
    """Log the sum of weights of each dataset at a given step."""
    for name, sumw in get_sumw_per_dataset(weights_list, dataset_names).items():
        logger.info(f"sum of weights of dataset {name} {step}: {sumw}")


def normalize_weights_per_dataset(weights_list, dataset_names):
    """
    Normalize the weights of each dataset so that sum(w) = 1 for each of them.
    Entries of weights_list belonging to the same dataset (e.g. multiple regions)
    are normalized together.
    """
    sumw_per_dataset = get_sumw_per_dataset(weights_list, dataset_names)

    normalized_weights_list = []
    for w, name in zip(weights_list, dataset_names):
        sumw = sumw_per_dataset[name]
        if sumw <= 0:
            raise ValueError(
                f"Sum of weights of dataset {name} is {sumw}, cannot normalize it to 1"
            )
        normalized_weights_list.append(w / sumw)

    return normalized_weights_list


def list_parquet_files(path):
    """Sorted list of the files in a parquet directory (searched recursively).

    As in pyarrow.dataset, the files and directories starting with "." or "_"
    (e.g. _SUCCESS) are ignored.
    """
    if os.path.isfile(path):
        return [path]
    files = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        for name in sorted(filenames):
            if not name.startswith((".", "_")):
                files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return files


def count_field(schema, n_variable):
    """Name of the jagged field (<collection>_<field>) used to create the
    number of objects per event of a collection (<collection>_N)."""
    collection = n_variable[: -len("_N")]
    fields = [
        f.name
        for f in schema
        if f.name.startswith(f"{collection}_")
        and f.name != n_variable
        and (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))
    ]
    if not fields:
        raise ValueError(
            f"Input variable {n_variable} is not in the parquet files and no "
            f"jagged field of the collection {collection} is available to create it"
        )
    return fields[0]


def count_objects(array, field, n_variable, pad_value):
    """Create the number of objects per event of a collection (<collection>_N)
    by counting the elements of one of its jagged fields (``field``, see
    count_field).

    The padded objects (added to have the same number per event), which are
    either None or equal to pad_value, are not counted, so only the real
    objects are.
    """
    logger.warning(
        f"Input variable {n_variable} not found in the parquet files: creating it "
        f"by counting the elements of {field} that are not None or {pad_value}"
    )
    # the comparison is None for the None elements, which are then not counted
    is_real = ak.fill_none(array[field] != pad_value, False)
    return ak.fill_none(ak.sum(is_real, axis=1), 0)


def get_fold_id(columns, event_column, k_folds, num_events):
    """k-fold index ``event % k_folds`` of each event, computed from the
    ``event_column`` column of ``columns``, or -1 for all the events if
    ``event_column`` is None. The modulo is computed on the int64 event
    numbers, which are not exactly representable as float32."""
    if event_column is None:
        return np.full(num_events, -1, dtype=np.int64)
    available = columns.fields if isinstance(columns, ak.Array) else list(columns.keys())
    if event_column not in available:
        raise ValueError(
            f"Column {event_column} (kfold_event_column) needed for kfold_split "
            f"event not found in the input files. Available columns: {available}"
        )
    events = columns[event_column]
    events = np.asarray(ak.to_numpy(events) if isinstance(events, ak.Array) else events)
    if len(events) != num_events:
        raise ValueError(
            f"Column {event_column} has {len(events)} entries but there are "
            f"{num_events} events"
        )
    return events.astype(np.int64) % k_folds


def get_variables(
    files,
    parquet_files,
    total_fraction_of_events,
    input_variables,
    sample_list,
    dataset_list,
    region_list,
    class_label,
    data_format,
    preprocess_variables_functions,
    novars=False,
    pad_value=-999.0,
    normalize_per_dataset=False,
    event_column=None,
    k_folds=0,
):
    """Load the input features and weights for a single class.

    ``class_label`` is the integer used as the per-event target (typically the
    internal class index 0..C-1).

    If ``event_column`` is given, the k-fold index ``event % k_folds`` of each
    event is computed from that column, otherwise it is set to -1.
    """
    if data_format == "root":
        raise ValueError("Not updated for root format!")
        for i, file_name in enumerate(files):
            logger.info(f"Loading file {file_name}")
            # open each file and get the Events tree using uproot
            file = uproot.open(f"{file_name}:Events")
            variables_array = np.array(
                [file[input].array(library="np") for input in input_variables]
            )
    elif data_format == "parquet":
        # function to load input files from chunks of .parquet the structure here presupposes
        # that each directory contains a single region of a single dataset
        variables_array_list = []
        dataset_names = []
        # here we loop over the directories
        for i, file_name in enumerate(parquet_files):
            logger.info(f"Loading directory {file_name}")

            # here i select the corresponding dataset to the parquet directory I'm looking at
            matching_dataset = [ds for ds in dataset_list if ds in file_name]
            if len(matching_dataset) != 1:
                logger.warning(
                    f"Could not find a unique matching dataset for {file_name} from the list {dataset_list}"
                )
            matching_dataset = matching_dataset[0]
            logger.info(f"Matching dataset {matching_dataset}")

            if "kl" in matching_dataset:
                kl_val = extract_param_value(matching_dataset, "kl")
            elif "C2V" in matching_dataset:
                kl_val = extract_param_value(matching_dataset, "C2V")
            else:
                kl_val = 9999.0

            logger.info(f"kl value found in dataset {matching_dataset} is {kl_val}")

            # index of the dataset in the class dataset list, to split the
            # evaluation per dataset (the kl value alone is not unique)
            dataset_idx = dataset_list.index(matching_dataset)

            # here i select the corresponding .coffea file as well
            matching_coffea = [cf for cf in files if matching_dataset in cf]
            if len(matching_coffea) != 1:
                logger.warning(
                    f"Could not find a unique matching coffea file for {file_name} from the list {files}"
                )
            matching_coffea = matching_coffea[0]

            # load the flat varibles
            # the files are passed explicitly in a fixed order, so that the
            # events are always loaded in the same order
            files_in_dir = list_parquet_files(file_name)
            logger.info(f"Reading {len(files_in_dir)} parquet files from {file_name}")
            dataset = ds.dataset(files_in_dir, format="parquet")
            logger.debug(f"input_variables {input_variables}")

            # load the corresponding .coffea file to get the sum of genweights
            logger.info(f"Loading metacondition from {matching_coffea}")
            file = load(matching_coffea)

            vars = [x for x in input_variables if ":" not in x]
            # number of objects of a collection (<collection>_N) requested as
            # input but not saved in the parquet files: created below
            missing_counts = [
                x
                for x in vars
                if x.endswith("_N") and x not in dataset.schema.names
            ]
            # jagged variables, added flattened to the variables array below
            jagged_variables = []
            max_pos = 0

            for k in input_variables:
                if ":" in k:
                    variable_name, pos = k.split(":")
                    if variable_name not in jagged_variables:
                        jagged_variables.append(variable_name)
                    if int(pos) > max_pos:
                        max_pos = int(pos)
            logger.debug(f"jagged_variables {jagged_variables}")

            count_fields = {k: count_field(dataset.schema, k) for k in missing_counts}

            if event_column is not None and event_column not in dataset.schema.names:
                raise ValueError(
                    f"Column {event_column} (kfold_event_column) needed for "
                    f"kfold_split event not found in the parquet files of {file_name}"
                )
            # read the parquet files only once and only the needed columns
            # (flat and jagged inputs, weight, fields to count the objects and
            # event number for the k-fold split)
            columns = list(
                dict.fromkeys(
                    [x for x in vars if x not in missing_counts]
                    + jagged_variables
                    + list(count_fields.values())
                    + ["weight"]
                    + ([event_column] if event_column is not None else [])
                )
            )
            table = dataset.to_table(columns=columns, use_threads=True)
            jagged_variables_array = ak.from_arrow(table)
            variables_array = jagged_variables_array[
                [x for x in vars if x not in missing_counts]
            ]

            for k in missing_counts:
                variables_array[k] = count_objects(
                    jagged_variables_array, count_fields[k], k, pad_value
                )

            # add the wanted number of flattened features from the jagged variables to the feature array
            # events with fewer objects than requested are padded with pad_value
            # (slicing instead of ak.pad_none, which does not support the 32-bit
            # list offsets of arrow in recent awkward versions)
            for k in jagged_variables:
                for pos in range(max_pos + 1):
                    # the range slice (not [:, pos]) gives an empty list for the
                    # events with fewer objects, so firsts returns None, which
                    # is then filled with pad_value (as for None-padded objects)
                    variables_array[k + ":" + str(pos)] = ak.fill_none(
                        ak.firsts(jagged_variables_array[k][:, pos : pos + 1]),
                        pad_value,
                    )

            # apply any preprocessing function to the variables
            for k in input_variables:
                if k in preprocess_variables_functions:
                    logger.info(
                        f"Applying preprocessing function {preprocess_variables_functions[k]} to variable {k}"
                    )
                    logger.info(f"vars_array[k] before {variables_array[k]}")
                    variables_array[k] = functions_dict[
                        preprocess_variables_functions[k][0]
                    ](variables_array[k], *preprocess_variables_functions[k][1])
                    logger.info(f"vars_array[k] after {variables_array[k]}")

            # add the weights normalized to mean 1
            variables_array["weights"] = jagged_variables_array["weight"] / (
                file["sum_genweights"][matching_dataset]
                if matching_dataset in file["sum_genweights"]
                else 1
            )

            # add the kl value
            variables_array["kl"] = np.full(len(variables_array), kl_val)
            variables_array["dataset_idx"] = np.full(len(variables_array), dataset_idx)
            variables_array["fold_id"] = get_fold_id(
                jagged_variables_array, event_column, k_folds, len(variables_array)
            )

            # concatenate in a single numpy matrix of shape (num_variables, num_events)
            # the None values of the inputs are padded with pad_value
            variables_array = np.array(
                [
                    ak.to_numpy(ak.fill_none(variables_array[f], pad_value))
                        for f in input_variables
                ]
                + [
                    ak.to_numpy(variables_array[f])
                    for f in ["weights", "kl", "dataset_idx", "fold_id"]
                ]
            )

            logger.info(f"variables_array complete shape {variables_array.shape}")
            variables_array_list.append(variables_array)
            dataset_names.append(matching_dataset)

        if len(variables_array_list) == 0:
            raise ValueError("No parquet data loaded")

        if normalize_per_dataset:
            # the weights are the fourth to last row (before kl, dataset_idx
            # and fold_id)
            log_sumw_per_dataset(
                [v[-4] for v in variables_array_list],
                dataset_names,
                "before normalization of each dataset to sum 1",
            )
            normalized_weights = normalize_weights_per_dataset(
                [v[-4] for v in variables_array_list], dataset_names
            )
            for v, w in zip(variables_array_list, normalized_weights):
                v[-4] = w
            log_sumw_per_dataset(
                [v[-4] for v in variables_array_list],
                dataset_names,
                "after normalization of each dataset to sum 1",
            )

        # concatenate along event axis
        variables_array = np.concatenate(variables_array_list, axis=1)

        # normalise the per-class event weights to mean 1, mirroring the
        # coffea path (weights = weights / np.mean(weights)). Done after the
        # concat so the relative xsec / sum_genweights scaling between a
        # class's datasets (e.g. the ttbar sub-samples) is preserved, while
        # the overall class scale matches the other classes going into the
        # training loss. Without this, classes with tiny physical weights
        # (ttbar, ~weight/sum_genweights) contribute almost nothing to the
        # loss and are never learned.
        # weights is the fourth-to-last row: input_variables + ["weights", "kl", "dataset_idx", "fold_id"]
        variables_array[-4] = variables_array[-4] / np.mean(variables_array[-4])

        logger.info(f"variables_array complete shape {variables_array.shape}")

    elif data_format == "coffea":
        vars_array = []
        weights = []
        dataset_names = []
        kl_values = []
        dataset_idx_values = []
        variables_dict = {}
        for i, file_name in enumerate(files):
            logger.info(f"Loading file {file_name}")
            file = load(file_name)
            logger.debug(f"sample_list: {sample_list}")
            if all([s not in list(file["columns"].keys()) for s in sample_list]):
                logger.warning(
                    f"sample_list {sample_list} not in available samples {list(file['columns'].keys())}"
                )

            for sample in list(file["columns"].keys()):
                logger.info("sample %s", sample)
                logger.debug(list(file["columns"].keys()))
                if sample in sample_list:
                    logger.debug(f"sample {sample} in file")
                    if all(
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
                            if all(
                                [
                                    region_file
                                    not in list(file["columns"][sample][dataset].keys())
                                    for region_file in region_list
                                ]
                            ):
                                logger.warning(
                                    f"region_list {region_list} not in available regions {list(file['columns'][sample][dataset].keys())}"
                                )

                            if "kl" in dataset:
                                kl_val = extract_param_value(dataset, "kl")
                            elif "C2V" in dataset:
                                kl_val = extract_param_value(dataset, "C2V")
                            else:
                                kl_val = 9999.0

                            logger.info(
                                f"kl value found in dataset {dataset} is {kl_val}"
                            )

                            for region_file in list(
                                file["columns"][sample][dataset].keys()
                            ):
                                if region_file in region_list:
                                    logger.info("region_file %s", region_file)
                                    logger.info(
                                        f"FOUND DATA : {file_name} {sample} {dataset} {region_file}"
                                    )
                                    if novars:
                                        vars_array.append(
                                            file["columns"][sample][dataset][
                                                region_file
                                            ]
                                        )
                                        weights.append(
                                            file["columns"][sample][dataset][
                                                region_file
                                            ]["weight"].value
                                            / (
                                                file["sum_genweights"][dataset]
                                                if dataset in file["sum_genweights"]
                                                else 1
                                            )
                                        )
                                        kl_values.append(
                                            np.full(
                                                len(
                                                    file["columns"][sample][dataset][
                                                        region_file
                                                    ]["weight"].value
                                                ),
                                                kl_val,
                                            )
                                        )

                                        if dataset in file["sum_genweights"]:
                                            logger.info(
                                                f"original weight: {file['columns'][sample][dataset][region_file]['weight'].value[0]}"
                                            )
                                            logger.info(
                                                f"sum_genweights: {file['sum_genweights'][dataset]}"
                                            )
                                    else:
                                        vars_array.append(
                                            file["columns"][sample][dataset][
                                                region_file
                                            ]["nominal"]
                                        )
                                        weights.append(
                                            file["columns"][sample][dataset][
                                                region_file
                                            ]["nominal"]["weight"].value
                                            / (
                                                file["sum_genweights"][dataset]
                                                if dataset in file["sum_genweights"]
                                                else 1
                                            )
                                        )
                                        kl_values.append(
                                            np.full(
                                                len(
                                                    file["columns"][sample][dataset][
                                                        region_file
                                                    ]["nominal"]["weight"].value
                                                ),
                                                kl_val,
                                            )
                                        )

                                        if dataset in file["sum_genweights"]:
                                            logger.info(
                                                f"original weight: {file['columns'][sample][dataset][region_file]['nominal']['weight'].value[0]}"
                                            )
                                            logger.info(
                                                f"sum_genweights: {file['sum_genweights'][dataset]}"
                                            )
                                    dataset_names.append(dataset)
                                    # index of the dataset in the class dataset list, to
                                    # split the evaluation per dataset
                                    dataset_idx_values.append(
                                        np.full(
                                            len(weights[-1]),
                                            dataset_list.index(dataset),
                                        )
                                    )
                                    logger.info(f"weight: {weights[-1]}")

        if len(vars_array) < 1:
            logger.error(
                f"Could not find any datasets in the files {files} with the sample_list {sample_list} and dataset_list {dataset_list} and region {region_list}"
            )
            raise ValueError

        logger.info(f"Found datasets: {len(vars_array)}")
        try:
            # check that all datasets have been found
            # NOTE: the assert could be set to >= in general
            assert len(vars_array) == len(dataset_list)
        except AssertionError:
            logger.error(
                f"Not all datasets were found in the files {files} with the sample_list {sample_list} and dataset_list {dataset_list} and region {region_list}"
            )
            raise AssertionError

        # Merge multiple lists:
        keys = set().union(*vars_array)
        logger.info(keys)
        concat = {}
        for key in keys:
            concat[key] = np.concatenate([var[key].value for var in vars_array], axis=0)
        vars_array = concat
        if normalize_per_dataset:
            log_sumw_per_dataset(
                weights,
                dataset_names,
                "before normalization of each dataset to sum 1",
            )
            weights = normalize_weights_per_dataset(weights, dataset_names)
            log_sumw_per_dataset(
                weights,
                dataset_names,
                "after normalization of each dataset to sum 1",
            )

        # number of events of each entry, to split the weights per dataset when logging
        split_indices = np.cumsum([len(w) for w in weights])[:-1]

        # Concatenate multiple weights
        weights = np.concatenate(weights, axis=0)

        # Concatenate multiple kl values
        kl_values = np.concatenate(kl_values, axis=0)

        # Concatenate multiple dataset indices
        dataset_idx_values = np.concatenate(dataset_idx_values, axis=0)

        fold_id_values = get_fold_id(
            vars_array, event_column, k_folds, len(weights)
        )

        for k in input_variables:
            logger.info(k)
            # unflatten all the jet variables
            collection = k.split("_")[0]
            if k in preprocess_variables_functions:
                logger.info(
                    f"Applying preprocessing function {preprocess_variables_functions[k]} to variable {k}"
                )
                logger.info(f"vars_array[k] before {vars_array[k]}")
                vars_array[k] = functions_dict[preprocess_variables_functions[k][0]](
                    vars_array[k], *preprocess_variables_functions[k][1]
                )
                logger.info(f"vars_array[k] after {vars_array[k]}")

            # check if collection_N is present to unflatten the variables
            if ":" in k:
                variable_name, pos = k.split(":")
                pos = int(pos)
                number_per_event = np.asarray(vars_array[f"{collection}_N"], dtype=np.int64)
                # events with fewer objects than requested are padded with pad_value
                ragged = ak.fill_none(ak.pad_none(ak.unflatten(vars_array[variable_name], number_per_event), pos + 1, axis=1), pad_value, axis=1)
                sliced = ragged[:, pos]
                variables_dict[k] = ak.to_numpy(sliced).reshape(-1,1)

                # Unflatten the flat array back into ragged per-event arrays

                # Check all events have enough jets
                # min_num = int(ak.min(number_per_event))
                # if min_num <= pos:
                #     raise ValueError(
                #         f"Requested jet index {pos} but some events only have {min_num} jets "
                #         f"for variable {variable_name}"
                #     )
                # if not ak.all(number_per_event == number_per_event[0]):
                #     logger.warning(
                #         f"Not all events have the same number of {collection} "
                #         f"(min={min_num}). Slicing index {pos} from each event."
                #     )
                # variables_dict[k] = ak.to_numpy(ragged[:, pos])

                # # Slice the pos-th jet from each event — works for variable-length events
                # variables_dict[k] = ak.to_numpy(ragged[:, pos])
                # variables_dict[k] = ak.to_numpy(
                #     ak.unflatten(
                #         vars_array[variable_name][
                #             np.arange(
                #                 int(pos),
                #                 len(vars_array[variable_name]),
                #                 min_num,
                #             )
                #         ],
                #         1,
                #     ),
                # )
            elif f"{collection}_N" in vars_array.keys() and k.split("_")[1] != "N":
                # whole collection: one column per object, padded to the
                # maximum number of objects per event, the missing objects
                # are set to pad_value
                number_per_event = np.asarray(vars_array[f"{collection}_N"], dtype=np.int64)
                max_number = int(np.max(number_per_event)) if len(number_per_event) else 0
                if np.any(number_per_event != max_number):
                    logger.warning(
                        f"number of {collection} per event is not the same for all "
                        f"events: padding {k} to the maximum number {max_number} "
                        f"with {pad_value}"
                    )
                variables_dict[k] = ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(
                            ak.unflatten(vars_array[k], number_per_event),
                            max_number,
                            axis=1,
                        ),
                        pad_value,
                        axis=1,
                    )
                ).reshape(len(number_per_event), max_number)
            else:
                variables_dict[k] = ak.to_numpy(ak.unflatten(vars_array[k], 1))

        weights = np.expand_dims(weights, axis=0)
        kl_values = np.expand_dims(kl_values, axis=0)
        dataset_idx_values = np.expand_dims(dataset_idx_values, axis=0)
        fold_id_values = np.expand_dims(fold_id_values, axis=0)

        # normalize the weights to have mean of 1
        log_sumw_per_dataset(
            np.split(weights[0], split_indices),
            dataset_names,
            "before normalization to mean 1",
        )
        weights = weights / np.mean(weights)
        log_sumw_per_dataset(
            np.split(weights[0], split_indices),
            dataset_names,
            "after normalization to mean 1",
        )

        variables_array = np.concatenate(
            [variables_dict[input] for input in input_variables], axis=1
        )
        variables_array = np.swapaxes(variables_array, 0, 1)

        logger.info(f"variables_array {variables_array.shape}")
        logger.info(f"weights {weights.shape}")
        variables_array = np.concatenate(
            [variables_array, weights, kl_values, dataset_idx_values, fold_id_values],
            axis=0,
        )
        logger.info(f"variables_array complete {variables_array.shape}")

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

    tot_lenght = len(variables[0])
    logger.info(f"variables length {tot_lenght}")

    logger.info(f"number of events (class label {class_label}): {variables.shape[1]}")

    flag_tensor = torch.full(
        (1, variables.shape[1]), float(class_label), dtype=torch.float32
    )

    # shuffle the variables
    idx = np.random.permutation(tot_lenght)
    variables = variables[:, idx]

    # get the kl, the dataset index and the k-fold index and remove them from
    # the variables. They are kept together as a (3, num_events) tensor of
    # [kl, dataset_idx, fold_id]
    kl_values = variables[-3:]
    variables = variables[:-3]

    X = (variables, flag_tensor, kl_values)
    return X, tot_lenght


def _find_class_files(cfg, class_def):
    """Return (coffea_files, parquet_dirs) for a given class definition."""
    coffea_files = []
    parquet_files = []

    if cfg.data_format == "root":
        for x in cfg.data_dirs:
            files = sorted(os.listdir(x))
            for file in files:
                for sample in class_def["samples"]:
                    if sample in file and "SR" in file:
                        coffea_files.append(x + file)
    elif cfg.data_format in ("coffea", "parquet"):
        for direct in cfg.data_dirs:
            if not os.path.isdir(direct):
                raise FileNotFoundError(f"Data directory not found: {direct}")
            for file in sorted(os.listdir(direct)):
                if file.endswith(".coffea"):
                    if any(d in file for d in class_def["datasets"]):
                        coffea_files.append(os.path.join(direct, file))

        if cfg.data_format == "parquet":
            parquet_dir = cfg.get("parquet_dir", None)
            if parquet_dir is not None and len(cfg.data_dirs) != 1:
                raise ValueError("parquet_dir can be used only with a single data_dir")
            for direct in cfg.data_dirs:
                if parquet_dir is not None:
                    # e.g. the files merged with merge_parquet.py of AnalysisConfigs
                    parquet_dirs_path = parquet_dir
                    logger.info(
                        f"Reading the parquet files from parquet_dir {parquet_dirs_path} "
                        f"(instead of the save_chunk directory in {direct}/config.json)"
                    )
                else:
                    with open(f"{direct}/config.json", "r") as f:
                        config = json.load(f)
                    parquet_dirs_path = root_to_local(
                        config["workflow"]["workflow_options"]["save_chunk"]
                    )
                    logger.info(
                        f"Reading the parquet files from {parquet_dirs_path} "
                        f"(save_chunk directory in {direct}/config.json)"
                    )
                if not os.path.isdir(parquet_dirs_path):
                    raise FileNotFoundError(
                        f"Local path not found on this node: {parquet_dirs_path}"
                    )
                # sorted, so that the files are always loaded in the same order
                for entry in sorted(
                    os.scandir(parquet_dirs_path), key=lambda e: e.name
                ):
                    logger.debug(f"Looking for files in {entry.path}")
                    if entry.name in class_def["datasets"]:
                        for region in class_def["regions"]:
                            region_path = entry.path + "/" + region
                            # with variations saved, read only the nominal one
                            if not cfg.novars:
                                region_path += "/nominal"
                                if not os.path.isdir(region_path):
                                    raise FileNotFoundError(
                                        f"Nominal variation not found: {region_path}. "
                                        "Set novars: True if the parquet files are "
                                        "saved without variations."
                                    )
                            parquet_files.append(region_path)
    else:
        raise ValueError(f"Data format {cfg.data_format} not supported")

    return coffea_files, parquet_files


def load_data(cfg, seed):
    batch_size = cfg.batch_size
    logger.debug(f"Batch size: {batch_size}")

    # initialize numpy seed
    np.random.seed(int(seed))

    total_fraction_of_events = cfg.train_fraction + cfg.val_fraction + cfg.test_fraction

    assert total_fraction_of_events <= 1.0, "Fractions must sum to less than 1.0"

    k_folds = cfg.k_folds or 0
    kfold = k_folds >= 2
    prepare_only = bool(cfg.get("kfold_prepare", False))
    if kfold:
        check_kfold_cfg(cfg, need_fold=not prepare_only)
        logger.info(
            f"K-fold training: fold {cfg.fold} of {k_folds}, split {cfg.kfold_split}"
        )
        cache_dir = cfg.get("kfold_cache", None)
        if cache_dir and kfold_cache_exists(cache_dir):
            if prepare_only:
                check_kfold_cache(cache_dir, cfg)
                logger.info(f"K-fold cache {cache_dir} already prepared")
                return None
            return load_data_from_kfold_cache(cfg, seed)
    elif prepare_only:
        raise ValueError("kfold_prepare needs k_folds >= 2")
    event_column = (
        cfg.kfold_event_column if kfold and cfg.kfold_split == "event" else None
    )

    logger.debug("Variables: %s", cfg.input_variables)

    class_defs = parse_classes(cfg)
    num_classes = len(class_defs)
    logger.info(f"Number of classes: {num_classes}")
    for c in class_defs:
        logger.info(
            f"  class idx={c['class_idx']} name={c['name']} lbl={c['lbl']} "
            f"samples={c['samples']} datasets={c['datasets']} regions={c['regions']}"
        )

    # Load each class separately.
    X_per_class = []
    n_per_class = []
    for c in class_defs:
        coffea_files, parquet_files = _find_class_files(cfg, c)
        logger.info(
            f"Class {c['name']} (idx={c['class_idx']}): coffea files {coffea_files}"
        )
        if cfg.data_format == "parquet":
            logger.info(
                f"Class {c['name']} (idx={c['class_idx']}): parquet files {parquet_files}"
            )
        X_class, n_class = get_variables(
            coffea_files,
            parquet_files,
            total_fraction_of_events,
            cfg.input_variables,
            c["samples"],
            c["datasets"],
            c["regions"],
            c["class_idx"],
            cfg.data_format,
            cfg.preprocess_variables_functions,
            cfg.novars,
            cfg.pad_value,
            cfg.normalize_weights_per_dataset,
            event_column,
            k_folds,
        )
        if X_per_class and X_class[0].shape[0] != X_per_class[0][0].shape[0]:
            raise ValueError(
                f"Class {c['name']} has {X_class[0].shape[0] - 1} input features "
                f"but class {class_defs[0]['name']} has "
                f"{X_per_class[0][0].shape[0] - 1}. This can happen when a whole "
                "jagged collection is used as input with coffea files, since it "
                "is padded to the maximum number of objects of each class: "
                "select the objects explicitly as <variable>:<position>."
            )
        X_per_class.append(X_class)
        n_per_class.append(n_class)
        logger.info(f"Number of events for class {c['name']}: {n_class}")

    if cfg.oversample_split + cfg.split_oversample + cfg.undersample > 1:
        raise ValueError("Select only oversample or undersample")

    # For binary backward compatibility, keep the legacy under/oversampling
    # behaviour (signal/background). For multiclass we do not yet support these
    # operations.
    is_binary = num_classes == 2
    if cfg.undersample and not is_binary:
        raise ValueError(
            "undersample is only supported for binary classification"
        )

    if cfg.undersample and is_binary:
        # class_idx convention: 0=bkg, 1=signal
        X_bkg = X_per_class[0]
        X_sig = X_per_class[1]
        logger.info("Performing undersampling of background")
        logger.info(
            f"Number of background events before undersampling {X_bkg[0].shape[1]}"
        )
        num_events_sig = X_sig[0].shape[1]
        X_bkg_f = X_bkg[0][:, :num_events_sig]
        X_bkg_l = X_bkg[1][:, :num_events_sig]
        X_bkg_k = X_bkg[2][:, :num_events_sig]
        X_per_class[0] = (X_bkg_f, X_bkg_l, X_bkg_k)
        logger.info(
            f"Number of background events after undersampling {X_per_class[0][0].shape[1]}"
        )

    # with the k-folding the oversampling is always done after the split into
    # folds (see split_kfold), otherwise copies of the same event would end up
    # both in the training and test datasets
    if cfg.oversample_split and not kfold:
        if is_binary:
            X_bkg = X_per_class[0]
            X_sig = X_per_class[1]
            logger.info("Performing oversampling of signal before splitting")
            num_events_sig = X_sig[0].shape[1]
            num_events_bkg = X_bkg[0].shape[1]
            repeat = num_events_bkg // num_events_sig + 1
            X_sig_f = X_sig[0].repeat((1, repeat))[:, :num_events_bkg]
            X_sig_l = X_sig[1].repeat((1, repeat))[:, :num_events_bkg]
            X_sig_k = X_sig[2].repeat((1, repeat))[:, :num_events_bkg]
            X_per_class[1] = (X_sig_f, X_sig_l, X_sig_k)
            if num_events_sig > num_events_bkg:
                raise ValueError(
                    "Number of signal events is greater than number of background events."
                )
            logger.info(
                f"Number of signal events after oversampling {X_per_class[1][0].shape[1]}"
            )
        else:
            n_per_class_current = [X[0].shape[1] for X in X_per_class]
            n_max = max(n_per_class_current)
            logger.info(f"Performing oversampling of all classes to {n_max} events (multiclass)")
            for i, (c, X) in enumerate(zip(class_defs, X_per_class)):
                n = X[0].shape[1]
                if n == n_max:
                    logger.info(f"Class {c['name']}: no oversampling needed ({n} events)")
                    continue
                repeat = n_max // n+1
                X_f = X[0].repeat((1, repeat))[:, :n_max]
                X_l = X[1].repeat((1, repeat))[:, :n_max]
                X_k = X[2].repeat((1, repeat))[:, :n_max]
                X_per_class[i] = (X_f, X_l, X_k)
                logger.info(
                    f"Class {c['name']}: oversampled from {n} to {n_max} events"
                )

    # Compute per-class weights so that the sum of weights is the same across
    # all classes (and per-class sum equals N / num_classes). This generalises
    # the previous binary balancing formula
    # ``(n_sig + n_bkg) / (2 * sumw_class)`` to arbitrary number of classes.
    n_per_class = [X[0].shape[1] for X in X_per_class]
    n_total = sum(n_per_class)
    sumw_per_class = [float(X[0][-1].sum()) for X in X_per_class]
    for c, n, sumw in zip(class_defs, n_per_class, sumw_per_class):
        logger.info(
            f"Class {c['name']} (idx={c['class_idx']}): n_events={n}, sum_weights={sumw}"
        )

    balance_weights = [n_total / (num_classes * sw) for sw in sumw_per_class]
    if not cfg.oversample_split and not cfg.split_oversample and not cfg.undersample:
        class_weights = balance_weights
    else:
        class_weights = [1.0 for _ in range(num_classes)]

    for c, cw in zip(class_defs, class_weights):
        logger.info(f"Class {c['name']} class_weight: {cw}")

    # Per-event class-weight tensors, ready for concatenation.
    clsw_tensors = [
        (torch.ones_like(X[0][-1], dtype=torch.float32) * w).unsqueeze(0)
        for X, w in zip(X_per_class, class_weights)
    ]
    for c, X, w in zip(class_defs, X_per_class, class_weights):
        rescaled = X[0][-1] * w
        logger.info(
            f"Class {c['name']} sum of weights after rescaling: {float(rescaled.sum())}"
        )

    X_fts = torch.cat([X[0] for X in X_per_class], dim=1).transpose(1, 0)
    X_lbl = torch.cat([X[1] for X in X_per_class], dim=1).transpose(1, 0).flatten()
    X_clsw = torch.cat(clsw_tensors, dim=1).transpose(1, 0)
    # (num_events, 3) tensor of [kl, dataset_idx, fold_id]
    X_k = torch.cat([X[2] for X in X_per_class], dim=1).transpose(1, 0)
    # (num_events, 1) tensor of the class weights that balance the classes,
    # used for the (never oversampled) test fold of the k-fold training
    X_clsw_bal = torch.cat(
        [
            torch.full((1, X[0].shape[1]), w, dtype=torch.float32)
            for X, w in zip(X_per_class, balance_weights)
        ],
        dim=1,
    ).transpose(1, 0)
    # free the per-class copies of the events
    del X_per_class, clsw_tensors

    logger.info(f"X_fts shape: {X_fts.shape}")
    logger.info(f"X_lbl shape: {X_lbl.shape}")
    logger.info(f"X_clsw shape: {X_clsw.shape}")
    logger.info(f"X_k shape: {X_k.shape}")

    tot_num_events = X_fts.shape[0]

    # shuffle the tensor with numpy random
    idx = np.random.permutation(tot_num_events)
    X_fts = X_fts[idx]
    X_lbl = X_lbl[idx]
    X_clsw = X_clsw[idx]
    X_k = X_k[idx]
    X_clsw_bal = X_clsw_bal[idx]

    if kfold:
        X_k = assign_fold_ids(X_k, cfg)
        arrays = {
            "fts": X_fts,
            "lbl": X_lbl,
            "clsw": X_clsw,
            "clsw_bal": X_clsw_bal,
            "k": X_k,
        }
        meta = {
            "input_size": X_fts.size(1) - 1,
            "num_classes": num_classes,
            "is_binary": is_binary,
            "n_events": tot_num_events,
            "class_info": get_class_info(class_defs),
        }
        cache_dir = cfg.get("kfold_cache", None)
        if cache_dir:
            save_kfold_cache(cache_dir, arrays, meta, cfg)
            if prepare_only:
                return None
            # use the memory-mapped cache, shared with the other folds
            del arrays, X_fts, X_lbl, X_clsw, X_clsw_bal, X_k
            return load_data_from_kfold_cache(cfg, seed)
        return make_kfold_loaders(cfg, arrays, meta, seed)

    train_size = math.floor(tot_num_events * cfg.train_fraction)
    val_size = math.floor(tot_num_events * cfg.val_fraction)
    test_size = math.floor(tot_num_events * cfg.test_fraction)

    tot_events = train_size + val_size + test_size

    # keep only total_fraction_of_events
    X_fts = X_fts[:tot_events]
    X_lbl = X_lbl[:tot_events]
    X_clsw = X_clsw[:tot_events]
    X_k = X_k[:tot_events]

    X = torch.utils.data.TensorDataset(X_fts, X_lbl, X_clsw, X_k)

    logger.info(f"Total size: {len(X)}")
    logger.info(f"Training size: {train_size}")
    logger.info(f"Validation size: {val_size}")
    logger.info(f"Test size: {test_size}")

    # shuffle and split
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        X, [train_size, val_size, test_size], generator=gen
    )

    if cfg.split_oversample:
        logger.info("Performing oversampling of signal after splitting")
        train_dataset = oversample_dataset(train_dataset, is_binary)
        val_dataset = oversample_dataset(val_dataset, is_binary)
        test_dataset = oversample_dataset(test_dataset, is_binary)

    class_names = [c["name"] for c in class_defs]
    for name, dataset in [
        ("Training", train_dataset),
        ("Validation", val_dataset),
        ("Test", test_dataset),
    ]:
        log_class_balance(name, *_balance_columns(dataset), class_names)

    # remove 1 because of the weights
    input_size = X_fts.size(1) - 1

    return make_loaders(
        cfg,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        input_size,
        num_classes,
        class_defs,
    )


def make_loaders(
    cfg,
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    input_size,
    num_classes,
    class_defs,
    drop_last_test=True,
):
    training_loader = None
    val_loader = None
    test_loader = None

    # with shuffle the RandomSampler draws the permutation of every epoch from
    # the global torch generator, seeded with cfg.seed (set_seed)
    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=cfg.shuffle_train,
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
            drop_last=drop_last_test,
            pin_memory=cfg.pin_memory,
        )
        logger.info("Test loader size: %d", len(test_loader))

    class_info = get_class_info(class_defs)

    return (
        training_loader,
        val_loader,
        test_loader,
        input_size,
        batch_size,
        num_classes,
        class_info,
    )


def get_class_info(class_defs):
    return [
        {
            "name": c["name"],
            "lbl": c["lbl"],
            "class_idx": c["class_idx"],
            "datasets": list(c["datasets"]),
        }
        for c in class_defs
    ]


def check_kfold_cfg(cfg, need_fold=True):
    if need_fold and (cfg.fold is None or not 0 <= cfg.fold < cfg.k_folds):
        raise ValueError(
            f"fold must be between 0 and k_folds - 1 = {cfg.k_folds - 1}, got {cfg.fold}"
        )
    if cfg.kfold_split not in ["random", "event"]:
        raise ValueError(
            f"kfold_split must be random or event, got {cfg.kfold_split}"
        )
    if cfg.train_fraction <= 0 or cfg.val_fraction <= 0:
        raise ValueError(
            "train_fraction and val_fraction must be positive with the k-folding"
        )
    check_kfold_onnx_cfg(cfg)


def check_kfold_onnx_cfg(cfg):
    if cfg.kfold_onnx not in ["average", "event"]:
        raise ValueError(f"kfold_onnx must be average or event, got {cfg.kfold_onnx}")
    if cfg.kfold_onnx == "event" and cfg.kfold_split != "event":
        raise ValueError(
            "kfold_onnx: event (the model chosen by event number) needs "
            "kfold_split: event"
        )


def assign_fold_ids(X_k, cfg):
    """Store in the third column of X_k the fold of each (already shuffled)
    event: event % k_folds (computed when loading) or, for kfold_split
    random, the position in the shuffled order % k_folds. Since the shuffle
    depends only on the seed, the folds are the same for all the folds."""
    if cfg.kfold_split == "event":
        fold_id = X_k[:, 2].to(torch.int64)
        if (fold_id < 0).any():
            raise ValueError("Missing event numbers for kfold_split event")
    else:
        fold_id = torch.arange(X_k.shape[0]) % cfg.k_folds
        X_k = X_k.clone()
        X_k[:, 2] = fold_id.to(X_k.dtype)
    for k in range(cfg.k_folds):
        logger.info(f"Fold {k}: {int((fold_id == k).sum())} events")
    return X_k


def oversample_indices(idx, labels, is_binary, rng):
    """Oversample (by repeating the indices) the minority class(es) of the
    events ``idx``, as oversample_dataset does with the events themselves,
    and shuffle them with the numpy generator ``rng``."""
    lbl = labels[idx]
    classes = np.unique(lbl)
    n_per_class = {c: int((lbl == c).sum()) for c in classes}
    if is_binary:
        if n_per_class.get(1, 0) > n_per_class.get(0, 0):
            raise ValueError(
                "Number of signal events is greater than number of background events."
            )
        to_oversample = [1]
        n_target = n_per_class.get(0, 0)
    else:
        to_oversample = list(classes)
        n_target = max(n_per_class.values())
    out = []
    for c in classes:
        idx_c = idx[lbl == c]
        if c in to_oversample and 0 < len(idx_c) < n_target:
            factor = n_target // len(idx_c) + 1
            logger.info(
                f"Class {int(c)}: oversampling {len(idx_c)} -> {n_target} (factor {factor})"
            )
            idx_c = np.tile(idx_c, factor)[:n_target]
        out.append(idx_c)
    out = np.concatenate(out)
    return out[rng.permutation(len(out))]


def kfold_indices(fold_id, labels, cfg, seed, is_binary):
    """Indices of the training, validation and test events of the fold
    ``cfg.fold``.

    The test events are the events of the fold, the other events are split
    into training and validation with the ratio train_fraction : val_fraction.
    The test events are never oversampled, so every event gets exactly one
    out-of-fold score and copies of an event never end up in the test dataset.
    The other events are oversampled:
      oversample_split: before the split into training and validation, so
        copies of the same event can be both in the training and validation
        datasets;
      split_oversample: after the split, separately in the training and
        validation datasets.
    """
    fold = cfg.fold
    test_idx = np.nonzero(fold_id == fold)[0]
    other_idx = np.nonzero(fold_id != fold)[0]

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    other_idx = other_idx[torch.randperm(len(other_idx), generator=gen).numpy()]
    # own generator for the oversampling, so that the folds do not depend on
    # the global random state (the same with and without the k-fold cache)
    rng = np.random.default_rng([int(seed), fold])

    logger.info(f"Total size: {len(fold_id)}")
    logger.info(f"Test size (fold {fold}, never oversampled): {len(test_idx)}")
    logger.info(f"Training + validation size (other folds): {len(other_idx)}")

    if cfg.oversample_split:
        logger.info(
            "Performing oversampling of the events of the other folds before "
            "the split into training and validation (copies of the same event "
            "can be both in the training and validation datasets, not in the "
            "test fold)"
        )
        other_idx = oversample_indices(other_idx, labels, is_binary, rng)
        logger.info(
            f"Training + validation size after oversampling: {len(other_idx)}"
        )

    train_size = round(
        len(other_idx) * cfg.train_fraction / (cfg.train_fraction + cfg.val_fraction)
    )
    train_idx = other_idx[:train_size]
    val_idx = other_idx[train_size:]
    logger.info(f"Training size: {len(train_idx)}")
    logger.info(f"Validation size: {len(val_idx)}")

    if cfg.split_oversample:
        logger.info(
            "Performing oversampling after the split into training and "
            "validation, separately in the two datasets (not in the test fold)"
        )
        train_idx = oversample_indices(train_idx, labels, is_binary, rng)
        val_idx = oversample_indices(val_idx, labels, is_binary, rng)
        logger.info(f"Training size after oversampling: {len(train_idx)}")
        logger.info(f"Validation size after oversampling: {len(val_idx)}")

    return train_idx, val_idx, test_idx


def log_class_balance(name, labels, event_weights, class_weights, class_names, event_ids=None):
    """Log, for each class of a dataset, the number of events (and of distinct
    events, if their ids are given, e.g. with oversampling), the sum of the
    event weights, the class weight and the sum of the weights used in the
    loss (event weight * class weight, negative weights clamped to 0 as in the
    training loop), so that the balance of the classes can be checked."""
    labels = np.asarray(labels).astype(np.int64).flatten()
    event_weights = np.asarray(event_weights, dtype=np.float64).flatten()
    class_weights = np.asarray(class_weights, dtype=np.float64).flatten()
    loss_weights = np.clip(event_weights * class_weights, 0, None)
    tot_events = len(labels)
    tot_loss_weights = loss_weights.sum()
    logger.info(f"{name} dataset: {tot_events} events, class balance:")
    sums = []
    for c, class_name in enumerate(class_names):
        mask = labels == c
        n = int(mask.sum())
        w = event_weights[mask]
        lw = loss_weights[mask]
        cw = np.unique(class_weights[mask])
        sums.append(lw.sum())
        unique = ""
        if event_ids is not None:
            # the weights of the copies of an event (oversampling) are summed,
            # since the copies are not independent events
            ids, inverse = np.unique(np.asarray(event_ids)[mask], return_inverse=True)
            lw_distinct = np.bincount(inverse, weights=lw, minlength=len(ids))
            unique = f", distinct {len(ids)}"
        else:
            lw_distinct = lw
        # effective number of independent events of the weights used in the loss
        ess = (
            lw_distinct.sum() ** 2 / (lw_distinct**2).sum()
            if n and (lw_distinct**2).sum() > 0
            else 0.0
        )
        logger.info(
            f"  {name} class {c} ({class_name}): events {n} "
            f"({n / max(tot_events, 1):.2%}){unique}, "
            f"sum event weights {w.sum():.6g} (mean {w.mean() if n else 0:.4g}, "
            f"negative {int((w < 0).sum())}), "
            f"class weight {', '.join(f'{x:.6g}' for x in cw)}, "
            f"sum loss weights {lw.sum():.6g} "
            f"({lw.sum() / tot_loss_weights if tot_loss_weights else 0:.2%}), "
            f"effective events {ess:.6g}"
        )
    nonzero = [x for x in sums if x > 0]
    if len(nonzero) > 1:
        logger.info(
            f"  {name}: ratio max / min of the sum of the loss weights of the "
            f"classes {max(nonzero) / min(nonzero):.4f} (1 = balanced)"
        )


def log_overlap(train_idx, val_idx, test_idx):
    """Log how many distinct events are shared by the k-fold datasets."""
    train_u, val_u, test_u = (np.unique(i) for i in (train_idx, val_idx, test_idx))
    logger.info(
        f"Distinct events shared by training and validation: "
        f"{np.intersect1d(train_u, val_u, assume_unique=True).size}, by training "
        f"and test: {np.intersect1d(train_u, test_u, assume_unique=True).size}, "
        f"by validation and test: "
        f"{np.intersect1d(val_u, test_u, assume_unique=True).size}"
    )


def make_kfold_loaders(cfg, arrays, meta, seed):
    """Loaders of the fold ``cfg.fold``, taking the batches of events from the
    ``arrays`` (in-memory tensors or memory-mapped arrays of the k-fold cache)
    by index."""
    labels = np.asarray(arrays["lbl"])
    fold_id = np.asarray(arrays["k"][:, 2]).astype(np.int64)
    train_idx, val_idx, test_idx = kfold_indices(
        fold_id, labels, cfg, seed, meta["is_binary"]
    )
    class_names = [c["name"] for c in meta["class_info"]]
    event_weights = np.asarray(arrays["fts"][:, -1])
    clsw = np.asarray(arrays["clsw"][:, 0])
    clsw_bal = np.asarray(arrays["clsw_bal"][:, 0])
    for name, idx, cw in [
        ("Training", train_idx, clsw),
        ("Validation", val_idx, clsw),
        ("Test", test_idx, clsw_bal),
    ]:
        log_class_balance(
            name,
            labels[idx],
            event_weights[idx],
            cw[idx],
            class_names,
            event_ids=idx,
        )
    log_overlap(train_idx, val_idx, test_idx)
    batch_size = cfg.batch_size
    sources = [arrays["fts"], arrays["lbl"], arrays["clsw"], arrays["k"]]
    # the test events get the class weights that balance the classes, since
    # they are never oversampled
    test_sources = [arrays["fts"], arrays["lbl"], arrays["clsw_bal"], arrays["k"]]

    training_loader = IndexedBatchLoader(
        sources, train_idx, batch_size, drop_last=True, shuffle=cfg.shuffle_train
    )
    logger.info("Training loader size: %d", len(training_loader))
    val_loader = None
    if not cfg.eval_model:
        val_loader = IndexedBatchLoader(sources, val_idx, batch_size, drop_last=True)
        logger.info("Validation loader size: %d", len(val_loader))
    test_loader = None
    if cfg.eval or cfg.eval_model:
        # every event of the test fold must get a score
        test_loader = IndexedBatchLoader(
            test_sources, test_idx, batch_size, drop_last=False
        )
        logger.info("Test loader size: %d", len(test_loader))
        # rows of the test events, to check in the merge that the test folds
        # are disjoint and cover all the events
        test_loader.event_rows = test_idx
        test_loader.n_events_total = meta["n_events"]

    return (
        training_loader,
        val_loader,
        test_loader,
        meta["input_size"],
        batch_size,
        meta["num_classes"],
        meta["class_info"],
    )


# ---------------------------------------------------------------------------
# K-fold cache: the events are loaded, shuffled and assigned to the folds only
# once, and saved as .npy files that all the folds memory-map. The pages of a
# memory-mapped file are shared by all the processes of a node, so the events
# take the RAM only once, however many folds run in parallel.
# ---------------------------------------------------------------------------

KFOLD_CACHE_ARRAYS = ["fts", "lbl", "clsw", "clsw_bal", "k"]
KFOLD_CACHE_META = "kfold_cache_meta.json"

# config options that change the content of the k-fold cache
KFOLD_CACHE_KEYS = [
    "input_variables",
    "preprocess_variables_functions",
    "run2",
    "signal_sample",
    "background_sample",
    "signal_region",
    "background_region",
    "signal_dataset",
    "background_dataset",
    "classes",
    "data_dirs",
    "parquet_dir",
    "data_format",
    "novars",
    "train_fraction",
    "val_fraction",
    "test_fraction",
    "normalize_weights_per_dataset",
    "oversample_split",
    "split_oversample",
    "undersample",
    "seed",
    "k_folds",
    "kfold_split",
    "kfold_event_column",
    "pad_value",
]


def kfold_cache_hash(cfg):
    from omegaconf import OmegaConf

    values = {
        k: OmegaConf.to_container(cfg[k], resolve=True)
        if OmegaConf.is_config(cfg.get(k, None))
        else cfg.get(k, None)
        for k in KFOLD_CACHE_KEYS
    }
    # the seed can be given as a string from the command line
    values["seed"] = int(values["seed"])
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode()
    ).hexdigest()


def kfold_cache_exists(cache_dir):
    return os.path.exists(os.path.join(cache_dir, KFOLD_CACHE_META))


def check_kfold_cache(cache_dir, cfg):
    with open(os.path.join(cache_dir, KFOLD_CACHE_META)) as f:
        meta = json.load(f)
    if meta["hash"] != kfold_cache_hash(cfg):
        raise ValueError(
            f"The k-fold cache {cache_dir} was made with a different config "
            "(data, fractions, oversampling, seed, k_folds or kfold_split): "
            "remove it or use another kfold_cache directory"
        )
    return meta


def save_kfold_cache(cache_dir, arrays, meta, cfg):
    os.makedirs(cache_dir, exist_ok=True)
    for name in KFOLD_CACHE_ARRAYS:
        np.save(os.path.join(cache_dir, f"{name}.npy"), np.asarray(arrays[name]))
    meta = dict(meta, hash=kfold_cache_hash(cfg))
    # written last: its presence marks a complete cache
    with open(os.path.join(cache_dir, KFOLD_CACHE_META), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved the k-fold cache of {meta['n_events']} events in {cache_dir}")

    # record of the split ([kl, dataset_idx, fold] of every event and the
    # metadata), kept for reproducibility when the cache itself is in tmp
    split_dir = cfg.get("kfold_split_dir", None)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, "k.npy"), np.asarray(arrays["k"]))
        with open(os.path.join(split_dir, KFOLD_CACHE_META), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved the record of the k-fold split in {split_dir}")


def kfold_test_info(test_idx, n_events):
    """Number and fingerprint of the test events of a fold, to check in the
    merge that the folds were trained with the expected split."""
    rows = np.sort(np.asarray(test_idx, dtype=np.int64))
    return {
        "n_events_total": int(n_events),
        "n_test": int(len(rows)),
        "test_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
    }


def open_kfold_cache(cache_dir, cfg):
    """Memory-mapped arrays and metadata of the k-fold cache."""
    meta = check_kfold_cache(cache_dir, cfg)
    logger.info(f"Reading the events from the k-fold cache {cache_dir}")
    arrays = {
        name: np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r")
        for name in KFOLD_CACHE_ARRAYS
    }
    return arrays, meta


def load_data_from_kfold_cache(cfg, seed):
    arrays, meta = open_kfold_cache(cfg.kfold_cache, cfg)
    return make_kfold_loaders(cfg, arrays, meta, seed)


def root_to_local(path_or_url: str):
    """Turn 'root://host:port//abs/path' into '/abs/path'. Leave local paths unchanged."""
    if path_or_url.startswith("root://"):
        u = urlsplit(
            path_or_url
        )  # scheme='root', netloc='host:port', path='//abs/path'
        p = u.path
        while p.startswith("//"):  # normalize to single leading slash
            p = p[1:]
        if not p.startswith("/"):
            p = "/" + p
        return p
    return path_or_url


def extract_param_value(s, param):
    """
    Extract a parameter value from a string.

    Parameters
    ----------
    s : str
        Input string.
    param : str
        Parameter name to extract (e.g. 'kl', 'CV', 'C2V', 'C3', 'kt').

    Returns
    -------
    float or None
        Extracted value or None if not found.
    """

    # pattern allows "-" or "_" after param name
    pattern = rf"{param}[-_]([mp0-9]+)"

    match = re.search(pattern, s)
    if not match:
        return None

    value_str = match.group(1)

    # convert encoding: m -> -, p -> .
    value_str = value_str.replace("m", "-").replace("p", ".")

    return float(value_str)


def _balance_columns(dataset):
    """Labels, event weights and class weights of a TensorDataset, or of a
    (nested) Subset of one, gathering only these columns."""
    if isinstance(dataset, torch.utils.data.TensorDataset):
        fts, lbl, clsw = dataset.tensors[:3]
        return lbl, fts[:, -1], clsw[:, 0]
    if isinstance(dataset, torch.utils.data.Subset):
        idx = torch.as_tensor(dataset.indices, dtype=torch.long)
        return tuple(t[idx] for t in _balance_columns(dataset.dataset))
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def _dataset_tensors(dataset):
    """Return the tensors of a TensorDataset, or of a (nested) Subset of one,
    or of an IndexedBatchLoader, in the order of the dataset."""
    if isinstance(dataset, IndexedBatchLoader):
        return dataset.gather()
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return list(dataset.tensors)
    if isinstance(dataset, torch.utils.data.Subset):
        idx = torch.as_tensor(dataset.indices, dtype=torch.long)
        return [t[idx] for t in _dataset_tensors(dataset.dataset)]
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


class IndexedBatchLoader:
    """Loader taking the batches of the events ``indices`` from ``sources``
    (tensors or numpy arrays, also memory-mapped, with the events on the
    first axis), in the order of ``indices``.

    Used by the k-fold training, where the events of a fold are a subset (with
    repetitions when oversampling) of all the events: the events are not
    copied, only gathered batch by batch. The next batch is prepared in a
    background thread while the current one is used. With ``shuffle`` the
    order of the events is permuted at every epoch with the global torch
    generator (seeded with the seed of the training).
    """

    def __init__(self, sources, indices, batch_size, drop_last=True, shuffle=False):
        self.sources = sources
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_events = len(self.indices)

    def __len__(self):
        if self.drop_last:
            return self.num_events // self.batch_size
        return math.ceil(self.num_events / self.batch_size)

    def num_bytes(self):
        return self.num_events * sum(
            np.dtype(np.asarray(s[:1]).dtype).itemsize * int(np.prod(s.shape[1:]))
            for s in self.sources
        )

    def _take(self, idx):
        return tuple(
            torch.as_tensor(np.asarray(s[idx]))
            if isinstance(s, np.ndarray)
            else s[torch.from_numpy(idx)]
            for s in self.sources
        )

    def gather(self):
        """All the events of the loader, in its order."""
        return list(self._take(self.indices))

    @property
    def dataset(self):
        """The events of the loader as a TensorDataset (copied in memory)."""
        return torch.utils.data.TensorDataset(*self.gather())

    def _batches(self, indices):
        for i in range(len(self)):
            start = i * self.batch_size
            yield self._take(indices[start : start + self.batch_size])

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            # drawn here (main thread) so that the generator state does not
            # depend on the background thread
            indices = indices[torch.randperm(self.num_events).numpy()]
        batches = queue.Queue(maxsize=2)
        done = object()

        def produce():
            try:
                for batch in self._batches(indices):
                    batches.put(batch)
            except BaseException as e:
                batches.put(e)
            batches.put(done)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        while True:
            batch = batches.get()
            if batch is done:
                break
            if isinstance(batch, BaseException):
                raise batch
            yield batch
        thread.join()


class DeviceBatchLoader:
    """Drop-in replacement of the DataLoader that takes the batches as slices
    of the dataset tensors held on `device`.

    On the GPU, the whole dataset is copied once to the device, so there are no
    worker processes, no collation and no host-to-device copy per batch. For
    the small DNNs used here this is the main bottleneck of the training. On
    the CPU (datasets too large for the GPU memory) the batches are zero-copy
    slices of the tensors in RAM, copied to the GPU in the training loop: this
    avoids the per-event indexing and collation of the DataLoader, which
    dominate with large batches. The batches have the same order and the same
    drop_last behaviour as the DataLoader with shuffle=False. With ``shuffle``
    the events are permuted at every epoch with the global torch generator
    (seeded with the seed of the training).
    """

    def __init__(self, dataset, batch_size, device, drop_last=True, shuffle=False):
        # the dataset itself is not kept, so that the original tensors (e.g.
        # of a Subset, which are copied here) can be freed
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.tensors = [t.to(device) for t in _dataset_tensors(dataset)]
        self.num_events = self.tensors[0].shape[0]

    def __len__(self):
        if self.drop_last:
            return self.num_events // self.batch_size
        return math.ceil(self.num_events / self.batch_size)

    def __iter__(self):
        perm = None
        if self.shuffle:
            perm = torch.randperm(self.num_events).to(self.tensors[0].device)
        for i in range(len(self)):
            start = i * self.batch_size
            if perm is None:
                yield tuple(t[start : start + self.batch_size] for t in self.tensors)
            else:
                idx = perm[start : start + self.batch_size]
                yield tuple(t[idx] for t in self.tensors)


def _loader_shuffle(loader):
    """Whether the loader reshuffles the events at every epoch."""
    if isinstance(loader, torch.utils.data.DataLoader):
        return isinstance(loader.sampler, torch.utils.data.RandomSampler)
    return getattr(loader, "shuffle", False)


def loaders_to_device(loaders, device, max_memory_fraction=0.5):
    """Replace the DataLoaders with DeviceBatchLoaders holding the data on
    `device`.

    The loaders are left unchanged if the device is not a GPU. If the datasets
    would take more than `max_memory_fraction` of the free GPU memory, they are
    kept in RAM and the batches are sliced there and copied to the GPU one by
    one. `None` loaders are kept as they are.
    """
    if device.type != "cuda":
        return loaders

    def num_bytes(loader):
        if isinstance(loader, IndexedBatchLoader):
            return loader.num_bytes()
        return sum(
            t.element_size() * t.nelement() for t in _dataset_tensors(loader.dataset)
        )

    datasets_bytes = sum(num_bytes(loader) for loader in loaders if loader is not None)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if datasets_bytes > max_memory_fraction * free_bytes:
        logger.warning(
            f"Datasets ({datasets_bytes / 1e9:.2f} GB) do not fit in the free GPU "
            f"memory ({free_bytes / 1e9:.2f} GB): keeping them in RAM and copying "
            "the batches to the GPU one by one"
        )
        device = torch.device("cpu")
        if any(isinstance(loader, IndexedBatchLoader) for loader in loaders):
            # the k-fold loaders already take the batches by index
            return loaders
    else:
        logger.info(
            f"Moving the datasets to {device} ({datasets_bytes / 1e9:.2f} GB)"
        )
    out = []
    for loader in loaders:
        if loader is None:
            out.append(None)
            continue
        new = DeviceBatchLoader(
            loader if isinstance(loader, IndexedBatchLoader) else loader.dataset,
            loader.batch_size,
            device,
            drop_last=loader.drop_last,
            shuffle=_loader_shuffle(loader),
        )
        # keep the rows of the test events of the k-fold training
        for attr in ["event_rows", "n_events_total"]:
            if hasattr(loader, attr):
                setattr(new, attr, getattr(loader, attr))
        out.append(new)
    return out
